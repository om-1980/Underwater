"""Hardware checks for the INSKAM USB camera and DFRobot SEN0598 on Linux.

Place beside config.py. Live mode records AVI, JPEGs and frame-linked sonar telemetry.
By default it tries the project ONNX model. Use --detect to require real inference.
"""
import argparse
import csv
import json
import os
import signal
from datetime import datetime, timezone
from tempfile import mkdtemp
from contextlib import ExitStack
from pathlib import Path
import sys
import time


def decode_reply(reply):
    """Return (distance in cm or None, diagnostic) for a SEN0598 reply."""
    if len(reply) != 4:
        return None, f"No complete UART reply ({len(reply)}/4 bytes)"
    if reply[0] != 0xFF or (sum(reply[:3]) & 0xFF) != reply[3]:
        return None, f"Invalid UART frame ({reply.hex(' ')})"
    mm = int.from_bytes(reply[1:3], "big")
    if mm == 0:
        return None, "Sensor replied: no valid underwater echo"
    if not 20 <= mm <= 3000:
        return None, f"Outside specified range ({mm} mm)"
    return mm / 10, "OK"


class SEN0598:
    def __init__(self, port):
        import serial
        self.serial = serial.Serial(port, 115200, timeout=0.1, write_timeout=0.1)
        self.last_request = None
        time.sleep(0.5)

    def measure(self):
        # Allow at least 100 ms between requests, even if replies arrive quickly.
        now = time.monotonic()
        if self.last_request is not None:
            time.sleep(max(0, 0.1 - (now - self.last_request)))
        self.serial.reset_input_buffer()
        self.last_request = time.monotonic()
        self.serial.write(b"\x55")
        return decode_reply(self.serial.read(4))

    def get_distance(self):
        """Distance in cm, or the existing project's unavailable value."""
        cm, _ = self.measure()
        return -1.0 if cm is None else cm

    def cleanup(self):
        self.serial.close()


def read_frame(camera):
    ok, frame = camera.read()
    if not ok or frame is None or not frame.size:
        raise RuntimeError("Camera returned no frame. Check its V4L2 capture node and USB connection.")
    return frame


class ProjectDetector:
    """Run the project's YOLOv8 ONNX export with its configured thresholds."""
    def __init__(self):
        import config
        import cv2
        self.config = config
        path = config.MODEL_ONNX
        if not path.is_file():
            raise FileNotFoundError(f"Missing detector file: {path}")
        # Both backends receive the same RGB tensor and use the same box decoder.
        try:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
            name = session.get_inputs()[0].name
            self.forward = lambda blob: session.run(None, {name: blob})[0]
            self.backend = "ONNX Runtime"
        except Exception as ort_error:
            try:
                net = cv2.dnn.readNetFromONNX(str(path))

                def forward(blob):
                    net.setInput(blob)
                    return net.forward()

                self.forward = forward
                self.backend = "OpenCV DNN"
            except (cv2.error, AttributeError) as cv_error:
                raise RuntimeError(f"ONNX Runtime: {ort_error}; OpenCV DNN: {cv_error}") from cv_error

    def predict(self, frame):
        import cv2
        import numpy as np
        side = self.config.INFERENCE_IMG_SIZE
        h, w = frame.shape[:2]
        scale = min(side / w, side / h)
        sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
        xpad, ypad = (side - sw) // 2, (side - sh) // 2
        padded = np.full((side, side, 3), 114, dtype=np.uint8)
        padded[ypad:ypad + sh, xpad:xpad + sw] = cv2.resize(frame, (sw, sh))
        blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, swapRB=True)
        rows = self.forward(blob)[0].T
        names = next((classes for classes in (self.config.COCO_CLASSES, self.config.UIEB_CLASSES)
                      if len(classes) == rows.shape[1] - 4), None)
        if names is None:
            raise ValueError(f"Unrecognized YOLO class count: {rows.shape[1] - 4}")
        scores = rows[:, 4:].max(axis=1)
        ids = rows[:, 4:].argmax(axis=1)
        threshold = self.config.CONFIDENCE_THRESHOLD
        hits = []
        for cls in np.unique(ids[scores >= threshold]):
            selected = (scores >= threshold) & (ids == cls)
            boxes = rows[selected, :4].copy()
            boxes[:, :2] -= boxes[:, 2:] * 0.5
            probabilities = scores[selected].tolist()
            kept = cv2.dnn.NMSBoxes(np.rint(boxes).astype(int).tolist(), probabilities, threshold,
                                    self.config.IOU_THRESHOLD)
            for idx in np.asarray(kept, dtype=int).reshape(-1):
                x, y, bw, bh = boxes[idx]
                x1, x2 = np.clip(np.rint((np.array([x, x + bw]) - xpad) * w / sw), 0, w - 1)
                y1, y2 = np.clip(np.rint((np.array([y, y + bh]) - ypad) * h / sh), 0, h - 1)
                if x2 > x1 and y2 > y1:
                    hits.append(dict(label=names[cls], confidence=round(probabilities[idx], 4),
                                     box=list(map(int, [x1, y1, x2, y2]))))
        return hits


def mark_frame(frame, hits, sonar_label, detecting):
    import cv2
    for hit in hits:
        x1, y1, x2, y2 = hit["box"]
        title = f"{hit['label']} {hit['confidence']:.0%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 128), 2)
        at = (max(0, min(x1, frame.shape[1] - 125)), max(18, y1 - 4))
        cv2.putText(frame, title, at, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, title, at, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 1)
    count = f"Objects: {len(hits)}" if detecting else "Detector OFF"
    text = f"{sonar_label} | {count}"
    at = (8, frame.shape[0] - 8)
    for color, weight in [((0, 0, 0), 3), ((255, 255, 255), 1)]:
        cv2.putText(frame, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, weight)
    return text


def record_live(args, camera, sonar, first_frame, stack):
    import cv2
    detector = None
    try:
        detector = ProjectDetector()
        detector.predict(first_frame)  # Check loading AND execution before recording.
    except Exception as error:
        if args.detect:
            raise RuntimeError(f"Model required but unavailable: {error}") from error
        detector = None
        print(f"Detector OFF: {error}\nContinuing camera and sonar recording.", flush=True)
    backend = detector.backend if detector else "OFF"
    args.output.mkdir(parents=True, exist_ok=True)
    folder = Path(mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=args.output))
    (folder / "images").mkdir()
    print(f"Recording to {folder.resolve()} | detector: {backend}", flush=True)
    h, w = first_frame.shape[:2]
    video_fps = 10
    writer = cv2.VideoWriter(str(folder / "live.avi"), cv2.VideoWriter_fourcc(*"MJPG"), video_fps, (w, h))
    stack.callback(writer.release)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open MJPEG video writer in {folder}.")
    log = stack.enter_context((folder / "telemetry.csv").open("w", newline="", encoding="utf-8"))
    csvout = csv.writer(log)
    csvout.writerow(["sample_index", "video_frame_index", "frame_utc", "sonar_utc", "elapsed_s", "sonar_cm",
                     "sonar_status", "model_backend", "detections_json", "image"])
    log.flush()
    start = time.monotonic()
    last_photo = float("-inf")
    previous = set()
    sample_index = 0
    video_index = -1
    previous_frame = None
    first_elapsed = None
    stopping = False

    def request_stop(signum, current_frame):
        nonlocal stopping
        stopping = True

    # Finish the current video/CSV pair before releasing the AVI header.
    old_handler = signal.signal(signal.SIGINT, request_stop)
    stack.callback(signal.signal, signal.SIGINT, old_handler)
    while not stopping:
        cm, detail = sonar.measure()
        sonar_time = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        frame = read_frame(camera)
        frame_time = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        elapsed = time.monotonic() - start
        if frame.shape[:2] != (h, w):
            raise RuntimeError("Camera resolution changed during recording.")
        hits = detector.predict(frame) if detector else []
        sonar_label = detail if cm is None else f"Sonar: {cm:.1f} cm"
        label = mark_frame(frame, hits, sonar_label, detector is not None)
        classes = {hit["label"] for hit in hits}
        photo = ""
        if elapsed - last_photo >= 5 or classes - previous:
            tag = "detection" if hits else "scene"
            photo = f"images/{sample_index:08d}_{tag}.jpg"
            if not cv2.imwrite(str(folder / photo), frame):
                raise OSError(f"Cannot save {folder / photo}")
            last_photo = elapsed
        previous = classes
        # AVI is fixed-rate. Hold the last observation across slow inference
        # intervals so playback follows the acquisition clock without new readings.
        if first_elapsed is None:
            first_elapsed = elapsed
        next_index = max(video_index + 1, round((elapsed - first_elapsed) * video_fps))
        for _ in range(video_index + 1, next_index):
            writer.write(previous_frame)
        writer.write(frame)
        previous_frame, video_index = frame, next_index
        csvout.writerow([sample_index, video_index, frame_time, sonar_time, round(elapsed, 3), cm, detail,
                         backend, json.dumps(hits, separators=(",", ":")), photo])
        log.flush()
        sample_index += 1
        print(f"{label} | saved sample {sample_index}", flush=True)
        if args.preview:
            cv2.imshow("Underwater hardware", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    print(f"Saved {sample_index} readings in {folder.resolve()}", flush=True)


def run(args):
    with ExitStack() as stack:
        if args.mode != "camera":
            sonar = SEN0598(args.port)
            stack.callback(sonar.cleanup)
            if args.mode == "sonar":
                while True:
                    cm, detail = sonar.measure()
                    print(detail if cm is None else f"{cm:.1f} cm", flush=True)
        import cv2
        camera = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        stack.callback(camera.release)
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open {args.camera}. Check v4l2-ctl --list-devices.")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 25)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(3):
            frame = read_frame(camera)
        print(f"Live camera {args.camera}: {frame.shape[1]} x {frame.shape[0]}", flush=True)
        if args.mode == "camera":
            if not cv2.imwrite(str(args.photo), frame):
                raise RuntimeError(f"Cannot save {args.photo}")
            print(f"Camera photo: {args.photo.resolve()}")
            return
        if args.preview and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("No desktop display. Recording with preview off.", flush=True)
            args.preview = False
        if args.preview:
            stack.callback(cv2.destroyAllWindows)
        record_live(args, camera, sonar, frame, stack)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("camera", "sonar", "live"))
    parser.add_argument("--camera", default="/dev/video0", help="V4L2 capture device")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="USB-TTL serial port")
    parser.add_argument("--photo", type=Path, default=Path("camera_test.jpg"))
    parser.add_argument("--preview", action="store_true", help="Show the recording on a desktop")
    parser.add_argument("--detect", action="store_true", help="Require working ONNX inference")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "recordings",
                        help="Parent folder for live sessions")
    args = parser.parse_args()
    if args.mode != "live" and (args.preview or args.detect):
        parser.error("--preview and --detect apply to live mode")
    try:
        run(args)
    except KeyboardInterrupt:
        print("Stopped.")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Hardware check: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
