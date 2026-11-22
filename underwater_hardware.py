"""Hardware checks for the INSKAM USB camera and DFRobot SEN0598 on Linux.

Put this file beside config.py from Underwater.zip to use --detect.
The camera, sonar and live checks work without the original project.
"""
import argparse
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


def run(args):
    with ExitStack() as stack:
        camera = sonar = detector = hud = None
        if args.mode != "sonar":
            import cv2
            camera = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
            stack.callback(camera.release)
            if not camera.isOpened():
                raise RuntimeError(f"Cannot open camera {args.camera}. Check v4l2-ctl --list-devices.")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 25)
            for _ in range(3):
                frame = read_frame(camera)
            print(f"Live camera {args.camera}: {frame.shape[1]} x {frame.shape[0]}", flush=True)
            if args.mode == "camera":
                if not cv2.imwrite(str(args.photo), frame):
                    raise RuntimeError(f"Cannot save {args.photo}")
                print(f"Camera photo: {args.photo.resolve()}")
                return
        if args.detect:
            import config
            from deploy.underwater_pi_fusion import UnderwaterDetector
            from utils.hud_overlay import HUDOverlay
            if not config.MODEL_ONNX.is_file():
                raise RuntimeError(f"Missing project model {config.MODEL_ONNX}")
            detector = UnderwaterDetector(model_path=str(config.MODEL_ONNX))
            if detector.session is None or detector.use_fallback:
                raise RuntimeError("ONNX model did not load. Check onnxruntime or omit --detect.")
            hud = HUDOverlay()
        sonar = SEN0598(args.port)
        stack.callback(sonar.cleanup)
        if args.preview:
            stack.callback(cv2.destroyAllWindows)
        while True:
            cm, detail = sonar.measure()
            label = detail if cm is None else f"{cm:.1f} cm"
            if camera is not None:
                frame = read_frame(camera)
                if detector is not None:
                    detections = detector.predict(frame)
                    frame = hud.draw(frame, detections, -1.0 if cm is None else cm)
                    label += f" | objects: {len(detections)}"
                else:
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 255, 255), 2)
                if args.preview:
                    cv2.imshow("Underwater hardware", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return
                label += f" | live frame {frame.shape[1]}x{frame.shape[0]}"
            print(label, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("camera", "sonar", "live"))
    parser.add_argument("--camera", default="/dev/video0", help="V4L2 capture device")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="USB-TTL serial port")
    parser.add_argument("--photo", type=Path, default=Path("camera_test.jpg"))
    parser.add_argument("--preview", action="store_true", help="Open a desktop window in live mode")
    parser.add_argument("--detect", action="store_true", help="Use the original project's ONNX detector in live mode")
    args = parser.parse_args()
    if args.mode != "live" and (args.preview or args.detect):
        parser.error("--preview and --detect apply to live mode")
    try:
        run(args)
    except KeyboardInterrupt:
        return 0
    except (ImportError, OSError, RuntimeError) as error:
        print(f"Hardware check: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
