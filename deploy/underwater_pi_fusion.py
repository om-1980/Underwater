"""
Underwater Sensor Fusion System for Raspberry Pi
Integrates:
1. Live Camera Stream (Picamera2 / USB / Test Video) with CLAHE underwater contrast enhancement
2. Ultrasonic Distance Sensor (HC-SR04 / JSN-SR04T) with underwater sound speed calibration & median filtering
3. Edge Deep Learning Inference (ONNX / TFLite / YOLOv8n) for marine object recognition
4. Obstacle Proximity Avoidance Decision Engine
5. Real-Time Augmented Reality HUD Telemetry
"""
import os
import sys
import time
import argparse
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import cv2
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from hardware.ultrasonic_sensor import UltrasonicSensor
from hardware.camera_stream import CameraStream
from utils.hud_overlay import HUDOverlay


class UnderwaterDetector:
    """Wrapper for fast model inference on Raspberry Pi (supports ONNX, PyTorch, and CV Fallback)."""
    def __init__(self, model_path: Optional[str] = None, conf_thresh: float = config.CONFIDENCE_THRESHOLD):
        self.conf_thresh = conf_thresh
        self.use_onnx = False
        self.use_fallback = False
        self.session = None
        self.pt_model = None

        resolved_path = None
        if model_path and Path(model_path).exists():
            resolved_path = Path(model_path)
        elif config.MODEL_ONNX.exists():
            resolved_path = config.MODEL_ONNX
        elif config.MODEL_TRAINED_PT.exists():
            resolved_path = config.MODEL_TRAINED_PT

        if resolved_path and str(resolved_path).endswith(".onnx"):
            try:
                import onnxruntime as ort
                self.use_onnx = True
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4  # Optimize for quad-core Raspberry Pi CPU
                self.session = ort.InferenceSession(str(resolved_path), opts, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                self.input_shape = self.session.get_inputs()[0].shape
                print(f"[+] ONNX Runtime initialized with model: {resolved_path}")
            except Exception as e:
                print(f"[!] Warning loading ONNX model: {e}")
                self.use_fallback = True
        else:
            try:
                from ultralytics import YOLO
                target_weights = str(resolved_path) if resolved_path else config.MODEL_BASE
                self.pt_model = YOLO(target_weights)
                print(f"[+] PyTorch YOLO model loaded with {target_weights}")
            except Exception as e:
                print(f"[!] PyTorch/Ultralytics not available in this environment: {e}")
                print("[i] Using Computer Vision Contour/Simulation Detector fallback.")
                self.use_fallback = True

    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs inference on single image frame.
        Returns list of detections: [{'class': name, 'conf': float, 'box': [x1, y1, x2, y2]}]
        """
        detections = []
        h, w, _ = frame.shape

        if self.use_fallback or (not self.use_onnx and self.pt_model is None):
            # Computer Vision foreground/contour detector for simulation and testing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 21, 4)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 400 < area < 40000:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    # Classify roughly by aspect ratio
                    aspect = bw / float(bh)
                    cls_name = "fish" if aspect > 1.2 else "crab"
                    detections.append({
                        "class": cls_name,
                        "conf": round(min(0.95, 0.5 + (area / 80000.0)), 2),
                        "box": [bx, by, bx + bw, by + bh]
                    })
            return detections[:5]  # Cap at top 5

        if self.use_onnx:
            # Preprocess for ONNX
            img_size = config.INFERENCE_IMG_SIZE
            resized = cv2.resize(frame, (img_size, img_size))
            blob = resized.astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)  # HWC to CHW
            blob = np.expand_dims(blob, axis=0)  # Add batch dimension

            # Run ONNX inference
            outputs = self.session.run(None, {self.input_name: blob})
            raw_out = outputs[0][0]  # Shape: (features, anchors) or (anchors, features)

            if raw_out.shape[0] < raw_out.shape[1]:
                raw_out = raw_out.T  # Transpose to (anchors, 4 + num_classes)

            boxes = []
            confidences = []
            class_ids = []

            for row in raw_out:
                box_cx, box_cy, bw, bh = row[:4]
                scores = row[4:]
                cls_id = int(np.argmax(scores))
                score = float(scores[cls_id])

                if score >= self.conf_thresh:
                    # Convert normalized coords back to original frame scale
                    scale_x = w / img_size
                    scale_y = h / img_size

                    x1 = int((box_cx - bw / 2) * scale_x)
                    y1 = int((box_cy - bh / 2) * scale_y)
                    x2 = int((box_cx + bw / 2) * scale_x)
                    y2 = int((box_cy + bh / 2) * scale_y)

                    boxes.append([x1, y1, x2 - x1, y2 - y1])
                    confidences.append(score)
                    class_ids.append(cls_id)

            # Apply Non-Maximum Suppression (NMS)
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, config.IOU_THRESHOLD)
                for idx in indices:
                    i = int(idx) if np.isscalar(idx) or isinstance(idx, (int, np.integer)) else int(idx[0])
                    bx, by, bw, bh = boxes[i]
                    cls_id = class_ids[i]

                    # Match with full real-world classes (person, boat, bottle, chair, laptop, fish, etc.)
                    num_classes = len(scores)
                    if num_classes == len(config.COCO_CLASSES):
                        cls_name = config.COCO_CLASSES[cls_id]
                    elif num_classes == len(config.UIEB_CLASSES):
                        cls_name = config.UIEB_CLASSES[cls_id]
                    elif cls_id < len(config.COCO_CLASSES):
                        cls_name = config.COCO_CLASSES[cls_id]
                    else:
                        cls_name = f"object_{cls_id}"

                    detections.append({
                        "class": cls_name,
                        "conf": confidences[i],
                        "box": [bx, by, bx + bw, by + bh]
                    })
        else:
            # PyTorch inference
            results = self.pt_model.predict(
                frame,
                imgsz=config.INFERENCE_IMG_SIZE,
                conf=self.conf_thresh,
                iou=config.IOU_THRESHOLD,
                verbose=False
            )
            for r in results:
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    box = [float(v) for v in b.xyxy[0]]
                    cls_name = r.names.get(cls_id, f"obj_{cls_id}")
                    detections.append({
                        "class": cls_name,
                        "conf": conf,
                        "box": box
                    })

        return detections


class SensorFusionPipeline:
    def __init__(self,
                 camera_source=config.CAMERA_INDEX,
                 model_path: Optional[str] = None,
                 headless: bool = False,
                 save_dir: Optional[Union[str, Path]] = None,
                 auto_save: bool = True,
                 save_interval: float = 0.0,
                 simulate: bool = False):
        self.headless = headless
        self.simulate = simulate
        self.camera = CameraStream(source=camera_source).start()
        self.sonar = UltrasonicSensor(simulate=simulate)
        self.detector = UnderwaterDetector(model_path=model_path)
        self.hud = HUDOverlay()

        # Telemetry & Capture State (-1.0 = Disconnected / Not yet read)
        self.current_distance_cm = -1.0
        self.is_running = False
        self.lock = threading.Lock()

        # Reference Image Saver Configuration
        self.save_dir = Path(save_dir) if save_dir else config.OUTPUT_CAPTURES_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save = auto_save
        self.save_interval = save_interval
        self.save_cooldown = config.SAVE_COOLDOWN_SEC
        self.last_save_time = 0.0
        self.last_interval_save = time.time()
        self.capture_count = 0

        # Initialize CSV log file
        self._init_csv_log()

    def _init_csv_log(self):
        """Creates telemetry CSV log file with headers if it doesn't already exist."""
        try:
            if not config.TELEMETRY_LOG_CSV.exists():
                with open(config.TELEMETRY_LOG_CSV, mode="w", newline="", encoding="utf-8") as f:
                    import csv
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Filename", "Distance_CM", "Status", "Detections", "Trigger_Reason"])
        except Exception as e:
            print(f"[!] Warning initializing telemetry CSV: {e}")

    def save_reference_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]], dist: float, reason: str) -> Optional[Path]:
        """Saves annotated reference image and logs telemetry to CSV."""
        try:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            millis = int(time.time() * 1000) % 1000
            filename = f"capture_{reason}_{timestamp_str}_{millis:03d}.jpg"
            file_path = self.save_dir / filename

            # Save image
            cv2.imwrite(str(file_path), frame)
            self.capture_count += 1
            self.last_save_time = time.time()

            # Format detections string
            det_str = "; ".join([f"{d['class']}:{d['conf']:.2f}" for d in detections]) if detections else "None"
            status = "DANGER" if 0 < dist < config.PROXIMITY_DANGER_CM else ("CAUTION" if dist < config.PROXIMITY_WARNING_CM else "CLEAR")

            # Append to CSV log
            try:
                import csv
                with open(config.TELEMETRY_LOG_CSV, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), filename, f"{dist:.1f}", status, det_str, reason])
            except Exception:
                pass

            print(f" [SAVE #{self.capture_count}] Saved reference image: {filename} (Reason: {reason})")
            return file_path
        except Exception as e:
            print(f"[!] Error saving reference frame: {e}")
            return None

    def _sonar_polling_loop(self):
        """Dedicated thread continuously sampling the ultrasonic distance sensor."""
        while self.is_running:
            dist = self.sonar.get_distance()
            with self.lock:
                self.current_distance_cm = dist
            time.sleep(0.06)  # Sample ~16 times per second

    def run(self, duration: int = 0):
        """Main sensor fusion loop."""
        self.is_running = True
        self.sonar_thread = threading.Thread(target=self._sonar_polling_loop, daemon=True)
        self.sonar_thread.start()

        print(f"\n=======================================================")
        print(f" Underwater Sensor Fusion Active")
        print(f" Camera Source: {self.camera.source}")
        print(f" Medium: {config.MEDIUM.upper()} ({int(config.SPEED_OF_SOUND.get(config.MEDIUM, 34300))} cm/s)")
        print(f" Output Capture Dir: {self.save_dir}")
        print(f" Auto-Save on Detection/Danger: {self.auto_save}")
        if duration > 0:
            print(f" Duration: {duration} seconds")
        print(f" Controls: 's' = Save snapshot, 'q' = Quit")
        print(f"=======================================================\n")

        start_time = time.time()
        try:
            while self.is_running:
                now = time.time()
                if duration > 0 and (now - start_time) >= duration:
                    print(f"\n[+] Completed requested duration of {duration}s.")
                    break

                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # 1. Fetch latest filtered ultrasonic distance
                with self.lock:
                    dist = self.current_distance_cm

                # 2. Run object detection
                detections = self.detector.predict(frame)

                # 3. Decision Engine: Collision Avoidance & Event Handling
                status = "CLEAR"
                if dist < 0:
                    status = "OFFLINE"
                elif 0 < dist < config.PROXIMITY_DANGER_CM:
                    status = "DANGER"
                elif dist < config.PROXIMITY_WARNING_CM:
                    status = "CAUTION"

                # 4. Render HUD Overlay
                annotated_frame = self.hud.draw(frame, detections, dist)

                # 5. Reference Image Auto-Saving
                if self.auto_save:
                    time_since_last_save = now - self.last_save_time
                    if time_since_last_save >= self.save_cooldown:
                        if status == "DANGER" and config.AUTO_SAVE_ON_DANGER:
                            self.save_reference_frame(annotated_frame, detections, dist, reason=f"danger_{int(dist)}cm")
                        elif len(detections) > 0 and config.AUTO_SAVE_ON_DETECTION:
                            top_cls = detections[0]["class"]
                            self.save_reference_frame(annotated_frame, detections, dist, reason=f"detect_{top_cls}")

                # Optional fixed interval saving
                if self.save_interval > 0 and (now - self.last_interval_save) >= self.save_interval:
                    self.save_reference_frame(annotated_frame, detections, dist, reason="interval")
                    self.last_interval_save = now

                # 6. Display or Log
                if not self.headless:
                    cv2.imshow("Underwater Telemetry & Object Detection", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Manual snapshot key
                        self.save_reference_frame(annotated_frame, detections, dist, reason="manual_snap")
                else:
                    det_summary = ", ".join([f"{d['class']} ({d['conf']:.2f})" for d in detections]) if detections else "None"
                    fps_val = self.hud.fps
                    t_stamp = time.strftime('%H:%M:%S')
                    sonar_text = f"{dist:5.1f} cm [{status:<7}]" if dist >= 0 else "[DISCONNECTED]"
                    print(f"[{t_stamp}] Sonar: {sonar_text:<18} | FPS: {fps_val:4.1f} | Detections: {det_summary}")
                    time.sleep(0.08)

        except KeyboardInterrupt:
            print("\n[+] Shutdown signal received...")
        finally:
            self.stop()

    def stop(self):
        self.is_running = False
        self.camera.release()
        self.sonar.cleanup()
        cv2.destroyAllWindows()
        print(f"[+] System halted. Total reference images saved: {self.capture_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Underwater Camera + Sonar Sensor Fusion Pipeline")
    parser.add_argument("--source", default=str(config.CAMERA_INDEX), help="Camera index (0 for live Pi cam), video file, or 'uieb'")
    parser.add_argument("--model", default=None, help="Path to custom model weights (.pt or .onnx)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window for headless Pi / Terminal")
    parser.add_argument("--duration", type=int, default=0, help="Run duration in seconds (0 for infinite)")
    parser.add_argument("--save-dir", default=None, help="Directory to save reference output images")
    parser.add_argument("--save-interval", type=float, default=0.0, help="Save periodic frame every N seconds (0 to disable)")
    parser.add_argument("--no-auto-save", action="store_true", help="Disable automatic saving on detection/danger")
    parser.add_argument("--simulate", action="store_true", help="Enable software simulation mode for ultrasonic sonar testing without hardware")
    args = parser.parse_args()

    pipeline = SensorFusionPipeline(
        camera_source=args.source,
        model_path=args.model,
        headless=args.headless,
        save_dir=args.save_dir,
        auto_save=not args.no_auto_save,
        save_interval=args.save_interval,
        simulate=args.simulate
    )
    pipeline.run(duration=args.duration)
