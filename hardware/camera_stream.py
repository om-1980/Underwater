"""
Threaded Camera Stream Reader for Raspberry Pi & USB Cameras
Supports:
- Raspberry Pi Camera Module (via Picamera2 or V4L2)
- USB Underwater Webcams / Endoscopes
- Prerecorded underwater test video files
- Synthetic underwater test pattern (when no camera hardware is attached)
- Real-time CLAHE underwater contrast enhancement
"""
import cv2
import time
import threading
from pathlib import Path
import numpy as np
from typing import Optional, Union

import config
from utils.image_enhancement import enhance_underwater_frame


class CameraStream:
    def __init__(self,
                 source: Union[int, str] = config.CAMERA_INDEX,
                 width: int = config.CAMERA_WIDTH,
                 height: int = config.CAMERA_HEIGHT,
                 fps: int = config.CAMERA_FPS,
                 enable_enhancement: bool = config.ENABLE_UNDERWATER_CLAHE):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_enhancement = enable_enhancement

        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame: Optional[np.ndarray] = None
        self.is_running = False
        self.lock = threading.Lock()
        self.is_synthetic = False

        self.is_uieb_stream = False
        self.uieb_images = []
        self.uieb_idx = 0
        self.picam2 = None

        # Check if source is explicitly requested as UIEB or path to UIEB images
        if str(self.source).lower() == "uieb" or (isinstance(self.source, (str, Path)) and Path(self.source).is_dir()):
            target_dir = Path(self.source) if isinstance(self.source, (str, Path)) and Path(self.source).is_dir() else config.UIEB_RAW_DIR
            if target_dir.exists():
                self.uieb_images = sorted(list(target_dir.glob("*.png")) + list(target_dir.glob("*.jpg")))
                if len(self.uieb_images) > 0:
                    self.is_uieb_stream = True
                    print(f"[+] CameraStream initialized with {len(self.uieb_images)} real underwater photos from UIEB Dataset!")
                    first_img = cv2.imread(str(self.uieb_images[0]))
                    if first_img is not None:
                        self.current_frame = cv2.resize(first_img, (self.width, self.height))
                    return

        self._init_capture()

    def _init_capture(self):
        # 1. Try Picamera2 on Raspberry Pi (Bookworm / Bullseye with CSI ribbon camera)
        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            config_cam = self.picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"}
            )
            self.picam2.configure(config_cam)
            self.picam2.start()
            frame = self.picam2.capture_array()
            if frame is not None and frame.size > 0:
                self.current_frame = frame
                print("[+] Raspberry Pi Picamera2 CSI Camera initialized successfully!")
                return
        except Exception:
            self.picam2 = None

        # 2. Try standard OpenCV VideoCapture (V4L2 / USB webcam / Pi camera legacy)
        try:
            src_val = int(self.source) if str(self.source).isdigit() else self.source
            self.cap = cv2.VideoCapture(src_val)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.current_frame = frame
                    print(f"[+] Live Camera opened successfully on source: {src_val}")
                    return
        except Exception as e:
            print(f"[!] Warning while opening camera {self.source}: {e}")

        # 3. Fallback to UIEB raw photos if available, else synthetic test feed
        if config.UIEB_RAW_DIR.exists():
            self.uieb_images = sorted(list(config.UIEB_RAW_DIR.glob("*.png")) + list(config.UIEB_RAW_DIR.glob("*.jpg")))
            if len(self.uieb_images) > 0:
                self.is_uieb_stream = True
                print(f"[i] Hardware camera not found. Using {len(self.uieb_images)} real UIEB underwater test photos.")
                first_img = cv2.imread(str(self.uieb_images[0]))
                if first_img is not None:
                    self.current_frame = cv2.resize(first_img, (self.width, self.height))
                return

        print(f"[i] Hardware camera not available on source {self.source}. Using Synthetic Underwater Video Generator.")
        self.is_synthetic = True
        self.current_frame = self._generate_synthetic_frame(0)

    def _generate_synthetic_frame(self, step: int) -> np.ndarray:
        """Generates dynamic murky underwater scene for testing without hardware."""
        h, w = self.height, self.width
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Deep murky cyan/green water
        for y in range(h):
            g = int(60 + (y / h) * 45)
            b = int(75 + (y / h) * 55)
            r = int(15 + (y / h) * 20)
            frame[y, :] = (b, g, r)

        # Draw a moving swimming fish
        t = step * 0.05
        fish_x = int((w / 2) + np.sin(t) * 160)
        fish_y = int((h / 2) + np.cos(t * 0.8) * 70)
        cv2.ellipse(frame, (fish_x, fish_y), (45, 20), 0, 0, 360, (190, 220, 200), -1)
        cv2.circle(frame, (fish_x + 30, fish_y - 4), 3, (20, 20, 20), -1)

        # Overlay text
        cv2.putText(frame, "[SYNTHETIC TEST CAMERA FEED]", (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 220), 1, cv2.LINE_AA)
        return frame

    def start(self):
        """Start background capture thread."""
        self.is_running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        return self

    def _update_loop(self):
        step = 0
        while self.is_running:
            if self.picam2 is not None:
                try:
                    frame = self.picam2.capture_array()
                    if frame is not None and frame.size > 0:
                        if self.enable_enhancement:
                            frame = enhance_underwater_frame(frame)
                        with self.lock:
                            self.current_frame = frame
                except Exception:
                    pass
                time.sleep(1.0 / self.fps)
            elif self.is_uieb_stream and len(self.uieb_images) > 0:
                # Cycle through real UIEB underwater photos
                img_path = self.uieb_images[self.uieb_idx]
                frame = cv2.imread(str(img_path))
                if frame is not None:
                    frame = cv2.resize(frame, (self.width, self.height))
                    if self.enable_enhancement:
                        frame = enhance_underwater_frame(frame)
                    with self.lock:
                        self.current_frame = frame
                self.uieb_idx = (self.uieb_idx + 1) % len(self.uieb_images)
                time.sleep(1.0 / max(1, min(self.fps, 10)))  # Smooth slide transition
            elif not self.is_synthetic and self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    if self.enable_enhancement:
                        frame = enhance_underwater_frame(frame)
                    with self.lock:
                        self.current_frame = frame
                else:
                    # Loop if video file finished
                    if isinstance(self.source, str):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                # Synthetic feed
                step += 1
                synth_frame = self._generate_synthetic_frame(step)
                if self.enable_enhancement:
                    synth_frame = enhance_underwater_frame(synth_frame)
                with self.lock:
                    self.current_frame = synth_frame
                time.sleep(1.0 / self.fps)

    def read(self) -> Optional[np.ndarray]:
        """Returns the latest captured frame."""
        with self.lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def release(self):
        """Stops thread and releases camera."""
        self.is_running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                print("[+] Picamera2 stopped.")
            except Exception:
                pass
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print("[+] Camera released.")


if __name__ == "__main__":
    stream = CameraStream().start()
    print("Testing Camera Stream... Press 'q' to quit.")
    while True:
        frame = stream.read()
        if frame is not None:
            cv2.imshow("Underwater Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    stream.release()
    cv2.destroyAllWindows()
