"""
HUD (Heads-Up Display) Overlay Module for Underwater Telemetry
Renders sensor fusion telemetry, real-time FPS, ultrasonic distance gauge,
and object bounding boxes with danger warnings.
"""
import cv2
import numpy as np
import time
from typing import List, Dict, Any

import config


class HUDOverlay:
    def __init__(self):
        self.last_time = time.time()
        self.fps = 0.0
        self.frame_count = 0
        self.fps_timer = time.time()

    def update_fps(self) -> float:
        self.frame_count += 1
        elapsed = time.time() - self.fps_timer
        if elapsed >= 0.5:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_timer = time.time()
        return self.fps

    def draw(self, frame: np.ndarray, detections: List[Dict[str, Any]], distance_cm: float) -> np.ndarray:
        """
        Draws telemetry HUD onto the provided frame.
        :param frame: BGR image numpy array
        :param detections: List of dicts: [{'class': str, 'conf': float, 'box': [x1, y1, x2, y2]}]
        :param distance_cm: Ultrasonic sensor distance reading in centimeters
        :return: Annotated frame
        """
        h, w, _ = frame.shape
        fps = self.update_fps()

        # -------------------------------------------------------------
        # 1. Draw Object Detection Bounding Boxes
        # -------------------------------------------------------------
        for det in detections:
            cls_name = det.get("class", "object")
            conf = det.get("conf", 0.0)
            box = det.get("box", [0, 0, 0, 0])
            x1, y1, x2, y2 = [int(v) for v in box]

            color = config.CLASS_COLORS.get(cls_name, config.DEFAULT_BOX_COLOR)

            # Draw sleek bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label banner
            label = f"{cls_name.upper()} {conf * 100:.0f}%"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            banner_y = max(y1, text_h + 8)

            # Background rectangle for text
            cv2.rectangle(frame, (x1, banner_y - text_h - 6), (x1 + text_w + 10, banner_y + 2), color, -1)
            cv2.putText(frame, label, (x1 + 5, banner_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 2. Draw Top Telemetry Bar (Semi-transparent Glassmorphism overlay)
        # -------------------------------------------------------------
        top_bar_h = 48
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, top_bar_h), (20, 24, 28), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Top Bar Dividers & Texts
        # Title / System status
        cv2.putText(frame, "UNDERWATER SENSOR FUSION", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2, cv2.LINE_AA)
        status_sub = f"Medium: {config.MEDIUM.upper()} ({int(config.SPEED_OF_SOUND.get(config.MEDIUM, 34300))} cm/s)"
        cv2.putText(frame, status_sub, (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

        # FPS indicator
        fps_text = f"FPS: {fps:4.1f}"
        cv2.putText(frame, fps_text, (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 2, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. Ultrasonic Distance Gauge (Right Panel / Gauge)
        # -------------------------------------------------------------
        if distance_cm < 0:
            dist_text = "SONAR: DISCONNECTED"
            status_color = (140, 140, 140)  # Neutral Gray
            alert_level = "DISCONNECTED"
        elif distance_cm < config.PROXIMITY_DANGER_CM:
            dist_text = f"SONAR: {distance_cm:5.1f} cm [DANGER]"
            status_color = (0, 0, 255)       # Red
            alert_level = "CRITICAL PROXIMITY"
        elif distance_cm < config.PROXIMITY_WARNING_CM:
            dist_text = f"SONAR: {distance_cm:5.1f} cm [CAUTION]"
            status_color = (0, 215, 255)     # Yellow / Gold
            alert_level = "OBSTACLE NEAR"
        else:
            dist_text = f"SONAR: {distance_cm:5.1f} cm [CLEAR]"
            status_color = (0, 255, 128)     # Green
            alert_level = "CLEAR"

        # Telemetry Card in Top Center
        card_w = 280
        card_h = 36
        card_x = w // 2 - card_w // 2
        card_y = 6
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (40, 44, 52), -1)
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), status_color, 2)
        cv2.putText(frame, dist_text, (card_x + 12, card_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 4. Critical Proximity Alarm (Flashing Center Banner)
        # -------------------------------------------------------------
        if 0 < distance_cm < config.PROXIMITY_DANGER_CM:
            # Flash based on time
            if int(time.time() * 4) % 2 == 0:
                banner_overlay = frame.copy()
                cv2.rectangle(banner_overlay, (0, h - 70), (w, h - 20), (0, 0, 180), -1)
                cv2.addWeighted(banner_overlay, 0.85, frame, 0.15, 0, frame)
                warning_msg = f"! COLLISION HAZARD: OBSTACLE AT {distance_cm:.1f} CM !"
                (tw, th), _ = cv2.getTextSize(warning_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.putText(frame, warning_msg, ((w - tw) // 2, h - 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return frame
