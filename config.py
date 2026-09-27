"""
Global Configuration for Underwater Camera + Ultrasonic Sensor Fusion System
Designed for Raspberry Pi & Cross-Platform PC Simulation
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
UIEB_RAW_DIR = BASE_DIR / "dataset" / "uieb_raw" / "raw-890"
UIEB_DATA_DIR = BASE_DIR / "dataset" / "uieb_data"
DATASET_DIR = UIEB_DATA_DIR
MODELS_DIR = BASE_DIR / "model" / "weights"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Output Captures & Telemetry Logging
OUTPUT_DIR = BASE_DIR / "deploy" / "output"
OUTPUT_CAPTURES_DIR = OUTPUT_DIR / "live_captures"
OUTPUT_CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
TELEMETRY_LOG_CSV = OUTPUT_DIR / "telemetry_log.csv"

# Reference Image Saving Triggers
AUTO_SAVE_ON_DETECTION = True   # Automatically save when marine organisms/objects are detected
AUTO_SAVE_ON_DANGER = True      # Automatically save when ultrasonic distance < PROXIMITY_DANGER_CM
SAVE_COOLDOWN_SEC = 1.5         # Minimum seconds between automatic image saves to avoid duplicate flooding

# -------------------------------------------------------------
# ULTRASONIC SENSOR CONFIGURATION (HC-SR04 / JSN-SR04T)
# -------------------------------------------------------------
# Raspberry Pi GPIO BCM pin numbers
PIN_TRIGGER = 23
PIN_ECHO = 24

# Acoustic medium speed of sound in cm/s:
# Air: ~34,300 cm/s (343 m/s)
# Fresh water (20°C): ~148,200 cm/s (1,482 m/s)
# Seawater (20°C, 35ppt): ~153,000 cm/s (1,530 m/s)
# Set to 'seawater', 'freshwater', or 'air' depending on whether sensor is submerged or in air housing
MEDIUM = os.getenv("UNDERWATER_MEDIUM", "seawater").lower()

SPEED_OF_SOUND = {
    "air": 34300.0,
    "freshwater": 148200.0,
    "seawater": 153000.0,
}

# Distance Thresholds (in centimeters)
PROXIMITY_DANGER_CM = 30.0   # Critical collision danger
PROXIMITY_WARNING_CM = 60.0  # Caution obstacle approaching
SENSOR_TIMEOUT_SEC = 0.05    # Maximum echo wait time to prevent blocking (50ms)
MEDIAN_FILTER_WINDOW = 5     # Rolling window size to eliminate bubble/multipath noise

# -------------------------------------------------------------
# CAMERA CONFIGURATION
# -------------------------------------------------------------
CAMERA_INDEX = 0             # 0 for default USB/PiCam, or video file path
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Real-time Underwater Image Enhancement (CLAHE)
ENABLE_UNDERWATER_CLAHE = True
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID_SIZE = (8, 8)

# -------------------------------------------------------------
# DEEP LEARNING MODEL CONFIGURATION
# -------------------------------------------------------------
# Recommended for Raspberry Pi: YOLOv8n (nano) or YOLO11n exported to ONNX or TFLite
MODEL_BASE = "yolov8n.pt"
MODEL_TRAINED_PT = MODELS_DIR / "underwater_yolov8n_best.pt"
MODEL_ONNX = MODELS_DIR / "underwater_yolov8n.onnx"
MODEL_TFLITE = MODELS_DIR / "underwater_yolov8n_saved_model" / "underwater_yolov8n_float16.tflite"

INFERENCE_IMG_SIZE = 320     # 320x320 for ultra-fast Raspberry Pi inference, 640 for high accuracy
CONFIDENCE_THRESHOLD = 0.35  # Detection confidence cutoff
IOU_THRESHOLD = 0.45

# Full Real-World Object Detection Classes (COCO 80 + Marine/Underwater Extensions)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# UIEB Underwater Marine Classes
UIEB_CLASSES = [
    "fish",
    "small_fish",
    "crab",
    "marine_object",
    "diver",
    "obstacle",
]

# Default classes dictionary
DEFAULT_CLASSES = COCO_CLASSES

# Color map for bounding boxes (BGR format for OpenCV)
CLASS_COLORS = {
    "person": (0, 255, 255),       # Yellow / Cyan
    "diver": (255, 0, 255),        # Magenta
    "fish": (0, 255, 128),         # Teal / Mint
    "small_fish": (255, 200, 0),   # Light Cyan
    "crab": (0, 140, 255),         # Orange
    "boat": (255, 128, 0),         # Sky Blue
    "bottle": (0, 255, 0),         # Green
    "obstacle": (0, 0, 255),       # Red
    "laptop": (255, 180, 50),      # Blue
    "chair": (180, 105, 255),      # Pink
    "cup": (200, 255, 0),          # Lime
    "cell phone": (0, 165, 255),   # Orange
}
DEFAULT_BOX_COLOR = (0, 255, 128)
