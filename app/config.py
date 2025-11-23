import os
import json
from pathlib import Path
from pathlib import Path

# Image storage configuration
SAVE_ANNOTATED_IMAGES = os.getenv("SAVE_ANNOTATED_IMAGES", "true").lower() == "true"
IMAGE_RETENTION_HOURS = int(os.getenv("IMAGE_RETENTION_HOURS", "24"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs/inspections"))
MAX_STORAGE_GB = float(os.getenv("MAX_STORAGE_GB", "10"))
_class_thresholds_raw = os.getenv("CLASS_CONF_THRESHOLDS", "").strip()
CLASS_CONF_THRESHOLDS = None
try:
    if _class_thresholds_raw:
        CLASS_CONF_THRESHOLDS = json.loads(_class_thresholds_raw)
except Exception:
    CLASS_CONF_THRESHOLDS = None

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODEL_DIR / "yolo11m_trained.pt")))
MODEL_URL = os.getenv("MODEL_URL", "")

# Post-processing filter settings
ENABLE_SHATTERED_GLASS_FILTER = os.getenv("ENABLE_SHATTERED_GLASS_FILTER", "true").lower() == "true"
SHATTERED_GLASS_CONFIDENCE_THRESHOLD = float(os.getenv("SHATTERED_GLASS_CONF_THRESHOLD", "0.85"))

# Car detection settings (two-stage detection)
CAR_DETECTOR_MODEL = Path(os.getenv("CAR_DETECTOR_MODEL", str(MODEL_DIR / "yolov8n.pt")))
CAR_DETECTION_CONF = float(os.getenv("CAR_DETECTION_CONF", "0.4"))
CAR_DETECTION_PADDING = int(os.getenv("CAR_DETECTION_PADDING", "20"))
DETECT_ONLY_CARS = os.getenv("DETECT_ONLY_CARS", "true").lower() == "true"
ENABLE_TWO_STAGE_DETECTION = os.getenv("ENABLE_TWO_STAGE_DETECTION", "true").lower() == "true"

# Ensure output directory exists
if SAVE_ANNOTATED_IMAGES:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
