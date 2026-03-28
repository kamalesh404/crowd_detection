"""
Global configuration for the Intelligent Crowd Safety & Stampede Prevention System.
All thresholds, paths, and system parameters are defined here.
"""

import os


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int_list(name: str, default):
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    if value == "":
        return []
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            return default
    return out

# ──────────────────────────────────────────────
# BASE PATHS
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RECORDING_DIR = os.path.join(BASE_DIR, "recordings")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
ASSET_DIR = os.path.join(BASE_DIR, "assets")
DB_PATH = os.path.join(BASE_DIR, "crowd_safety.db")

# ──────────────────────────────────────────────
# VIDEO / CAMERA
# ──────────────────────────────────────────────
DEFAULT_SOURCE = 0           # 0 = webcam, or path to video file or RTSP string
TARGET_FPS = 30
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DISPLAY_SCALE = 1.0
FLIP_CAMERA_FEED = _env_bool("FLIP_CAMERA_FEED", True)

# ──────────────────────────────────────────────
# AI MODEL
# ──────────────────────────────────────────────
YOLO_MODEL = "yolov8n.pt"          # yolov8n / yolov8s / yolov8m / yolov8l
YOLO_CONFIDENCE = _env_float("YOLO_CONFIDENCE", 0.25)
YOLO_IOU_THRESHOLD = _env_float("YOLO_IOU_THRESHOLD", 0.45)
YOLO_CLASSES = _env_int_list("YOLO_CLASSES", [0])  # [] disables class filtering
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "auto")  # auto / cuda / cpu / cuda:0
YOLO_IMGSZ = _env_int("YOLO_IMGSZ", 640)
YOLO_FRAME_STRIDE = max(1, _env_int("YOLO_FRAME_STRIDE", 1))
YOLO_USE_HALF = _env_bool("YOLO_USE_HALF", True)
PRIVACY_BLUR_FACES = _env_bool("PRIVACY_BLUR_FACES", False)

# ──────────────────────────────────────────────
# TRACKING
# ──────────────────────────────────────────────
TRACKER_MAX_AGE = 30           # frames before a lost track is deleted
TRACKER_MIN_HITS = max(1, _env_int("TRACKER_MIN_HITS", 1))  # frames to confirm track
TRACKER_IOU_THRESHOLD = 0.3
TRAJECTORY_HISTORY = 60        # number of past positions to store

# ──────────────────────────────────────────────
# DENSITY & COUNTING
# ──────────────────────────────────────────────
MAX_CAPACITY = 100             # venue maximum safe capacity
WARNING_CAPACITY_PCT = 0.70    # 70% → Warning
DANGER_CAPACITY_PCT = 0.90     # 90% → Danger

DENSITY_GRID_ROWS = 4
DENSITY_GRID_COLS = 6

DENSITY_WARNING = 0.30         # people / sq pixel (normalised)
DENSITY_DANGER = 0.55
DENSITY_CRITICAL = 0.75

# ──────────────────────────────────────────────
# MOVEMENT / SPEED
# ──────────────────────────────────────────────
SPEED_WINDOW = 10              # frames to average speed over
SPEED_WARNING_MULTIPLIER = 1.8
SPEED_DANGER_MULTIPLIER = 2.4
MIN_DISTANCE_BETWEEN_PEOPLE = 50   # pixels; below this = compression

# ──────────────────────────────────────────────
# ALERT LEVELS
# ──────────────────────────────────────────────
ALERT_SAFE = "SAFE"
ALERT_WARNING = "WARNING"
ALERT_DANGER = "DANGER"
ALERT_CRITICAL = "CRITICAL"

ALERT_COOLDOWN_SECONDS = 10    # minimum seconds between repeated alerts

# ──────────────────────────────────────────────
# RISK PREDICTION (LSTM)
# ──────────────────────────────────────────────
LSTM_SEQUENCE_LEN = 60         # seconds of history
LSTM_FORECAST_STEPS = [30, 60, 120]

# ──────────────────────────────────────────────
# SOCIAL DISTANCING
# ──────────────────────────────────────────────
SOCIAL_DISTANCE_PX = 100       # minimum safe pixel distance

# ──────────────────────────────────────────────
# LOST CHILD DETECTION
# ──────────────────────────────────────────────
CHILD_BBOX_HEIGHT_RATIO = 0.55     # child bbox height / avg adult height ratio
CHILD_ALONE_SECONDS = 15           # stationary alone → alert
CHILD_PROXIMITY_RADIUS = 120       # pixels; must have adult within this range

# ──────────────────────────────────────────────
# HEATMAP
# ──────────────────────────────────────────────
HEATMAP_DECAY = 0.95           # per-frame decay factor

# ──────────────────────────────────────────────
# NOTIFICATION (fill in real credentials via .env)
# ──────────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_FROM", "")
ALERT_PHONE_NUMBERS = os.getenv("ALERT_PHONES", "").split(",")

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
ALERT_EMAILS = os.getenv("ALERT_EMAILS", "").split(",")

WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

# ──────────────────────────────────────────────
# WEB / API
# ──────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ──────────────────────────────────────────────
# SIMULATION MODE
# ──────────────────────────────────────────────
SIMULATION_MODE = False
SIM_INITIAL_PEOPLE = 5
SIM_MAX_PEOPLE = 80
