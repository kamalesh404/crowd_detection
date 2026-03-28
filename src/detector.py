"""
YOLOv8 Person Detector
Detects people in each frame and returns bounding-box results.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    YOLO_MODEL, YOLO_CONFIDENCE, YOLO_IOU_THRESHOLD, YOLO_CLASSES, MODEL_DIR,
    YOLO_DEVICE, YOLO_IMGSZ, YOLO_FRAME_STRIDE, YOLO_USE_HALF
)


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    track_id: int = -1

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def bbox(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self):
        return (self.cx, self.cy)


class PersonDetector:
    """Wraps YOLOv8 for real-time person detection."""

    def __init__(
        self,
        model_name: str = YOLO_MODEL,
        device: str = YOLO_DEVICE,
        imgsz: int = YOLO_IMGSZ,
        frame_stride: int = YOLO_FRAME_STRIDE,
        use_half: bool = YOLO_USE_HALF,
    ):
        self.model = None
        self.model_name = model_name
        self.device = device
        self.imgsz = imgsz
        self.frame_stride = max(1, frame_stride)
        self.use_half = use_half
        self.using_half = False
        self.resolved_device = "cpu"
        self.frame_count = 0
        self.total_detections = 0
        self._last_detections: List[Detection] = []
        self.class_ids = YOLO_CLASSES if YOLO_CLASSES else None
        self._warned_nms_fallback = False
        self._load_model()

    def _resolve_device(self) -> str:
        if self.device and self.device != "auto":
            return self.device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    def _load_model(self):
        try:
            from ultralytics import YOLO
            model_path = os.path.join(MODEL_DIR, self.model_name)
            # Download to models dir if not there
            if not os.path.exists(model_path):
                model_path = self.model_name   # ultralytics auto-downloads
            self.model = YOLO(model_path)
            self.resolved_device = self._resolve_device()
            if self.resolved_device.startswith("cuda") and not self._cuda_nms_supported():
                print("[Detector] WARNING: torchvision CUDA NMS is unavailable. Falling back to CPU inference.")
                self.resolved_device = "cpu"
            try:
                self.model.to(self.resolved_device)
            except Exception:
                # Some backends may not support explicit .to(); inference call still sets device.
                pass
            if self.use_half and self.resolved_device.startswith("cuda"):
                self.using_half = True
            self.class_ids = self._resolve_person_class_ids()
            print(
                f"[Detector] Model loaded: {self.model_name} "
                f"(device={self.resolved_device}, imgsz={self.imgsz}, "
                f"stride={self.frame_stride}, fp16={self.using_half}, classes={self.class_ids})"
            )
        except Exception as e:
            print(f"[Detector] WARNING: Could not load YOLO model ({e}). Using simulation.")
            self.model = None

    def _resolve_person_class_ids(self):
        if YOLO_CLASSES:
            return YOLO_CLASSES
        try:
            names = getattr(self.model, "names", {}) if self.model is not None else {}
            if isinstance(names, dict):
                person_ids = [cid for cid, name in names.items() if str(name).lower() == "person"]
                return person_ids or None
        except Exception:
            pass
        return None

    def _cuda_nms_supported(self) -> bool:
        try:
            import torchvision
            import torch
            if not torch.cuda.is_available():
                return False
            # CPU-only torchvision builds expose "+cpu" in version and miss CUDA NMS kernels.
            v = str(getattr(torchvision, "__version__", "")).lower()
            return "+cpu" not in v
        except Exception:
            return False

    def _fallback_to_cpu(self, reason: str):
        if self.resolved_device == "cpu":
            return
        self.resolved_device = "cpu"
        self.using_half = False
        try:
            if self.model is not None:
                self.model.to("cpu")
        except Exception:
            pass
        if not self._warned_nms_fallback:
            print(f"[Detector] WARNING: Switched inference to CPU ({reason}).")
            self._warned_nms_fallback = True

    def _sanitize_bbox(self, x1: int, y1: int, x2: int, y2: int, shape) -> Optional[tuple]:
        h, w = shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single frame. Returns list of Detection objects."""
        self.frame_count += 1
        if self.model is None:
            return []
        if self.frame_stride > 1 and (self.frame_count % self.frame_stride) != 0:
            return self._last_detections

        try:
            results = self.model(
                frame,
                conf=YOLO_CONFIDENCE,
                iou=YOLO_IOU_THRESHOLD,
                classes=self.class_ids,
                device=self.resolved_device,
                imgsz=self.imgsz,
                half=self.using_half,
                verbose=False
            )
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bbox = self._sanitize_bbox(x1, y1, x2, y2, frame.shape)
                    if bbox is None:
                        continue
                    conf = float(box.conf[0])
                    detections.append(Detection(*bbox, conf))
            self._last_detections = detections
            self.total_detections += len(detections)
            return detections
        except Exception as e:
            msg = str(e)
            if "torchvision::nms" in msg and "CUDA" in msg:
                self._fallback_to_cpu("torchvision CUDA NMS missing")
                try:
                    results = self.model(
                        frame,
                        conf=YOLO_CONFIDENCE,
                        iou=YOLO_IOU_THRESHOLD,
                        classes=self.class_ids,
                        device=self.resolved_device,
                        imgsz=self.imgsz,
                        half=False,
                        verbose=False
                    )
                    detections = []
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            bbox = self._sanitize_bbox(x1, y1, x2, y2, frame.shape)
                            if bbox is None:
                                continue
                            conf = float(box.conf[0])
                            detections.append(Detection(*bbox, conf))
                    self._last_detections = detections
                    self.total_detections += len(detections)
                    return detections
                except Exception as retry_error:
                    print(f"[Detector] Inference retry on CPU failed: {retry_error}")
                    return self._last_detections
            print(f"[Detector] Inference error: {e}")
            return self._last_detections

    def draw_detections(self, frame: np.ndarray, detections: List[Detection],
                        show_confidence: bool = True, blur_faces: bool = True) -> np.ndarray:
        """Draw bounding boxes on frame with optional privacy face blurring."""
        for det in detections:
            # Privacy: Blur top 20% of bounding box (approx head location)
            if blur_faces:
                head_h = int(det.height * 0.25)
                hx1, hy1, hx2, hy2 = det.x1, det.y1, det.x2, det.y1 + head_h
                
                # Bounds check
                hx1, hy1 = max(0, hx1), max(0, hy1)
                hx2, hy2 = min(frame.shape[1], hx2), min(frame.shape[0], hy2)
                
                if hx2 > hx1 and hy2 > hy1:
                    roi = frame[hy1:hy2, hx1:hx2]
                    try:
                        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)
                        frame[hy1:hy2, hx1:hx2] = blurred_roi
                    except Exception:
                        pass
        
            color = (0, 255, 100)
            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            if show_confidence:
                label = f"#{det.track_id} {det.confidence:.2f}" if det.track_id >= 0 else f"{det.confidence:.2f}"
                cv2.putText(frame, label, (det.x1, det.y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
