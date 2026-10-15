import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from procedures import CLASS_NAMES
from .common import check_names, sha256
from procedures.events import MotorcycleCapture

# OpenCV BGR: cyan motorcycles, bright green helmets, orange-red uncovered heads.
COLORS = [(255, 220, 0), (70, 255, 70), (40, 65, 255)]


def select_device(requested="auto"):
    """Prefer CUDA for auto; preserve an explicit device choice."""
    import torch
    requested = str(requested or "auto").strip().lower()
    available = torch.cuda.is_available()
    selected = ("cuda:0" if available else "cpu") if requested == "auto" else requested
    if selected.isdigit():
        selected = f"cuda:{selected}"
    if selected == "cuda":
        selected = "cuda:0"
    if selected.startswith("cuda"):
        if not available:
            raise ValueError("CUDA was requested, but this PyTorch environment cannot use it. Use --device auto or check your PyTorch/CUDA installation.")
        name = torch.cuda.get_device_name(int(selected.split(":")[-1]))
        reason = "CUDA selected automatically." if requested == "auto" else "CUDA selected explicitly."
    else:
        name = selected.upper()
        reason = ("CUDA is unavailable in this PyTorch environment; using CPU." if requested == "auto"
                  else f"{name} selected explicitly." + (" CUDA is also available." if available else ""))
    return selected, name, available, reason


class Detector:
    """One model, serialized inference; tracking state belongs to each source."""
    def __init__(self, weights, device="auto", imgsz=960):
        from ultralytics import YOLO
        weights = Path(weights).resolve()
        if not weights.is_file():
            raise ValueError(f"Checkpoint not found: {weights}")
        self.device, device_name, cuda_available, device_reason = select_device(device)
        self.model = YOLO(str(weights))
        self.model.to(self.device)
        check_names(self.model.names)
        self.lock = threading.Lock()
        self.imgsz = imgsz
        self.metadata = {"checkpoint": weights.name, "sha256": sha256(weights),
                         "imgsz": imgsz, "device": self.device, "device_name": device_name,
                         "cuda_available": cuda_available, "device_reason": device_reason,
                         "names": CLASS_NAMES}

    def predict(self, frame):
        with self.lock:
            # Preserve low-score bike boxes for ByteTrack's second association stage.
            result = self.model.predict(frame, conf=0.1, iou=0.5, imgsz=self.imgsz,
                                        device=self.device, verbose=False)[0]
        return result.boxes.cpu()


def annotate(frame, detections, threshold):
    output = frame.copy()
    height, width = frame.shape[:2]
    thickness = 1 if min(height, width) < 720 else 2
    font_scale = max(0.35, min(0.60, min(height, width) / 1800))
    for d in detections:
        if d["confidence"] < threshold:
            continue
        box = d["box"]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, x2 = max(0, x1), min(width - 1, x2)
        y1, y2 = max(0, y1), min(height - 1, y2)
        color = COLORS[d["class_id"]]
        label = f"{CLASS_NAMES[d['class_id']]} {d['confidence']:.2f}"
        if d.get("track_id") is not None:
            label += f"  #{d['track_id']}"
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        left = max(0, min(x1, width - tw - 4))
        top = max(0, y1 - th - baseline - 5)
        cv2.rectangle(output, (left, top), (min(width - 1, left + tw + 4), top + th + baseline + 4), color, -1)
        cv2.putText(output, label, (left + 2, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return output


class Pipeline:
    def __init__(self, detector, fps=30, display_conf=0.45, candidate_conf=0.6):
        self.detector = detector
        self.display_conf = display_conf
        self.capture = MotorcycleCapture(min_confidence=candidate_conf, bike_confidence=display_conf,
                                         required_frames=5, require_track_id=True)
        # A still image cannot supply five consecutive frames.
        self.image_capture = MotorcycleCapture(min_confidence=candidate_conf, bike_confidence=display_conf,
                                               required_frames=1, require_track_id=False)
        self.tracker = None
        self.fps = fps

    def _start_tracker(self):
        from ultralytics.trackers.byte_tracker import BYTETracker, STrack
        # Ultralytics otherwise uses a process-global ID counter. Each source needs its own.
        class SourceTrack(STrack):
            counter = 0

            @classmethod
            def next_id(cls):
                cls.counter += 1
                return cls.counter

        class SourceTracker(BYTETracker):
            track_class = SourceTrack

            @staticmethod
            def reset_id():
                SourceTrack.counter = 0

        config = Path(__file__).resolve().parents[1] / "configs/bytetrack.yaml"
        settings = yaml.safe_load(config.read_text())
        settings["track_buffer"] = max(1, int(round(self.fps)))  # approximately one second
        self.tracker = SourceTracker(SimpleNamespace(**settings))

    def process(self, frame, frame_index, timestamp, track=True, camera_sequence=None):
        boxes = self.detector.predict(frame)
        detections = [{"box": box.tolist(), "confidence": float(conf), "class_id": int(cls), "track_id": None}
                      for box, conf, cls in zip(boxes.xyxy.numpy(), boxes.conf.numpy(), boxes.cls.numpy())]
        if track:
            if self.tracker is None:
                self._start_tracker()
            indices = [i for i, d in enumerate(detections) if d["class_id"] == 0]
            bikes = boxes[indices].numpy()
            # Ultralytics reports the matched detection index in the last column.
            for row in self.tracker.update(bikes, frame):
                local_index = int(row[-1])
                if 0 <= local_index < len(indices):
                    detections[indices[local_index]]["track_id"] = int(row[4])
        policy = self.capture if track else self.image_capture
        events = policy.update(detections, frame_index, timestamp, source_frame=camera_sequence)
        return annotate(frame, detections, self.display_conf), detections, events


def save_jpeg(path, frame):
    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not success:
        raise ValueError("Image encoding failed.")
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(buffer.tobytes())
    temporary.replace(path)
