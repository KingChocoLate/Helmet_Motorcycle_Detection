"""Motorcycle crops, including the consecutive-frame confirmation policy."""
from collections import deque
from dataclasses import dataclass, field


def match_head_to_bike(head, bikes):
    """Find the most plausible rider-inclusive motorcycle for a NoHelmet head.

    The head's *center* may be inside a bike box even when the full head box
    crosses its edge. If several motorcycle boxes overlap, prefer the one
    whose centerline is clearly closest to the head. Return None for ties,
    rather than guessing and potentially storing the wrong motorcycle.
    """
    x1, y1, x2, y2 = head["box"]
    if x2 <= x1 or y2 <= y1:
        return None
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    candidates = []
    for index, bike in enumerate(bikes):
        left, top, right, bottom = bike["box"]
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            continue
        # Only heads in the top 70% of the combined bike+rider box qualify.
        if not (left <= cx <= right and top <= cy <= top + .70 * height):
            continue
        x_offset = abs(cx - (left + right) / 2) / width
        y_position = (cy - top) / height
        score = x_offset + .15 * abs(y_position - .18)
        candidates.append((score, index))
    if not candidates:
        return None
    candidates.sort()
    # A small score difference is not enough to decide between two bikes.
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < .09:
        return None
    return candidates[0][1]


class MotorcycleCapture:
    """Capture a motorcycle after five consecutive NoHelmet-positive frames.

    A positive frame needs an above-threshold NoHelmet head associated with
    a rider-inclusive BikeWithRider box. Use the head center in the upper
    part of the rider box and prefer a clear geometric match when motorcycles
    overlap. For video/camera the same ByteTrack ID must be positive in five
    consecutive frames; a missed or ambiguous frame resets that streak.
    """

    def __init__(self, min_confidence=0.6, bike_confidence=0.45,
                 required_frames=5, require_track_id=True):
        self.min_confidence = min_confidence
        self.bike_confidence = bike_confidence
        self.required_frames = required_frames
        self.require_track_id = require_track_id
        self.streaks = {}
        self.emitted = set()
        self.last_source_frame = None

    def update(self, detections, frame_index, timestamp, source_frame=None):
        # A dropped camera frame must not count as consecutive evidence.
        source_frame = frame_index if source_frame is None else source_frame
        if self.last_source_frame is not None and source_frame != self.last_source_frame + 1:
            self.streaks.clear()
        self.last_source_frame = source_frame

        bikes = [d for d in detections if d["class_id"] == 0
                 and d["confidence"] >= self.bike_confidence]
        matches = {}
        for head in detections:
            if head["class_id"] != 2 or head["confidence"] < self.min_confidence:
                continue
            index = match_head_to_bike(head, bikes)
            if index is not None:
                matches.setdefault(index, []).append(head)

        positives = {}
        for index, heads in matches.items():
            bike = bikes[index]
            identity = bike.get("track_id") if self.require_track_id else index
            if identity is not None:
                positives[identity] = (bike, heads)

        # A missing NoHelmet prediction clears the streak immediately.
        self.streaks = {track_id: self.streaks[track_id]
                        for track_id in positives if track_id in self.streaks}
        events = []
        for track_id, (bike, heads) in positives.items():
            self.streaks[track_id] = self.streaks.get(track_id, 0) + 1
            if self.streaks[track_id] < self.required_frames or track_id in self.emitted:
                continue
            self.emitted.add(track_id)
            best = max(heads, key=lambda d: d["confidence"])
            events.append({"capture_mode": "motorcycle_nohelmet_crop",
                           "track_id": bike.get("track_id"),
                           "frame_index": frame_index, "timestamp_seconds": round(timestamp, 3),
                           "confidence": best["confidence"], "bike_box": list(bike["box"]),
                           "head_box": list(best["box"]),
                           "head_boxes": [list(h["box"]) for h in heads],
                           "nohelmet_count": len(heads),
                           "consecutive_nohelmet_frames": self.streaks[track_id],
                           "required_consecutive_frames": self.required_frames,
                           "review": "pending", "note": ""})
        return events

class NoHelmetCapture:
    """Basic box captures, independent of motorcycle association or tracking.

    Capture all qualifying boxes immediately, then wait one second before
    saving another video/camera frame. Repeated people are not deduplicated.
    """

    def __init__(self, min_confidence=0.6, interval_seconds=1.0):
        self.min_confidence = min_confidence
        self.interval_seconds = interval_seconds
        self.last_capture = None

    def update(self, detections, frame_index, timestamp):
        if self.last_capture is not None and timestamp - self.last_capture < self.interval_seconds - 1e-6:
            return []
        events = [
            {"capture_mode": "nohelmet_crop", "track_id": None,
             "frame_index": frame_index, "timestamp_seconds": round(timestamp, 3),
             "confidence": d["confidence"], "head_box": list(d["box"]),
             "review": "pending", "note": ""}
            for d in detections
            if d["class_id"] == 2 and d["confidence"] >= self.min_confidence
        ]
        if events:
            self.last_capture = timestamp
        return events


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda box: max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    return intersection / max(area(a) + area(b) - intersection, 1e-9)


def associate(head, bikes):
    x1, y1, x2, y2 = head["box"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    candidates = []
    for bike in bikes:
        left, top, right, bottom = bike["box"]
        # Restrict heads to the upper 70% of a rider-inclusive box.
        if left <= cx <= right and top <= cy <= top + 0.7 * (bottom - top):
            candidates.append(bike)
    # Dense overlapping motorcycles are deferred rather than assigned arbitrarily.
    return candidates[0] if len(candidates) == 1 else None


@dataclass
class TrackHistory:
    votes: deque = field(default_factory=lambda: deque(maxlen=5))
    last_seen: float = 0


class CandidateCapture:
    def __init__(self, min_confidence=0.6, required_hits=3, max_gap_seconds=1.0):
        self.min_confidence = min_confidence
        self.required_hits = required_hits
        self.max_gap_seconds = max_gap_seconds
        self.histories = {}
        self.emitted = set()

    def update(self, detections, frame_index, timestamp):
        bikes = [d for d in detections if d["class_id"] == 0 and d.get("track_id") is not None]
        helmets = [d for d in detections if d["class_id"] == 1 and d["confidence"] >= self.min_confidence]
        hits = {}
        for head in detections:
            if head["class_id"] != 2 or head["confidence"] < self.min_confidence:
                continue
            if any(iou(head["box"], helmet["box"]) > 0.3 for helmet in helmets):
                continue
            bike = associate(head, bikes)
            if bike is not None:
                identity = bike["track_id"]
                if identity not in hits or hits[identity]["confidence"] < head["confidence"]:
                    hits[identity] = head
        visible = {d["track_id"]: d for d in bikes}
        for identity in list(self.histories):
            if timestamp - self.histories[identity].last_seen > self.max_gap_seconds:
                del self.histories[identity]
        for identity in visible:
            self.histories.setdefault(identity, TrackHistory()).last_seen = timestamp
        output = []
        for identity, history in self.histories.items():
            history.votes.append(identity in hits)
            if identity in hits and sum(history.votes) >= self.required_hits and identity not in self.emitted:
                self.emitted.add(identity)
                output.append({"track_id": identity, "frame_index": frame_index,
                               "timestamp_seconds": round(timestamp, 3), "confidence": hits[identity]["confidence"],
                               "bike_box": visible[identity]["box"], "head_box": hits[identity]["box"],
                               "hits_in_window": sum(history.votes), "window_size": len(history.votes),
                               "review": "pending", "note": ""})
        return output
