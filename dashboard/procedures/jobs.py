"""Local research jobs, persisted outputs and human review decisions."""
import csv
import io
import json
import logging
import math
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from procedures.common import now, write_json
from procedures.inference import Pipeline, save_jpeg

LOG = logging.getLogger(__name__)


class AnnotatedVideoWriter:
    """Encode frames directly to H.264 instead of re-encoding an intermediate MP4."""

    def __init__(self, folder, fps, width, height):
        self.folder = Path(folder)
        self.frame_shape = (height, width, 3)
        self.process = None
        self.fallback = None
        self.log = None
        self.closed = False
        self.warning = None
        ffmpeg = shutil.which("ffmpeg")
        available = False
        if ffmpeg:
            try:
                encoders = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                                          capture_output=True, timeout=10)
                available = encoders.returncode == 0 and b"libx264" in encoders.stdout
            except (OSError, subprocess.TimeoutExpired):
                pass
        self.browser_video = available
        self.filename = "annotated.mp4" if available else "annotated_raw.mp4"
        if available:
            self.log = (self.folder / "encoder.log").open("wb")
            try:
                self.process = subprocess.Popen([
                    ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-threads", "4",
                    "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-g", str(max(1, round(fps))), "-movflags", "+faststart",
                    str(self.folder / self.filename)],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
            except Exception:
                self.log.close()
                raise
        else:
            self.fallback = cv2.VideoWriter(str(self.folder / self.filename),
                                           cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not self.fallback.isOpened():
                self.fallback.release()
                raise ValueError("No working video encoder was found.")
            self.warning = "Install FFmpeg with libx264 and analyze again for browser playback. The MP4 download remains available."

    def write(self, frame):
        if frame.shape != self.frame_shape or str(frame.dtype) != "uint8":
            raise ValueError("Video frame dimensions or pixel format changed during processing.")
        if self.process is not None:
            try:
                self.process.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError("FFmpeg stopped while writing the video. Check this run's encoder.log.") from exc
        else:
            self.fallback.write(frame)

    def finish(self):
        if self.closed:
            return
        if self.process is not None:
            try:
                self.process.stdin.close()
                code = self.process.wait(timeout=60)
                if code != 0:
                    raise RuntimeError("FFmpeg could not finish the video. Check this run's encoder.log.")
            except Exception:
                self.abort()
                raise
            self.log.close()
        else:
            self.fallback.release()
        self.closed = True

    def abort(self):
        if self.closed:
            return
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            self.log.close()
        elif self.fallback is not None:
            self.fallback.release()
        self.closed = True


class JobStore:
    def __init__(self, root, detector=None, max_jobs=2):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.detector = detector
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="helmet-video")
        self.slots = threading.BoundedSemaphore(max_jobs)
        self.cancel = {}
        self.sessions = {}
        self.jobs = {}
        for file in self.root.glob("*/job.json"):
            data = json.loads(file.read_text())
            if data["status"] in {"queued", "running", "live"}:
                data.update(status="interrupted", error="Server stopped before this run finished.")
                write_json(file, data)
            self.jobs[data["id"]] = data

    def get(self, identity):
        with self.lock:
            if identity not in self.jobs:
                raise KeyError(identity)
            return json.loads(json.dumps(self.jobs[identity]))

    def update(self, identity, **values):
        with self.lock:
            self.jobs[identity].update(values)
            write_json(self.root / identity / "job.json", self.jobs[identity])

    def create(self, kind, filename, display_conf, candidate_conf):
        if self.detector is None:
            raise ValueError("Set HELMET_WEIGHTS to your trained three-class checkpoint, then restart.")
        if not self.slots.acquire(blocking=False):
            raise RuntimeError("Two jobs are already active. Finish or stop one before starting another.")
        identity = uuid.uuid4().hex
        folder = self.root / identity
        folder.mkdir()
        with self.lock:
            self.jobs[identity] = {"id": identity, "kind": kind, "filename": filename,
                    "status": "queued", "created": now(), "processed_frames": 0, "total_frames": None,
                    "events": [], "detections": [], "error": None, "warning": None,
                    "display_conf": display_conf, "candidate_conf": candidate_conf,
                    "model": self.detector.metadata, "preview": False, "video": None,
                    "capture_mode": "motorcycle_nohelmet_crop",
                    "required_consecutive_frames": 1 if kind == "image" else 5,
                    "capture_rule": ("Still image · single-frame capture" if kind == "image"
                                     else "Video / camera · 5 consecutive NoHelmet frames · once per track") }
            self.cancel[identity] = threading.Event()
            self.update(identity)
        return identity

    def reject_upload(self, identity, message):
        self.update(identity, status="failed", error=message)
        self.slots.release()

    def submit(self, identity, path):
        self.pool.submit(self.process, identity, path)

    def _events(self, identity, frame, events):
        if not events:
            return
        folder = self.root / identity
        height, width = frame.shape[:2]
        with self.lock:
            for event in events:
                # Save the rider-inclusive motorcycle, using original, unobscured pixels.
                box = event.get("bike_box", event.get("head_box", []))
                if len(box) != 4 or not all(math.isfinite(v) for v in box):
                    continue
                x1, y1 = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
                x2, y2 = min(width, math.ceil(box[2])), min(height, math.ceil(box[3]))
                if x2 <= x1 or y2 <= y1:
                    continue
                event["id"] = uuid.uuid4().hex[:12]
                event["snapshot"] = f"event-{event['id']}.jpg"
                event["crop_box"] = [x1, y1, x2, y2]
                save_jpeg(folder / event["snapshot"], frame[y1:y2, x1:x2])
                self.jobs[identity]["events"].append(event)

    def process(self, identity, path):
        folder = self.root / identity
        job = self.get(identity)
        start = time.perf_counter()
        cap, writer = None, None
        count = 0
        last_progress = start
        timings = {"decode": 0.0, "detect_track_draw": 0.0, "video_write": 0.0, "capture": 0.0}
        try:
            self.update(identity, status="running")
            if self.cancel[identity].is_set():
                self.update(identity, status="cancelled")
                return
            if job["kind"] == "image":
                frame = cv2.imread(str(path))
                if frame is None:
                    raise ValueError("The uploaded image cannot be decoded.")
                pipe = Pipeline(self.detector, display_conf=job["display_conf"], candidate_conf=job["candidate_conf"])
                annotated, detections, events = pipe.process(frame, 0, 0, track=False)
                self._events(identity, frame, events)
                save_jpeg(folder / "preview.jpg", annotated)
                write_json(folder / "detections.json", detections)
                self.update(identity, status="completed", preview=True, detections=detections,
                            processed_frames=1, total_frames=1, elapsed_seconds=round(time.perf_counter() - start, 3))
                return
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError("The uploaded video cannot be opened.")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not 1 <= fps <= 240:
                raise ValueError("Video has no usable constant frame rate. Convert it to constant frame rate first.")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.update(identity, fps=fps, total_frames=total if total > 0 else None)
            pipe = Pipeline(self.detector, fps=fps, display_conf=job["display_conf"], candidate_conf=job["candidate_conf"])
            with (folder / "frames.jsonl").open("w") as records:
                while not self.cancel[identity].is_set():
                    step_start = time.perf_counter()
                    success, frame = cap.read()
                    timings["decode"] += time.perf_counter() - step_start
                    if not success:
                        break
                    if count >= 90000 or count / fps >= 900:
                        raise ValueError("This local demo accepts at most 15 minutes or 90,000 video frames per run.")
                    if frame.shape[0] * frame.shape[1] > 3840 * 2160:
                        raise ValueError("Video resolution exceeds the 4K demo limit.")
                    if writer is None:
                        writer = AnnotatedVideoWriter(folder, fps, frame.shape[1], frame.shape[0])
                    timestamp = count / fps
                    step_start = time.perf_counter()
                    annotated, detections, events = pipe.process(frame, count, timestamp)
                    timings["detect_track_draw"] += time.perf_counter() - step_start
                    step_start = time.perf_counter()
                    writer.write(annotated)
                    timings["video_write"] += time.perf_counter() - step_start
                    records.write(json.dumps({"frame_index": count, "timestamp_seconds": timestamp,
                                              "detections": detections}) + "\n")
                    step_start = time.perf_counter()
                    self._events(identity, frame, events)
                    timings["capture"] += time.perf_counter() - step_start
                    count += 1
                    if count == 1:
                        save_jpeg(folder / "preview.jpg", annotated)
                    current = time.perf_counter()
                    if count == 1 or current - last_progress >= 0.5:
                        speed = count / max(current - start, 1e-6)
                        self.update(identity, processed_frames=count, preview=True, detections=detections,
                                    phase="detecting", processed_fps=round(speed, 2),
                                    estimated_remaining_seconds=round(max(0, total - count) / speed) if total > 0 else None)
                        last_progress = current
            if count == 0:
                raise ValueError("Video contained no decodable frames.")
            save_jpeg(folder / "preview.jpg", annotated)
            elapsed = time.perf_counter() - start
            self.update(identity, phase="encoding", processed_frames=count, preview=True,
                        estimated_remaining_seconds=None)
            finish_start = time.perf_counter()
            writer.finish()
            timings["finalize"] = time.perf_counter() - finish_start
            video, browser_video, warning = writer.filename, writer.browser_video, writer.warning
            writer = None
            cancelled = self.cancel[identity].is_set()
            if not cancelled and total > 0 and abs(total - count) > 1:
                warning = (warning + " " if warning else "") + f"Decoder stopped after {count} frames; container reported {total}. Check for a truncated file."
            self.update(identity, status="cancelled" if cancelled else "completed", processed_frames=count,
                        phase="finished", preview=True, video=video, browser_video=browser_video, warning=warning,
                        processing_seconds=round(elapsed, 3), processed_fps=round(count / max(elapsed, 1e-6), 2),
                        stage_seconds={name: round(seconds, 3) for name, seconds in timings.items()},
                        service_seconds=round(time.perf_counter() - start, 3), finished=now(), detections=detections)
        except Exception as exc:
            LOG.exception("Processing failed for %s", identity)
            self.update(identity, status="failed", processed_frames=count, error=str(exc))
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.abort()
            self.slots.release()

    def start_camera(self, display_conf, candidate_conf):
        identity = self.create("camera", "Browser camera", display_conf, candidate_conf)
        try:
            self.sessions[identity] = {"pipeline": Pipeline(self.detector, fps=10, display_conf=display_conf,
                                    candidate_conf=candidate_conf), "lock": threading.Lock(),
                                    "start": time.monotonic(), "last": time.monotonic(), "sequence": -1}
            self.update(identity, status="live")
        except Exception:
            self.reject_upload(identity, "Camera pipeline could not start.")
            raise
        return identity

    def camera_frame(self, identity, frame, sequence):
        session = self.sessions.get(identity)
        if session is None:
            raise ValueError("Camera session ended; start a new session.")
        if not session["lock"].acquire(blocking=False):
            raise RuntimeError("Previous frame is still processing.")
        try:
            if self.get(identity)["status"] != "live" or sequence <= session["sequence"]:
                raise ValueError("Stale frame or closed session.")
            session["sequence"] = sequence
            session["last"] = time.monotonic()
            count = self.get(identity)["processed_frames"]
            timestamp = session["last"] - session["start"]
            annotated, detections, events = session["pipeline"].process(frame, count, timestamp, camera_sequence=sequence)
            save_jpeg(self.root / identity / "preview.jpg", annotated)
            self._events(identity, frame, events)
            with (self.root / identity / "frames.jsonl").open("a") as handle:
                handle.write(json.dumps({"frame_index": count, "received_sequence": sequence,
                                        "timestamp_seconds": timestamp, "detections": detections}) + "\n")
            self.update(identity, processed_frames=count + 1, preview=True, detections=detections)
            return self.get(identity)
        finally:
            session["lock"].release()

    def stop(self, identity):
        job = self.get(identity)
        if job["kind"] == "camera" and identity in self.sessions:
            session = self.sessions[identity]
            with session["lock"]:
                if self.get(identity)["status"] == "live":
                    self.update(identity, status="completed", finished=now())
                    self.slots.release()
                self.sessions.pop(identity, None)
        elif identity in self.cancel:
            self.cancel[identity].set()

    def expire_cameras(self):
        for identity, session in list(self.sessions.items()):
            if time.monotonic() - session["last"] > 60:
                self.stop(identity)

    def review(self, identity, event_id, status, note):
        with self.lock:
            self.get(identity)
            for event in self.jobs[identity]["events"]:
                if event["id"] == event_id:
                    event.update(review=status, note=note, reviewed_at=now())
                    self.update(identity)
                    return event
        raise KeyError(event_id)

    def export(self, identity):
        job = self.get(identity)
        if job["status"] in {"queued", "running", "live"}:
            raise ValueError("Stop or finish this run before exporting a consistent record.")
        folder = self.root / identity
        target = folder / "research_export.zip"
        # Snapshot the review state at export time; do not include the raw uploaded video.
        with self.lock, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("job.json", json.dumps(self.jobs[identity], indent=2))
            stream = io.StringIO()
            fields = ["id", "track_id", "frame_index", "timestamp_seconds", "confidence", "review", "note"]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for event in self.jobs[identity]["events"]:
                safe = dict(event)
                if safe["note"].startswith(("=", "+", "-", "@", "\t", "\r")):
                    safe["note"] = "'" + safe["note"]
                writer.writerow(safe)
            archive.writestr("reviews.csv", stream.getvalue())
            for file in folder.iterdir():
                if file.name in {"preview.jpg", "frames.jsonl", "detections.json"} or file.name.startswith("event-"):
                    archive.write(file, file.name)
        return target

    def close(self):
        for event in self.cancel.values():
            event.set()
        for identity in list(self.sessions):
            self.stop(identity)
        self.pool.shutdown(wait=True, cancel_futures=False)
