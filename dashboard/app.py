import asyncio
import io
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from procedures.jobs import JobStore

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
# Put your preferred trained checkpoint here; no command-line flag is needed.
DEFAULT_WEIGHTS = Path("/home/sophat-odom/Documents/Path_to_AI/helmet_detection/src/dashboard/weights/best.pt")
MAX_UPLOAD = 250 * 1024 * 1024


def decode_image(contents):
    try:
        with Image.open(io.BytesIO(contents)) as image:
            if image.width * image.height > 20_000_000:
                raise ValueError("Image exceeds 20 megapixels.")
            if image.getexif().get(274, 1) != 1:
                from PIL import ImageOps
                image = ImageOps.exif_transpose(image)
            return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise HTTPException(400, f"Cannot read this image: {exc}") from exc


def enlarged_evidence_jpeg(original, longest_side=1200):
    """Return a larger JPEG for convenient saving; never alter stored evidence.

    Enlarging a small crop increases display size, not underlying detail.
    Already-large crops are returned byte-for-byte unchanged.
    """
    image = cv2.imdecode(np.frombuffer(original, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(422, "Evidence image cannot be decoded.")
    height, width = image.shape[:2]
    original_long_side = max(height, width)
    if original_long_side >= longest_side:
        return original
    scale = longest_side / original_long_side
    enlarged = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_CUBIC,
    )
    success, encoded = cv2.imencode(".jpg", enlarged, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise HTTPException(500, "Could not prepare enlarged evidence image.")
    return encoded.tobytes()


class Options(BaseModel):
    display_conf: float = Field(0.45, ge=0.1, le=0.95)
    candidate_conf: float = Field(0.6, ge=0.1, le=0.95)


class Review(BaseModel):
    status: str = Field(pattern="^(accepted|rejected|uncertain|pending)$")
    note: str = Field(default="", max_length=1000)


def create_app(store=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.model_lock = asyncio.Lock()
        app.state.default_weights = DEFAULT_WEIGHTS.expanduser().resolve()
        app.state.model_error = None
        if store is not None:
            app.state.store = store
        else:
            detector = None
            model_error = None
            weights = app.state.default_weights
            if weights:
                try:
                    from procedures.inference import Detector
                    detector = Detector(weights, os.getenv("HELMET_DEVICE", "auto"), int(os.getenv("HELMET_IMGSZ", "960")))
                except Exception as exc:
                    model_error = str(exc)
            app.state.model_error = model_error
            app.state.store = JobStore(os.getenv("HELMET_RUNS", str(PROJECT_ROOT / "artifacts/dashboard")), detector)

        async def reap():
            while True:
                await asyncio.sleep(15)
                await run_in_threadpool(app.state.store.expire_cameras)

        task = asyncio.create_task(reap())
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await run_in_threadpool(app.state.store.close)

    app = FastAPI(title="Cambodian helmet study", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")

    @app.exception_handler(ValueError)
    async def bad_request(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(RuntimeError)
    async def busy(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Run or event not found."}, status_code=404)

    @app.middleware("http")
    async def length_limit(request, call_next):
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > MAX_UPLOAD + 1024 * 1024):
            return JSONResponse({"detail": "Upload limit is 250 MB."}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index(request: Request):
        return templates.TemplateResponse(request=request, name="index.html")

    @app.get("/api/status")
    def status(request: Request):
        store = request.app.state.store
        return {"ready": store.detector is not None,
                "model": store.detector.metadata if store.detector else None,
                "switching": request.app.state.model_lock.locked(),
                "default_checkpoint": request.app.state.default_weights.name,
                "detail": getattr(request.app.state, "model_error", None) or
                          (None if store.detector else "Place best.pt in the weights folder, or upload a checkpoint below.")}

    def check_model_request(request):
        # Same-origin browser requests only; checkpoint loading is a local operation.
        if request.headers.get("X-Helmet-Model") != "1":
            raise HTTPException(403, "Use the dashboard model controls.")
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Model changes must come from this dashboard.")
        if request.app.state.model_lock.locked():
            raise HTTPException(409, "Another checkpoint is loading.")
        with request.app.state.store.lock:
            if any(j["status"] in {"queued", "running", "live"}
                   for j in request.app.state.store.jobs.values()):
                raise HTTPException(409, "Finish or stop active image, video or camera runs first.")

    async def load_model(request, path, label):
        from procedures.inference import Detector
        try:
            detector = await run_in_threadpool(
                Detector, path, os.getenv("HELMET_DEVICE", "auto"),
                int(os.getenv("HELMET_IMGSZ", "960")))
        except Exception as exc:
            raise HTTPException(400, f"Could not load checkpoint: {exc}. Previous model kept.") from exc
        detector.metadata["checkpoint"] = label
        with request.app.state.store.lock:
            request.app.state.store.detector = detector
        request.app.state.model_error = None

    @app.post("/api/model/upload")
    async def upload_model(request: Request, file: UploadFile = File(...)):
        path = None
        loaded = False
        try:
            check_model_request(request)
            async with request.app.state.model_lock:
                filename = Path((file.filename or "").replace("\\", "/")).name[:160]
                if Path(filename).suffix.lower() != ".pt":
                    raise HTTPException(400, "Choose a trained YOLO .pt checkpoint.")
                directory = request.app.state.store.root / "uploaded_weights"
                directory.mkdir(exist_ok=True)
                path = directory / f"{uuid.uuid4().hex}.pt"
                size = 0
                with path.open("wb") as output:
                    while chunk := await file.read(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_UPLOAD:
                            raise HTTPException(413, "Checkpoint upload limit is 250 MB.")
                        output.write(chunk)
                if not size:
                    raise HTTPException(400, "Checkpoint file is empty.")
                await load_model(request, path, filename)
                loaded = True
            return status(request)
        finally:
            await file.close()
            if path is not None and not loaded:
                path.unlink(missing_ok=True)

    @app.post("/api/model/default")
    async def default_model(request: Request):
        check_model_request(request)
        async with request.app.state.model_lock:
            path = request.app.state.default_weights
            await load_model(request, path, path.name)
        return status(request)

    @app.get("/api/jobs")
    def jobs(request: Request):
        store = request.app.state.store
        with store.lock:
            return sorted([{"id": j["id"], "filename": j["filename"], "status": j["status"], "created": j["created"]}
                           for j in store.jobs.values()], key=lambda j: j["created"], reverse=True)[:100]

    @app.post("/api/jobs")
    async def upload(request: Request, file: UploadFile = File(...), kind: str = Form(...),
                     display_conf: float = Form(0.45), candidate_conf: float = Form(0.6)):
        if kind not in {"image", "video"}:
            raise HTTPException(400, "Choose image or video.")
        if not (0.1 <= display_conf <= 0.95 and 0.1 <= candidate_conf <= 0.95):
            raise HTTPException(400, "Confidence must be between 0.1 and 0.95.")
        store = request.app.state.store
        filename = Path((file.filename or "upload").replace("\\", "/")).name[:160]
        if request.app.state.model_lock.locked():
            raise HTTPException(409, "Wait for the checkpoint to finish loading.")
        async with request.app.state.model_lock:
            identity = store.create(kind, filename, display_conf, candidate_conf)
        path = store.root / identity / ("input.jpg" if kind == "image" else "input.mp4")
        size, limit = 0, 16 * 1024 * 1024 if kind == "image" else MAX_UPLOAD
        try:
            with path.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"Upload exceeds {limit // (1024 * 1024)} MB.")
                    output.write(chunk)
            if kind == "image":
                frame = await run_in_threadpool(decode_image, path.read_bytes())
                from procedures.inference import save_jpeg
                await run_in_threadpool(save_jpeg, path, frame)
            store.submit(identity, path)
        except Exception as exc:
            path.unlink(missing_ok=True)
            store.reject_upload(identity, str(exc))
            raise
        finally:
            await file.close()
        return {"id": identity}

    @app.get("/api/jobs/{identity}")
    def job(request: Request, identity: str):
        return request.app.state.store.get(identity)

    @app.post("/api/jobs/{identity}/stop")
    def stop(request: Request, identity: str):
        request.app.state.store.stop(identity)
        return {"status": "stop requested"}

    @app.put("/api/jobs/{identity}/events/{event_id}")
    def review(request: Request, identity: str, event_id: str, body: Review):
        return request.app.state.store.review(identity, event_id, body.status, body.note)

    @app.get("/api/jobs/{identity}/files/{filename}")
    def artifact(request: Request, identity: str, filename: str, enlarge: bool = False):
        store = request.app.state.store
        job = store.get(identity)
        allowed = {"preview.jpg", job.get("video")}
        allowed.update(e["snapshot"] for e in job["events"])
        if filename not in allowed:
            raise HTTPException(404, "Artifact not found.")
        path = store.root / identity / filename
        if not path.is_file():
            raise HTTPException(404, "Artifact is not ready.")
        # Save image downloads for both still images and video/camera frames.
        # Keep the original snapshot on disk and in the research export.
        if enlarge and filename in {e["snapshot"] for e in job["events"]}:
            return Response(content=enlarged_evidence_jpeg(path.read_bytes()),
                            media_type="image/jpeg")
        return FileResponse(path)

    @app.get("/api/jobs/{identity}/export")
    def export(request: Request, identity: str):
        path = request.app.state.store.export(identity)
        return FileResponse(path, filename=f"helmet-research-{identity[:8]}.zip")

    @app.post("/api/camera")
    async def camera(request: Request, body: Options):
        if request.app.state.model_lock.locked():
            raise HTTPException(409, "Wait for the checkpoint to finish loading.")
        async with request.app.state.model_lock:
            return {"id": request.app.state.store.start_camera(body.display_conf, body.candidate_conf)}

    @app.post("/api/camera/{identity}/frame")
    async def camera_frame(request: Request, identity: str, sequence: int = Form(...), file: UploadFile = File(...)):
        contents = await file.read(8 * 1024 * 1024 + 1)
        await file.close()
        if len(contents) > 8 * 1024 * 1024:
            raise HTTPException(413, "Camera frame exceeds 8 MB.")
        frame = await run_in_threadpool(decode_image, contents)
        return await run_in_threadpool(request.app.state.store.camera_frame, identity, frame, sequence)

    return app


app = create_app()
