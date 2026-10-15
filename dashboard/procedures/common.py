import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from procedures import CLASS_NAMES


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    packages = {}
    for name in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "fastapi"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def check_names(names):
    ordered = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    if ordered != list(CLASS_NAMES):
        raise ValueError(f"Expected class order {CLASS_NAMES}; checkpoint/dataset has {ordered}.")
