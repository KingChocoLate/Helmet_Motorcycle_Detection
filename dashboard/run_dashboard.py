"""Start the local dashboard from any working directory.

Examples:
    python run_dashboard.py --weights /path/to/best.pt --device 0
    python ../run_dashboard.py --weights /path/to/best.pt --device cpu
"""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="Trained three-class YOLO checkpoint (.pt)")
    parser.add_argument("--device", help="auto selects CUDA when available; 0 forces GPU; cpu forces CPU (default: HELMET_DEVICE or auto)")
    parser.add_argument("--imgsz", type=int, help="Detection input size (default: 960)")
    parser.add_argument("--host", default="127.0.0.1", help="Local bind address")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.weights is not None:
        weights = args.weights.expanduser().resolve()
        if not weights.is_file() or weights.suffix.lower() != ".pt":
            parser.error(f"checkpoint does not exist or is not a .pt file: {weights}")
        os.environ["HELMET_WEIGHTS"] = str(weights)
    if args.device is not None:
        os.environ["HELMET_DEVICE"] = args.device
    if args.imgsz is not None:
        if args.imgsz < 32:
            parser.error("--imgsz must be at least 32")
        os.environ["HELMET_IMGSZ"] = str(args.imgsz)

    import uvicorn
    uvicorn.run("app:app", host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()

