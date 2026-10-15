"""Evaluate one or more checkpoints on the same validation or test split."""
import argparse
import csv
from pathlib import Path
from ultralytics import YOLO
from study_common import NAMES, check_names, check_version, device, resolve_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--weights", nargs="+", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="runs/helmet_evaluation")
    args = p.parse_args()
    check_version()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Use a new --output folder to keep previous results.")
    data_yaml, data = resolve_data(args.data, output, required_splits=(args.split,))
    if args.split not in data:
        raise ValueError(f"data.yaml is missing {args.split}")
    rows = []
    for index, checkpoint in enumerate(args.weights):
        model = YOLO(checkpoint)
        check_names(model.names)
        metrics = model.val(data=str(data_yaml), split=args.split,
            imgsz=args.imgsz, batch=args.batch, device=device(args.device),
            workers=args.workers, conf=.001, iou=.7, max_det=300,
            augment=False, half=False, plots=True,
            project=str(output), name=f"model_{index+1}")
        indexes = {int(cls): i for i, cls in enumerate(metrics.box.ap_class_index)}
        for cls, name in enumerate(NAMES):
            # Blank metrics indicate a class absent from evaluation ground truth.
            values = [None]*4 if cls not in indexes else list(map(float, metrics.box.class_result(indexes[cls])))
            rows.append(dict(checkpoint=str(Path(checkpoint).resolve()), split=args.split,
                class_name=name, precision=values[0], recall=values[1],
                AP50=values[2], AP50_95=values[3]))
    with (output / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output / 'comparison.csv'}")


if __name__ == "__main__":
    main()
