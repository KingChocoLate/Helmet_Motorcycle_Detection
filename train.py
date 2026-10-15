"""YOLO11s: one clean + two online variants, guarded mosaic, no rotation."""
import argparse
from functools import partial
from pathlib import Path
import random
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from study_common import (AUGMENTATION, PIXEL_AUGMENTATION, CROP_POLICY, ThreeViewDataset, ThreeViewTrainer,
                          check_version, device, resolve_data, set_positive_weight, calculate_positive_weight,
                          log_crop_stats)


def preview(args, data):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ds = ThreeViewDataset(img_path=data["train"], data=data, imgsz=args.imgsz,
        batch_size=1, augment=True, rect=False, cache=False,
        hyp=get_cfg(overrides=AUGMENTATION))
    requested = 1.0 if args.experiment == "baseline" else args.pos_weight
    loss_settings = calculate_positive_weight(ds.labels, requested)
    print(f"Original training objects: {loss_settings['original_train_objects']}")
    print(f"Preview positive weights ({args.experiment}): {loss_settings['positive_weights']}")
    if all(v == 1.0 for v in loss_settings["positive_weights"]):
        print("All positive weights are 1: ordinary, unweighted classification loss.")
    (Path(args.output) / "loss_settings_preview.json").write_text(json.dumps(loss_settings, indent=2))
    rows = []
    samples = Path(args.output) / "preview_samples"
    samples.mkdir(parents=True, exist_ok=True)
    for index in random.sample(range(len(ds.labels)), min(args.preview_samples, len(ds.labels))):
        tiles = []
        for view in range(3):
            before = ds.mosaic_transform.summary()["applied"]
            sample = ds[index + view * len(ds.labels)]
            used_mosaic = ds.mosaic_transform.summary()["applied"] > before
            image = sample["img"].permute(1, 2, 0).numpy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            height, width = image.shape[:2]
            for cls, (x, y, w, h) in zip(sample["cls"].flatten(), sample["bboxes"]):
                a = (round(float(x-w/2)*width), round(float(y-h/2)*height))
                b = (round(float(x+w/2)*width), round(float(y+h/2)*height))
                cv2.rectangle(image, a, b, [(255,255,0),(0,255,0),(0,80,255)][int(cls)], 2)
            cv2.imwrite(str(samples / f"source_{index}_view_{view}.png"), image)
            image = cv2.resize(image, (480, 480))
            image = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT)
            label = ["Clean", "Variant 1", "Variant 2"][view] + (" + mosaic" if used_mosaic else "")
            cv2.putText(image, label, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
            tiles.append(image)
        rows.append(np.concatenate(tiles, axis=1))
    target = Path(args.output) / "preview.jpg"
    cv2.imwrite(str(target), np.concatenate(rows))
    print(f"Full-resolution boxed views: {samples}")
    print(f"Saved {target}; {len(ds.labels)} originals -> {len(ds)} visits per epoch")
    stats = {"geometry": ds.geometry.summary(), "mosaic": ds.mosaic_transform.summary()}
    (Path(args.output) / "crop_stats_preview.json").write_text(json.dumps(stats, indent=2))
    (Path(args.output) / "augmentation_preview.json").write_text(json.dumps({
        "augmentation": AUGMENTATION, "pixel_augmentation": PIXEL_AUGMENTATION,
        "crop_policy": CROP_POLICY}, indent=2))
    print(f"Preview geometry fallback: {stats['geometry']['fallback_fraction']:.1%}; "
          f"mosaic applied: {stats['mosaic']['applied_fraction']:.1%}. These are sampled views only.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--mode", choices=["train", "preview"], default="train")
    p.add_argument("--preview-samples", type=int, default=8)
    p.add_argument("--experiment", choices=["both", "baseline", "weighted"], default="baseline")
    p.add_argument("--weights", default="yolo11s.pt")
    p.add_argument("--pos-weight", default="auto",
                   help="Weighted run: auto calculates sqrt(max_count/class_count) for all 3 classes, without a cap. A number overrides NoHelmet only. Baseline uses [1,1,1].")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="runs/helmet_combined_augmentation")
    p.add_argument("--resume", default=None,
               help="Path to an interrupted run's weights/last.pt")
    args = p.parse_args()
    if args.epochs < 2 or args.preview_samples < 1:
        p.error("Use at least 2 epochs and at least 1 preview sample.")
    if args.pos_weight.lower() != "auto":
        try:
            value = float(args.pos_weight)
        except ValueError:
            p.error("--pos-weight must be auto or a finite positive number")
        if not np.isfinite(value) or value <= 0:
            p.error("--pos-weight must be auto or a finite positive number")
    check_version()
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.is_file() or resume_path.name != "last.pt":
            p.error("--resume must point to an existing weights/last.pt")

        run_dir = resume_path.parent.parent
        loss_file = run_dir / "loss_settings.json"

        if not loss_file.is_file():
            raise FileNotFoundError(
                f"Cannot safely resume: missing {loss_file}"
            )

        saved_loss = json.loads(loss_file.read_text())

        # Recover the SAME loss configuration used before interruption.
        if saved_loss["method"] == "sqrt_max_class_count_over_class_count":
            resume_weight = "auto"
            resume_experiment = "weighted"
        else:
            resume_weight = float(saved_loss["requested"])
            resume_experiment = (
                "baseline" if resume_weight == 1.0 else "weighted"
            )

        print(f"Resuming: {resume_path}")
        print(f"Experiment: {resume_experiment}")
        print(f"Previous positive weights: {saved_loss['positive_weights']}")

        model = YOLO(str(resume_path))

        # Reattach our custom loss and crop-stat callback.
        model.add_callback(
            "on_pretrain_routine_end",
            partial(set_positive_weight, weight=resume_weight),
        )
        model.add_callback("on_train_epoch_end", log_crop_stats)

        model.train(
            trainer=ThreeViewTrainer,
            resume=True,
        )

        print(f"Resume completed from {resume_path}")
        return
    data_yaml, data = resolve_data(args.data, args.output)
    if args.mode == "preview":
        preview(args, data)
        return
    experiments = ["baseline", "weighted"] if args.experiment == "both" else [args.experiment]
    selected_device = device(args.device)
    augmentation = dict(AUGMENTATION)
    # Last 10% of epochs without mosaic, capped at 20 (100 epochs -> 20).
    augmentation["close_mosaic"] = min(AUGMENTATION["close_mosaic"], max(1, args.epochs // 5))
    for experiment in experiments:
        name = f"{experiment}_s{args.seed}"
        if (Path(args.output) / name).exists():
            raise FileExistsError(f"{name} already exists; use a new --output folder.")
        # Each run reloads the SAME initial weights, never the baseline best.pt.
        model = YOLO(args.weights)
        if len(model.names) != 80 or model.model.yaml.get("scale") != "s" or "C3k2" not in str(model.model):
            raise ValueError("Use the official COCO-pretrained yolo11s.pt checkpoint.")
        weight = 1.0 if experiment == "baseline" else args.pos_weight
        model.add_callback("on_pretrain_routine_end", partial(set_positive_weight, weight=weight))
        model.add_callback("on_train_epoch_end", log_crop_stats)
        model.train(trainer=ThreeViewTrainer, data=str(data_yaml),
            project=str(Path(args.output).resolve()), name=name,
            epochs=args.epochs, patience=args.epochs, imgsz=args.imgsz,
            batch=args.batch, seed=args.seed, workers=args.workers,
            device=selected_device, deterministic=True, optimizer="auto",
            amp=selected_device != "cpu", cache=False, rect=False,
            cls_pw=0.0, plots=True, **augmentation)
        (Path(model.trainer.save_dir) / "study.json").write_text(json.dumps({
            "positive_weights": model.trainer.helmet_loss_settings["positive_weights"],
            "loss_settings": model.trainer.helmet_loss_settings, "views_per_original": 3,
            "pixel_augmentation": PIXEL_AUGMENTATION, "crop_policy": CROP_POLICY,
            "initial_weights": args.weights, "arguments": vars(args),
            "augmentation": augmentation}, indent=2))


if __name__ == "__main__":
    main()