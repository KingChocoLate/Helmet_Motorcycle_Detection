"""Three-view YOLO training with guarded mosaic and optional class weighting."""
import json
import multiprocessing as mp
import random
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose, Format, LetterBox, Mosaic, RandomPerspective, RandomHSV, RandomFlip
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import unwrap_model

NAMES = ["BikeWithRider", "Helmet", "NoHelmet"]
AUGMENTATION = dict(
    hsv_h=0.015, hsv_s=0.70, hsv_v=0.40, fliplr=0.5, flipud=0.0,
    translate=0.10, scale=0.50, degrees=0.0, shear=0.0, perspective=0.0,
    mosaic=0.50, mixup=0.0, cutmix=0.0, copy_paste=0.0, close_mosaic=20, bgr=0.0,
)


# Applied independently to each augmented view; the clean view bypasses these.
PIXEL_AUGMENTATION = dict(blur_probability=0.20, blur_sigma=[0.3, 1.0],
                          noise_probability=0.15, noise_density=[0.0, 0.0152])
CROP_POLICY = dict(min_bike_visibility=0.70,
                   min_head_visibility=0.60, max_top_crop=0.20,
                   max_attempts=8, mosaic_attempts=5, allow_fully_outside=True,
                   fallback="fit_whole_input_without_more_cropping")


def device(value):
    if value == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if "," in value:
        raise ValueError("Use one GPU or cpu for this simple kit.")
    return value


def check_version():
    if ultralytics.__version__ != "8.4.115":
        raise RuntimeError("This loss/data extension requires ultralytics==8.4.115.")


def check_names(names):
    if isinstance(names, dict):
        names = [names[i] if i in names else names[str(i)] for i in range(len(names))]
    if names != NAMES:
        raise ValueError(f"Expected class IDs 0,1,2 = {NAMES}; found {names}")


def resolve_data(source, output, required_splits=("train", "val", "test")):
    """Resolve dataset paths, strictly checking only the requested splits.

    Training callers retain the original strict behavior by default. Evaluation
    callers can request, for example, required_splits=("test",), which permits
    Roboflow test-only exports that list absent train/val directories.
    Unrequested split paths are retained in the generated YAML because the
    Ultralytics detection-dataset schema requires train and val keys, even when
    split="test". Those retained paths are NOT evaluated or used for training.
    """
    required = set(required_splits)
    valid = {"train", "val", "test"}
    if not required or not required.issubset(valid):
        raise ValueError(f"required_splits must be a nonempty subset of {sorted(valid)}.")

    source = Path(source).expanduser().resolve()
    data = yaml.safe_load(source.read_text())
    check_names(data.get("names"))
    root = Path(data.get("path") or source.parent).expanduser()
    if not root.is_absolute():
        root = source.parent / root

    result = {"names": NAMES}
    for split in ("train", "val", "test"):
        if not data.get(split):
            if split in required:
                raise ValueError(f"data.yaml is missing required {split} images directory.")
            continue

        p = Path(data[split]).expanduser()
        candidates = [p] if p.is_absolute() else [root / p, source.parent / p]
        if not p.is_absolute() and p.parts[0] == "..":
            candidates.append(source.parent.joinpath(*p.parts[1:]))
        found = {candidate.resolve() for candidate in candidates if candidate.is_dir()}
        if len(found) > 1:
            raise ValueError(f"Set {split} to an unambiguous absolute images directory in data.yaml.")
        if len(found) == 1:
            result[split] = str(found.pop())
        elif split in required:
            raise ValueError(f"Set {split} to an existing absolute images directory in data.yaml.")
        else:
            # Preserve an unrequested path as a nonvalidated YAML placeholder.
            # Never substitute test images as train/val images.
            if p.is_absolute():
                placeholder = p
            elif p.parts[0] == "..":
                placeholder = source.parent.joinpath(*p.parts[1:])
            else:
                placeholder = root / p
            result[split] = str(placeholder.resolve())

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "resolved_data.yaml"
    target.write_text(yaml.safe_dump(result, sort_keys=False))
    return target, result


def clip_boxes(boxes, size):
    clipped = boxes.copy()
    width, height = size
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, width)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, height)
    return clipped


def crop_ok(full_boxes, visible_boxes, classes):
    """Reject misleading partial crops while allowing objects to leave the frame completely.

    BikeWithRider is a context class, so any still-visible instance must retain at
    least 80% of its transformed box. Helmet/NoHelmet keep the existing 60% area
    rule plus the 20% maximum crop from the top. Full boxes are retained through
    mosaic so sequential mosaic + affine crops are checked cumulatively.
    """
    class_ids = classes.reshape(-1).astype(np.int64)
    dimensions = np.maximum(full_boxes[:, 2:] - full_boxes[:, :2], 0)
    full_area = np.prod(dimensions, axis=1)
    visible_area = np.prod(np.maximum(visible_boxes[:, 2:] - visible_boxes[:, :2], 0), axis=1)
    fraction = visible_area / np.maximum(full_area, 1e-12)

    good = np.ones(len(class_ids), dtype=bool)

    bikes = class_ids == 0
    good[bikes] = fraction[bikes] >= CROP_POLICY["min_bike_visibility"] - 1e-6

    heads = np.isin(class_ids, [1, 2])
    top_crop = np.maximum(visible_boxes[:, 1] - full_boxes[:, 1], 0) / np.maximum(dimensions[:, 1], 1e-12)
    good[heads] = ((fraction[heads] >= CROP_POLICY["min_head_visibility"] - 1e-6) &
                   (top_crop[heads] <= CROP_POLICY["max_top_crop"] + 1e-6))

    outside = (visible_area == 0) & CROP_POLICY["allow_fully_outside"]
    return bool(np.all(good | outside))


class HeadAwareMosaic(Mosaic):
    """Native four-image mosaic, checked BEFORE pixels or labels are clipped."""
    def __init__(self, dataset, imgsz, p, counts):
        super().__init__(dataset, imgsz=imgsz, p=p, n=4)
        self.counts = counts
        self.fit = LetterBox(new_shape=(imgsz, imgsz), scaleup=True)

    def get_indexes(self):
        # __len__ is tripled, but get_image_and_label needs ORIGINAL indices.
        return random.choices(range(len(self.dataset.labels)), k=3)

    def record(self, selected=False, applied=False, proposals=0, fallback=False):
        with self.counts.get_lock():
            for i, value in enumerate([1, selected, applied, proposals, fallback]):
                self.counts[i] += int(value)

    def summary(self):
        with self.counts.get_lock():
            views, selected, applied, proposals, fallbacks = self.counts[:]
        return {"processed_augmented_views": views, "selected": selected,
                "applied": applied, "layout_proposals": proposals, "fallbacks": fallbacks,
                "applied_fraction": applied / views if views else 0.0,
                "fallback_fraction_of_selected": fallbacks / selected if selected else 0.0}

    def __call__(self, labels):
        if self.p == 0 or random.random() >= self.p:
            self.record()
            return self.fit(labels)
        for attempt in range(1, CROP_POLICY["mosaic_attempts"] + 1):
            trial = labels.copy()
            trial["instances"] = deepcopy(labels["instances"])
            params = super().get_params(trial)
            boxes, classes = [], []
            for item in params["layout"]:
                patch = item["labels_patch"]
                instances = deepcopy(patch["instances"])
                instances.convert_bbox("xyxy")
                height, width = item["img_shape"]
                instances.denormalize(width, height)
                instances.add_padding(item["padw"], item["padh"])
                boxes.append(instances.bboxes.copy())
                classes.append(patch["cls"])
            reference = np.concatenate(boxes, axis=0)
            classes = np.concatenate(classes, axis=0)
            clipped = clip_boxes(reference, (self.imgsz * 2, self.imgsz * 2))
            if not crop_ok(reference, clipped, classes):
                continue
            trial = self.apply_image(trial, params)
            trial = self.apply_instances(trial, params)
            # Native Mosaic removes only zero-area boxes; align full references.
            keep = np.prod(np.maximum(clipped[:, 2:] - clipped[:, :2], 0), axis=1) > 0
            trial["head_reference"] = reference[keep]
            trial.pop("mix_labels", None)
            self.record(selected=True, applied=True, proposals=attempt)
            return trial
        self.record(selected=True, proposals=CROP_POLICY["mosaic_attempts"], fallback=True)
        return self.fit(labels)


class HeadAwareAffine(RandomPerspective):
    """Native scale/translation with cumulative mosaic + affine crop checking.

    Detection boxes only; rotation/shear/perspective are disabled by the recipe.
    Visibility is clipped area / transformed area BEFORE clipping, so normal
    shrinking is not mistaken for cropping. Fully outside heads are allowed.
    """
    def __init__(self, *args, counts, **kwargs):
        super().__init__(*args, **kwargs)
        # Shared by DataLoader workers, so fallback counts cover their views too.
        self.counts = counts
        self.fit = LetterBox(new_shape=(self.size[1], self.size[0]), scaleup=True)

    def record(self, attempts, fallback):
        with self.counts.get_lock():
            self.counts[0] += 1
            self.counts[1] += attempts
            self.counts[2] += int(fallback)

    def summary(self):
        with self.counts.get_lock():
            views, proposals, fallbacks = self.counts[:]
        return {"processed_augmented_views": views, "geometry_proposals": proposals,
                "fallbacks": fallbacks,
                "fallback_fraction": fallbacks / views if views else 0.0}

    def acceptable(self, boxes, classes, params, reference=None):
        protected = np.isin(classes.reshape(-1), [0, 1, 2])
        if not protected.any():
            return True
        original = boxes
        transformed = self.apply_bboxes(original, params["M"])
        clipped = clip_boxes(transformed, params["size"])
        full = self.apply_bboxes(reference if reference is not None else boxes, params["M"])
        if not crop_ok(full, clipped, classes):
            return False
        visible = np.prod(np.maximum(clipped[protected, 2:] - clipped[protected, :2], 0), axis=1) > 0
        # Also retry if native filtering would discard a still-visible protected object.
        retained = self.box_candidates(
            box1=(original[protected] * params["scale"]).T,
            box2=clipped[protected].T,
            area_thr=0.10,
        )
        return bool(np.all((~visible & CROP_POLICY["allow_fully_outside"]) | retained))

    def __call__(self, labels):
        instances = deepcopy(labels["instances"])
        instances.convert_bbox("xyxy")
        instances.denormalize(labels["img"].shape[1], labels["img"].shape[0])
        for attempt in range(1, CROP_POLICY["max_attempts"] + 1):
            params = super().get_params(labels)
            if self.acceptable(instances.bboxes, labels["cls"], params, labels.get("head_reference")):
                self.record(attempt, False)
                # Warp pixels only once, after the geometry has been accepted.
                labels = self.apply_image(labels, params)
                labels = self.apply_instances(labels, params)
                labels.pop("head_reference", None)
                return labels
        # Fit a 2*imgsz mosaic to imgsz instead of silently returning a large frame.
        # This preserves all currently visible labels without further cropping.
        self.record(CROP_POLICY["max_attempts"], True)
        labels.pop("head_reference", None)
        return self.fit(labels)


class BlurAndNoise:
    """Independent, occasional Gaussian blur and Roboflow-style pixel noise."""
    def __call__(self, labels):
        if np.random.random() < PIXEL_AUGMENTATION["blur_probability"]:
            sigma = float(np.random.uniform(*PIXEL_AUGMENTATION["blur_sigma"]))
            labels["img"] = cv2.GaussianBlur(labels["img"], (0, 0), sigmaX=sigma)
        if np.random.random() < PIXEL_AUGMENTATION["noise_probability"]:
            image = labels["img"].copy()
            height, width, channels = image.shape
            density = np.random.uniform(*PIXEL_AUGMENTATION["noise_density"])
            count = round(height * width * density)
            if count:
                positions = np.random.choice(height * width, count, replace=False)
                image.reshape(-1, channels)[positions] = np.random.choice([0, 255], (count, 1))
            labels["img"] = image
        return labels


class ThreeViewDataset(YOLODataset):
    """One clean + two independently augmented visits per original per epoch."""
    def __len__(self):
        return len(self.labels) * (3 if self.augment else 1)

    def build_transforms(self, hyp=None):
        if not self.augment:
            return super().build_transforms(hyp)
        disabled = ("degrees", "shear", "perspective", "flipud",
                    "mixup", "cutmix", "copy_paste", "bgr")
        if any(getattr(hyp, name, 0) != 0 for name in disabled) or self.rect:
            raise ValueError("Rotation, shear, perspective, vertical flip and extra mixing must stay off; rect=False.")
        # Explicit native transforms keep optional Albumentations extras out.
        self.clean = Compose([
            LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=True),
            Format(bbox_format="xywh", normalize=True, batch_idx=True, bgr=0.0),
        ])
        # Preserve shared counters when native close_mosaic rebuilds transforms.
        if not hasattr(self, "geometry_counts"):
            self.geometry_counts = mp.get_context("spawn").Array("q", 3)
            self.mosaic_counts = mp.get_context("spawn").Array("q", 5)
        self.mosaic_transform = HeadAwareMosaic(self, self.imgsz, hyp.mosaic, self.mosaic_counts)
        self.geometry = HeadAwareAffine(counts=self.geometry_counts,
            degrees=0, translate=hyp.translate, scale=hyp.scale,
            shear=0, perspective=0, size=(self.imgsz, self.imgsz))
        return Compose([
            self.mosaic_transform,
            self.geometry,
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(p=hyp.fliplr, direction="horizontal"),
            BlurAndNoise(),
            Format(bbox_format="xywh", normalize=True, batch_idx=True, bgr=0.0),
        ])

    def __getitem__(self, index):
        if not self.augment:
            return super().__getitem__(index)
        view, original = divmod(index, len(self.labels))
        labels = self.get_image_and_label(original)
        return (self.clean if view == 0 else self.transforms)(labels)


class ThreeViewTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        return ThreeViewDataset(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
            augment=True, hyp=self.args, rect=False, cache=False,
            stride=max(int(unwrap_model(self.model).stride.max()), 32),
            pad=0.0, prefix="train: ", task="detect", data=self.data,
        )


def log_crop_stats(trainer):
    """Cumulative worker counts; prefetch can include views ahead of the epoch."""
    dataset = trainer.train_loader.dataset
    geometry = dataset.geometry.summary()
    mosaic = dataset.mosaic_transform.summary()
    stats = {"epoch": trainer.epoch + 1, "active_mosaic_probability": dataset.mosaic_transform.p,
             "geometry": geometry, "mosaic": mosaic,
             "note": "Cumulative processed worker views; prefetch may run ahead of consumed batches."}
    with (Path(trainer.save_dir) / "crop_stats.jsonl").open("a") as output:
        output.write(json.dumps(stats) + "\n")
    print(f"Geometry fallback: {geometry['fallback_fraction']:.1%}; "
          f"mosaic applied: {mosaic['applied_fraction']:.1%}; active p={dataset.mosaic_transform.p}")


def calculate_positive_weight(labels, requested="auto"):
    """Count ORIGINAL training objects once, before the three-view augmentation.

    Auto uses sqrt(maximum class count / each class count), without a cap.
    This is a study heuristic, not the exact
    negative/positive target ratio of YOLO's dense classification loss.
    It estimates imbalance, not the relative real-world cost of errors.
    """
    counts = np.zeros(3, dtype=np.int64)
    for label in labels:
        ids = np.asarray(label["cls"]).reshape(-1)
        if not np.isin(ids, [0, 1, 2]).all():
            raise ValueError("Training labels must use class IDs 0, 1 and 2.")
        counts += np.bincount(ids.astype(np.int64), minlength=3)
    if (counts == 0).any():
        raise ValueError(f"All three classes need training objects; found {dict(zip(NAMES, counts.tolist()))}.")
    automatic = str(requested).lower() == "auto"
    if automatic:
        values = np.sqrt(counts.max() / counts).tolist()
    else:
        # Preserve the earlier optional numeric override for NoHelmet only.
        value = float(requested)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("--pos-weight must be auto or a finite positive number.")
        values = [1.0, 1.0, value]
    return {
        "method": "sqrt_max_class_count_over_class_count" if automatic else "fixed_nohelmet",
        "requested": str(requested),
        "original_train_objects": dict(zip(NAMES, counts.tolist())),
        "class_order": NAMES,
        "reference_class_count": int(counts.max()) if automatic else None,
        "formula": "sqrt(max_count / class_count), no cap" if automatic else "[1, 1, requested]",
        "positive_weights": values,
        "count_source": "original training labels; no augmented, validation or test labels",
        "constant_during_training": True,
    }


def set_positive_weight(trainer, weight):
    """Apply per-class positive BCE weights; auto balances all three classes."""
    settings = calculate_positive_weight(trainer.train_loader.dataset.labels, weight)
    values = settings["positive_weights"]
    for model in (unwrap_model(trainer.model), trainer.ema.ema):
        check_names(model.names)
        loss = v8DetectionLoss(model)
        loss.bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(values, device=next(model.parameters()).device),
            reduction="none",
        )
        model.criterion = loss
    trainer.helmet_loss_settings = settings
    (Path(trainer.save_dir) / "loss_settings.json").write_text(json.dumps(settings, indent=2))
    print(f"Original training objects: {settings['original_train_objects']}")
    print(f"BCE positive weights: {values}; training views: {len(trainer.train_loader.dataset)}")
    if settings["method"] == "sqrt_max_class_count_over_class_count" and all(v == 1.0 for v in values):
        print("NOTE: All class counts are equal. Auto weights equal the baseline loss.")