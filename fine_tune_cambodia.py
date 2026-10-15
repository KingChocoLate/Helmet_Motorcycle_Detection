"""Controlled 160/40 Cambodian adaptation of a previously Indian-trained 3-class YOLO11s.

Keep train.py and the 100-epoch Indian experiment unchanged.  Requires the
project's tested study_common.py and ultralytics==8.4.115 environment.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

from study_common import (
    AUGMENTATION, CROP_POLICY, PIXEL_AUGMENTATION, NAMES,
    ThreeViewTrainer, check_names, check_version, device,
    log_crop_stats, resolve_data, set_positive_weight,
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def count_objects(images_dir):
    """Read the independent original annotations (NOT the augmented visits)."""
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
    if not images:
        raise ValueError(f'No images: {images_dir}')
    label_dir = images_dir.parent / 'labels'
    counts = [0, 0, 0]
    for image in images:
        label = label_dir / (image.stem + '.txt')
        if not label.is_file():
            raise FileNotFoundError(f'Missing corresponding annotation: {label}')
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f'Expected 5-value detection label: {label}')
            cls = int(parts[0]); coords = list(map(float, parts[1:]))
            if str(cls) != parts[0] or cls not in range(3):
                raise ValueError(f'Invalid class ID in {label}: {parts[0]}')
            if not (0 <= coords[0] <= 1 and 0 <= coords[1] <= 1 and 0 < coords[2] <= 1 and 0 < coords[3] <= 1):
                raise ValueError(f'Invalid normalized bounding box in {label}')
            counts[cls] += 1
    return {'images': len(images), 'object_counts': dict(zip(NAMES, counts))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='data.yaml for separate Cambodian train/val images')
    parser.add_argument('--weights', required=True, help='INDIAN baseline_s42/weights/best.pt')
    parser.add_argument('--output', default='runs/cambodian_finetune_v1')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    if args.epochs < 11:
        parser.error('Use at least 11 epochs when close_mosaic=10.')
    check_version()
    output = Path(args.output).expanduser().resolve()
    save_dir = output / f'baseline_adapted_s{args.seed}'
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}. Choose a fresh --output path.')
    weights = Path(args.weights).expanduser().resolve(strict=True)
    if weights.suffix.lower() != '.pt':
        parser.error('--weights must point to the Indian-trained baseline best.pt')

    # Resolve only train+val and strictly verify their directories. No test set here.
    data_yaml, data = resolve_data(args.data, output, required_splits=('train', 'val'))
    if 'test' in data:
        raise ValueError('Adaptation data.yaml must not include a test path. Keep the 600-image test set separate.')
    train_dir, val_dir = Path(data['train']), Path(data['val'])
    if train_dir.resolve() == val_dir.resolve():
        raise ValueError('Fine-tuning train and validation directories must differ.')
    train_stats = count_objects(train_dir)
    val_stats = count_objects(val_dir)
    if train_stats['images'] != 160 or val_stats['images'] != 40:
        raise ValueError(f'Expected the documented 160/40 split; found {train_stats["images"]}/{val_stats["images"]}.')

    print('Dataset:', data_yaml)
    print('Train:', train_stats)
    print('Local validation:', val_stats)
    print('Indian source checkpoint:', weights)

    model = YOLO(str(weights))
    check_names(model.names)
    if len(model.names) != 3 or model.model.yaml.get('scale') != 's' or 'C3k2' not in str(model.model):
        raise ValueError('Use an Indian-trained, THREE-CLASS YOLO11s best.pt; not the 80-class COCO checkpoint.')

    selected_device = device(args.device)
    augmentation = dict(AUGMENTATION)
    augmentation.update(mosaic=0.20, scale=0.30, translate=0.08,
                        close_mosaic=10, degrees=0.0, shear=0.0,
                        perspective=0.0, flipud=0.0, mixup=0.0,
                        cutmix=0.0, copy_paste=0.0)

    # Keep the original three-view, crop-guarded dataset pipeline. Standard BCE:
    # do not apply the weighted model's non-unit class weights here.
    model.add_callback('on_pretrain_routine_end', partial(set_positive_weight, weight=1.0))
    model.add_callback('on_train_epoch_end', log_crop_stats)

    record = {
        'experiment': 'Cambodian supervised adaptation from Indian-trained baseline',
        'predefined_checkpoint_source': str(weights),
        'source_weights_sha256': sha256(weights),
        'data_yaml': str(data_yaml),
        'train': train_stats, 'val': val_stats,
        'note': 'The 600-image Cambodian test set is NOT used for training, validation, or epoch selection.',
        'selected_checkpoint_rule': 'Ultralytics best.pt from local Cambodian validation fitness/mAP50-95',
        'config': vars(args), 'optimizer': 'AdamW', 'lr0': 0.0001,
        'lrf': 0.10, 'cos_lr': True, 'warmup_epochs': 2.0, 'warmup_bias_lr': 0.0001,
        'weight_decay': 0.0005, 'three_view_training': '1 clean + 2 independently augmented visits per original per epoch',
        'class_positive_weights': [1.0,1.0,1.0],
        'augmentation': augmentation, 'pixel_augmentation': PIXEL_AUGMENTATION,
        'crop_policy': CROP_POLICY,
        'ultralytics': ultralytics.__version__, 'torch': torch.__version__,
    }
    (output / 'adaptation_protocol.json').write_text(json.dumps(record, indent=2))

    print('Training with explicit AdamW lr0=0.0001, 40-epoch default, local 40-image validation split.')
    model.train(
        trainer=ThreeViewTrainer, data=str(data_yaml),
        project=str(output), name=save_dir.name,
        epochs=args.epochs, patience=args.epochs,
        imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, device=selected_device,
        seed=args.seed, deterministic=True,
        optimizer='AdamW', lr0=0.0001, lrf=0.10,
        cos_lr=True, warmup_epochs=2.0, warmup_bias_lr=0.0001,
        weight_decay=0.0005, freeze=0,
        amp=selected_device!='cpu', cache=False, rect=False,
        resume=False, val=True, plots=True,
        **augmentation,
    )
    print('Selected checkpoint (validation):', save_dir / 'weights' / 'best.pt')
    print('Latest checkpoint:', save_dir / 'weights' / 'last.pt')
    print('Protocol log:', output / 'adaptation_protocol.json')

if __name__=='__main__':
    main()