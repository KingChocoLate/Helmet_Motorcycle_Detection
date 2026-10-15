"""Generate a completed BikeWithRider tracking report from already saved metrics.

No inference, no video decoding, and no tracker execution. Uses the existing
bytetrack_eval.py report_text() function, with a backward-compatible correction
for the two protocol fields missing from older versions of the script.

Usage:
    python3 finish_bytetrack_report.py --metrics outputs/mot_v1/metrics.json
"""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, default=Path('outputs/mot_v1/metrics.json'))
    parser.add_argument('--evaluator', type=Path, default=Path('bytetrack_eval.py'))
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    metrics_path = args.metrics.expanduser().resolve()
    evaluator_path = args.evaluator.expanduser().resolve()
    if not metrics_path.is_file():
        raise SystemExit(f'No saved metrics file: {metrics_path}\nDo not rerun inference yet; check the --out path used in the failed evaluation.')
    if not evaluator_path.is_file():
        raise SystemExit(f'Missing evaluator script: {evaluator_path}')

    result = json.loads(metrics_path.read_text(encoding='utf-8'))
    if not all(k in result for k in ('annotation_audit', 'protocol', 'metrics')):
        raise SystemExit('metrics.json is missing a required section; cannot create a verified report')
    audit, protocol, metrics = result['annotation_audit'], result['protocol'], result['metrics']
    if 'idf1' not in metrics or 'mota' not in metrics:
        raise SystemExit('metrics.json lacks calculated MOT metrics. Please share the file for diagnosis.')

    # Older evaluator versions saved the values in annotation_audit but did not
    # copy them to 'protocol', which caused KeyError during report generation.
    protocol.setdefault('input_video_frame_stride',
                        audit.get('input_video_frame_stride', protocol.get('source_video_frame_stride')))
    protocol.setdefault('input_video_fps', audit.get('input_video_fps'))
    protocol.setdefault('effective_sample_fps', audit.get('effective_sample_fps'))
    required = ('input_video_frame_stride', 'input_video_fps', 'effective_sample_fps')
    missing = [k for k in required if protocol.get(k) is None]
    if missing:
        raise SystemExit(f'Cannot reconstruct original evaluation settings: {missing}')
    if protocol['input_video_frame_stride'] == 1 and audit.get('input_video_frames') != audit.get('total_annotated_frames'):
        raise SystemExit('Video/GT frame count mismatch in saved audit; inspect before reporting.')

    spec = importlib.util.spec_from_file_location('bytetrack_evaluator_saved', evaluator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'report_text'):
        raise SystemExit(f'No report_text() function in {evaluator_path}')

    report = module.report_text(result)
    report_path = (args.out or metrics_path.with_name('REPORT_INSERTION_FILLED.md')).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding='utf-8')

    print('SUCCESS: Used already-saved evaluation metrics; NO model rerun was performed.')
    print('Saved report:', report_path)
    print('\nBikeWithRider MOT metrics (unreviewed):')
    for name in ('idf1','mota','num_switches','precision','recall','num_false_positives','num_misses','mean_matched_iou'):
        print(f'  {name:24} {metrics.get(name)}')
    print('\nCheck earlier ground-truth ID-gap warnings and image alignment before submitting.')


if __name__ == '__main__':
    main()
