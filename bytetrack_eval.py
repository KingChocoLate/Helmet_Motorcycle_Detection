"""Offline reproducible YOLO11s + Ultralytics ByteTrack MOT evaluation (BikeWithRider only).

BikeWithRider-only evaluation on 225 frames with persistent ground-truth IDs.
Default mode expects the already sampled, 225-frame (~10 FPS) annotation-aligned video.
Use --stride 6 ONLY when passing the original ~60 FPS source video containing 1346 frames.
Annotation format: class cx cy w h track_id (normalized), class 0 BikeWithRider.

Important: The audit validates file structure and creates ground-truth video overlays.
It does not prove visual correspondence; inspect the audit previews before scoring.

Requires: ultralytics, motmetrics, opencv-python, torch, numpy, pandas, scipy.
"""
import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = ROOT / 'clip10.mp4'
DEFAULT_LABEL_DIR = ROOT / 'bytetrack_eval/labels/train'
CLASS_NAMES = {0: 'BikeWithRider', 1: 'Helmet', 2: 'NoHelmet'}


def sha256(path):
    d = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            d.update(b)
    return d.hexdigest()


def load_gt(labels_dir, nframes=225):
    """Returns array of per-frame (track_id, x, y, w, h) normalized xywh, class 0 only."""
    data = []
    class_counts = Counter()
    id_frames = defaultdict(list)
    warnings = []
    for f in range(nframes):
        path = labels_dir / f'frame_{f:06d}.txt'
        if not path.exists():
            raise FileNotFoundError(f'Missing annotation for frame {f}: {path}')
        per_frame = []
        ids_in_frame = set()
        for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            p = line.split()
            if len(p) != 6:
                raise ValueError(f'{path}:{lineno} expected 6 columns (class cx cy w h track_id), got {p}')
            cls = int(p[0]); cx, cy, w, h = [float(x) for x in p[1:5]]; tid = int(p[5])
            if cls not in CLASS_NAMES or any(not math.isfinite(q) for q in (cx,cy,w,h)):
                raise ValueError(f'Invalid class or nonfinite coordinate in {path}:{lineno}')
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and w > 0 and h > 0 and
                    cx - w/2 >= -1e-3 and cy-h/2 >= -1e-3 and cx+w/2 <= 1+1e-3 and cy+h/2 <= 1+1e-3):
                raise ValueError(f'Invalid normalized bounding box in {path}:{lineno}: {p}')
            class_counts[cls] += 1
            if cls != 0:
                continue
            if tid in ids_in_frame:
                raise ValueError(f'Duplicate BikeWithRider ID={tid} in {path}')
            ids_in_frame.add(tid)
            per_frame.append((tid, cx, cy, w, h))
            id_frames[tid].append(f)
        data.append(per_frame)
    for tid, frames in sorted(id_frames.items()):
        gaps = [(a, b, b-a) for a,b in zip(frames,frames[1:]) if b-a>1]
        if gaps:
            maximum=max(gaps,key=lambda x:x[2])
            if maximum[2]>10:
                warnings.append({'id':tid,'longest_gap_frames':maximum[2]-1,'from':maximum[0],'to':maximum[1],
                                 'action':'Review whether same physical rider reappears or ID was reused'})
    return data, class_counts, id_frames, warnings


def xywh_pixels(normalized, image_w, image_h):
    t,cx,cy,w,h=normalized
    return [((cx-w/2)*image_w), ((cy-h/2)*image_h), (w*image_w), (h*image_h)]


def compute_dataset_audit(gt, counts, ids, warnings):
    return {
        'total_annotated_frames':len(gt),
        'BikeWithRider_instances':int(counts.get(0,0)),
        'BikeWithRider_track_ids':len(ids),
        'Helmet_instances_partial_NOT_scored':int(counts.get(1,0)),
        'NoHelmet_instances_NOT_scored':int(counts.get(2,0)),
        'BikeWithRider_unique_counts_per_frame':{'min':min(map(len,gt)),'max':max(map(len,gt)), 'avg':round(sum(map(len,gt))/len(gt),2)},
        'ground_truth_review_warnings':warnings,
    }


def draw_prediction_preview(image, tracked, idx, input_frame_idx, output_path):
    img=image.copy()
    for tid, bbox, score in tracked:
        x,y,w,h=bbox
        x1,y1,x2,y2=int(x),int(y),int(x+w),int(y+h)
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,180,0),2)
        cv2.putText(img,f'ID {tid}  {score:.2f}',(max(0,x1),max(18,y1-4)), cv2.FONT_HERSHEY_SIMPLEX,.55,(255,180,0),2,cv2.LINE_AA)
    cv2.putText(img,f'BikeWithRider | annotated frame {idx} | input video frame {input_frame_idx}',(24,40),
                cv2.FONT_HERSHEY_SIMPLEX, .88,(10,10,10),3,cv2.LINE_AA)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(output_path),img)



def save_alignment_previews(video_path, gt, width, height, stride, output_dir):
    """Draw ACTUAL ground-truth boxes on selected input frames for human alignment checks.

    Labels alone cannot establish that frame N in a re-encoded clip matches annotation N.
    We decode in order, as in the tracker loop, rather than relying on keyframe seeking.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = {0, 30, 90, 150, len(gt) - 1}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot decode video for alignment previews: {video_path}')
    sample_i = 0
    input_i = 0
    saved = []
    try:
        while sample_i < len(gt):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Video ended before alignment check reached annotated frame {sample_i}')
            if input_i % stride:
                input_i += 1
                continue
            if sample_i in samples:
                canvas = frame.copy()
                for tid, cx, cy, bw, bh in gt[sample_i]:
                    x, y, w, h = xywh_pixels((tid,cx,cy,bw,bh), width, height)
                    left, top = int(round(x)), int(round(y))
                    right, bottom = int(round(x + w)), int(round(y + h))
                    cv2.rectangle(canvas, (left, top), (right, bottom), (0, 225, 80), 2)
                    cv2.putText(canvas, f'GT ID {tid}', (max(0, left), max(18, top-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,225,80), 2, cv2.LINE_AA)
                cv2.putText(canvas, f'Ground truth | frame {sample_i} | input frame {input_i}',
                            (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10,10,10), 3, cv2.LINE_AA)
                path = output_dir / f'audit_gt_frame_{sample_i:03d}.jpg'
                if not cv2.imwrite(str(path), canvas):
                    raise IOError(f'Failed to save GT overlay: {path}')
                saved.append(str(path))
            input_i += 1
            sample_i += 1
    finally:
        cap.release()
    return saved


def report_text(summary):
    m=summary['metrics']; a=summary['annotation_audit']; p=summary['protocol']
    def fmt(k,d=3):
        v=m.get(k)
        if v is None:return '[not available]'
        return f'{v:.{d}f}' if isinstance(v,(int,float)) else str(v)
    sampling_explanation = (
        f'All {a["total_annotated_frames"]} consecutive frames of the already-sampled '
        f'{p["input_video_fps"]:.2f} FPS clip were processed without further skipping.'
        if p['input_video_frame_stride'] == 1 else
        f'Every {p["input_video_frame_stride"]}th frame was selected from the original '
        f'{p["input_video_fps"]:.2f} FPS video, yielding '
        f'{p["effective_sample_fps"]:.2f} annotated frames per second.'
    )
    return f'''# Report insertion draft — verified BikeWithRider MOT evaluation

The numerical values below were calculated by the accompanying script after actually
running the supplied YOLO11s checkpoint with Ultralytics ByteTrack. Do not reuse
these values for a different model, video or threshold configuration.

## Add to Chapter 3 after application architecture (new subsection: Tracking evaluation protocol)

A separate video-based evaluation was conducted for the BikeWithRider class using
{a['total_annotated_frames']} annotated frames from a Cambodian traffic video.
{sampling_explanation} Ground-truth
BikeWithRider annotations included {a['BikeWithRider_instances']} bounding boxes associated
with {a['BikeWithRider_track_ids']} distinct annotated identities. Ground-truth labels used
normalized YOLO coordinates and a persistent track identifier; they were converted to
pixel-space boxes for multi-object tracking evaluation. The {a['Helmet_instances_partial_NOT_scored']}
partially annotated Helmet instances were excluded, and NoHelmet was not scored.

The selected detector checkpoint was evaluated through the Ultralytics YOLO.track() API
with tracker=\"bytetrack.yaml\", persist=True, class filtering for BikeWithRider (class 0),
input size {p['imgsz']}, and a confidence threshold of {p['conf']}. Identity estimates were
reset by starting a new model/tracker evaluation for the sequence. Predictions and ground
truth were matched per frame using a bounding-box IoU threshold of {p['match_iou']}.
Tracking was evaluated using the MOTMetrics library, reporting identity F1 score (IDF1),
multiple object tracking accuracy (MOTA), detection precision/recall, false positives,
misses and identity switches (IDSW). Tracking metrics apply only to BikeWithRider; this
evaluation does not assess Helmet/NoHelmet tracking, helmet-to-rider association, or
violation-event counting.

## Add to Chapter 4 (new subsection: BikeWithRider ByteTrack tracking results)

The BikeWithRider-only tracking evaluation comprised {a['total_annotated_frames']} annotated
frames with {a['BikeWithRider_instances']} ground-truth object instances and
{a['BikeWithRider_track_ids']} unique identities. The evaluated checkpoint
({p['checkpoint_filename']}) achieved IDF1={fmt('idf1')}, MOTA={fmt('mota')},
ID switches={fmt('num_switches',0)}, detection precision={fmt('precision')},
and recall={fmt('recall')} at the selected matching threshold. The evaluation recorded
{fmt('num_false_positives',0)} false positives and {fmt('num_misses',0)} missed ground-truth
instances. These measurements describe BikeWithRider identity tracking in this sampled
video only; they do not establish helmet-status tracking or fully validated compliance
monitoring.

The tracking evaluation processed frames at an effective video sampling rate of
{p['effective_sample_fps']:.2f} frames/s. The measured detector-and-tracker
call time was {p['model_tracking_seconds']:.2f} seconds for {a['total_annotated_frames']}
processed frames (approximately {p['model_tracking_calls_per_second']:.2f} calls per second
on {p['device']}); this is not end-to-end web-application FPS because it excludes some
I/O, browser rendering and network overhead.

## Add to Chapter 5 (tracking conclusion and limitations)

The ByteTrack component was quantitatively evaluated for BikeWithRider identity tracking
on one annotated Cambodian traffic sequence. This addresses the rider-level tracking part
of the prototype evaluation under the tested conditions. The Helmet annotations were
incomplete and the NoHelmet class lacked ground-truth tracking IDs, so neither helmet-status
tracking nor association of heads with motorcycles was quantitatively assessed. Further
work is required to collect diverse, independently annotated sequences and evaluate
head-to-rider association, NoHelmet event reliability, and end-to-end application throughput.
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,default=DEFAULT_VIDEO)
    parser.add_argument('--labels',type=Path,default=DEFAULT_LABEL_DIR)
    parser.add_argument('--model',type=Path,default=ROOT/'models/adapted_v1.pt')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/adapted_v1')
    parser.add_argument('--imgsz',type=int,default=960)
    parser.add_argument('--conf',type=float,default=.10)
    parser.add_argument('--nms-iou',type=float,default=.70)
    parser.add_argument('--match-iou',type=float,default=.50)
    parser.add_argument('--stride',type=int,default=1,help='1: all frames of 225-frame annotated clip (default). 6: every 6th original ~60 FPS frame')
    parser.add_argument('--num-frames',type=int,default=225)
    parser.add_argument('--device',default='auto',help='auto, cpu, or 0 for first CUDA GPU')
    parser.add_argument('--audit-only',action='store_true',help='Validate labels and video alignment, without inference')
    args=parser.parse_args()
    args.video=args.video.resolve();args.labels=args.labels.resolve();args.model=args.model.resolve();args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=True)
    if not args.video.exists(): raise FileNotFoundError(f'Missing source video: {args.video}')
    gt, counts, ids, warnings=load_gt(args.labels,args.num_frames)
    audit=compute_dataset_audit(gt,counts,ids,warnings)
    cap=cv2.VideoCapture(str(args.video))
    if not cap.isOpened():raise RuntimeError('Could not decode source video')
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=float(cap.get(cv2.CAP_PROP_FPS));original_frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if args.stride < 1 or args.num_frames < 1:
        raise ValueError('--stride and --num-frames must be positive integers')
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f'Invalid video metadata: FPS={fps}, size={width}x{height}')
    need = (args.num_frames - 1) * args.stride + 1
    if original_frame_count > 0 and original_frame_count < need:
        raise ValueError(f'Video contains {original_frame_count} frames, but stride={args.stride} '
                         f'needs at least {need}. For an already-sampled 225-frame clip, '
                         f'use --stride 1 (or omit --stride).')
    if original_frame_count > 0:
        available_samples = (original_frame_count + args.stride - 1) // args.stride
        if available_samples != args.num_frames:
            raise ValueError(f'Video has {original_frame_count} input frames at stride={args.stride}, '
                             f'giving {available_samples} sampled frames, but labels specify '
                             f'{args.num_frames}. Use --stride 1 for 225-frame clips or '
                             f'--stride 6 for a 1,346-frame original video. '
                             f'Never re-sample a clip that is already sampled.')
    audit['input_video_frames']=original_frame_count
    audit['input_video_fps']=fps
    audit['input_video_size']=[width,height]
    audit['input_video_frame_stride']=args.stride
    audit['effective_sample_fps']=fps/args.stride
    print(json.dumps(audit,indent=2))
    (args.out/'annotation_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    if args.audit_only:
        previews = save_alignment_previews(args.video, gt, width, height, args.stride, args.out)
        print('Annotation audit passed; model and tracker were NOT run.')
        print('IMPORTANT: Visually inspect these GT overlays to confirm frame alignment:')
        for item in previews:
            print('  ', item)
        return
    print('IMPORTANT: Compare audit_gt_frame_*.jpg with their labels before trusting MOT scores.')
    if not args.model.exists():raise FileNotFoundError(f'Checkpoint not found: {args.model}')
    try:
        import motmetrics as mm
        import torch
        from ultralytics import YOLO
        import ultralytics
    except ImportError as e:
        raise RuntimeError('Install dependencies: py -m pip install ultralytics motmetrics pandas scipy') from e
    device=('0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    model=YOLO(str(args.model))
    name=str(model.names.get(0)) if isinstance(model.names,dict) else str(model.names[0])
    if name.lower().replace('_','')!='bikewithrider':
        raise ValueError(f'Model class 0 is {name!r}, expected BikeWithRider; stopping to prevent invalid evaluation')
    acc=mm.MOTAccumulator(auto_id=False)
    pr_rows=[];gt_rows=[];per_frame=[]
    cap=cv2.VideoCapture(str(args.video))
    sample_index=0; raw_index=0; total_track_time=0.0
    previews={15,60,120,180,220}
    while sample_index < args.num_frames:
        ok,frame=cap.read()
        if not ok: raise RuntimeError(f'Failed to decode original video at raw frame {raw_index}')
        if raw_index % args.stride != 0:
            raw_index+=1
            continue
        if str(device) != 'cpu' and torch.cuda.is_available():
            torch.cuda.synchronize()
        tic=time.perf_counter()
        result=model.track(frame,persist=True,tracker='bytetrack.yaml',classes=[0],
                           imgsz=args.imgsz,conf=args.conf,iou=args.nms_iou,
                           device=device,verbose=False)[0]
        total_track_time += time.perf_counter()-tic
        boxes=result.boxes
        tracked=[]
        if boxes is not None and boxes.id is not None:
            xyxy=boxes.xyxy.detach().cpu().numpy()
            trackids=boxes.id.detach().cpu().numpy().astype(int)
            confs=boxes.conf.detach().cpu().numpy()
            clses=boxes.cls.detach().cpu().numpy().astype(int)
            for (x1,y1,x2,y2),tid,score,cls in zip(xyxy,trackids,confs,clses):
                if cls!=0: continue
                bbox=[max(0,float(x1)),max(0,float(y1)),max(0,float(x2-x1)),max(0,float(y2-y1))]
                tracked.append((int(tid),bbox,float(score)))
        if len({t[0] for t in tracked}) != len(tracked):
            raise ValueError(f'Duplicate predicted tracker IDs in sampled frame {sample_index}')
        labels=gt[sample_index]
        gt_ids=[int(x[0])+1 for x in labels] # Convert 0-based GT IDs to 1-based MOT IDs
        gt_xywh=[xywh_pixels(x,width,height) for x in labels]
        pr_ids=[x[0] for x in tracked]
        pr_xywh=[x[1] for x in tracked]
        dists=mm.distances.iou_matrix(np.asarray(gt_xywh).reshape((-1,4)),
                                       np.asarray(pr_xywh).reshape((-1,4)),
                                       max_iou=1-args.match_iou)
        acc.update(gt_ids,pr_ids,dists,frameid=sample_index+1)
        for tid,xywh in zip(gt_ids,gt_xywh):
            gt_rows.append([sample_index+1,tid,*[round(x,4) for x in xywh],1,1,1])
        for tid,bbox,score in tracked:
            pr_rows.append([sample_index+1,tid,*[round(x,4) for x in bbox],round(score,6),1,-1])
        per_frame.append([sample_index,raw_index,len(labels),len(tracked)])
        if sample_index in previews:
            draw_prediction_preview(frame,tracked,sample_index,raw_index,args.out/f'preview_{sample_index:03d}.jpg')
        sample_index+=1;raw_index+=1
        if sample_index%25==0: print(f'Processed {sample_index}/{args.num_frames} sampled frames',flush=True)
    cap.release()
    mh=mm.metrics.create()
    metric_names=['num_frames','num_objects','num_unique_objects','num_predictions',
                  'num_matches','num_misses','num_false_positives','num_switches',
                  'num_fragmentations','precision','recall','mota','motp','idf1','idp','idr',
                  'mostly_tracked','partially_tracked','mostly_lost']
    summary_df=mh.compute(acc,metrics=metric_names,name='BikeWithRider')
    values={k:(None if not np.isfinite(float(summary_df.iloc[0][k])) else float(summary_df.iloc[0][k]))
            for k in metric_names}
    if values['motp'] is not None: values['mean_matched_iou']=1-values['motp']
    protocol={
      'checkpoint_filename':args.model.name,'checkpoint_sha256':sha256(args.model),
      'source_video_filename':args.video.name,'source_video_sha256':sha256(args.video),
      'source_video_frame_stride':args.stride,'effective_sample_fps':fps/args.stride,
      'imgsz':args.imgsz,'conf':args.conf,'nms_iou':args.nms_iou,
      'match_iou':args.match_iou,'device':device,
      'model_tracking_seconds':round(total_track_time,3),
      'model_tracking_calls_per_second':round(args.num_frames/total_track_time,3),
      'ultralytics_version':ultralytics.__version__,'torch_version':torch.__version__,
      'motmetrics_version':getattr(mm,'__version__','unknown'),'opencv_version':cv2.__version__,
      'metric_scope':f'BikeWithRider only (class 0) on {args.num_frames} annotated frames',
    }
    report={'annotation_audit':audit,'protocol':protocol,'metrics':values}
    (args.out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with open(args.out/'metrics.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['metric','value']);writer.writerows(values.items())
    for file,rows in [('gt_mot.txt',gt_rows),('pred_mot.txt',pr_rows)]:
        with open(args.out/file,'w',newline='',encoding='utf-8') as f:csv.writer(f).writerows(rows)
    with open(args.out/'per_frame_counts.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['sampled_frame_0_based','source_frame_0_based','gt_boxes','tracked_boxes']);writer.writerows(per_frame)
    (args.out/'REPORT_INSERTION_FILLED.md').write_text(report_text(report),encoding='utf-8')
    print('\n=== BikeWithRider tracking results ===')
    for k in ['idf1','mota','num_switches','precision','recall','num_false_positives','num_misses','mean_matched_iou']:
        print(f'{k:24}: {values.get(k)}')
    print('\nSaved results:',args.out)
    print('IMPORTANT: Helmet and NoHelmet were deliberately not scored.')


if __name__=='__main__':main()
