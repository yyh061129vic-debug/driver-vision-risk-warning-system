from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MMSEG_ROOT = ROOT / 'mmsegmentation-main'
DEFAULT_VIDEO = ROOT / 'test1.mp4'
DEFAULT_SEG_CONFIG = (
    MMSEG_ROOT / 'configs' / 'segformer' /
    'segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py'
)
DEFAULT_SEG_CHECKPOINT = (
    MMSEG_ROOT / 'checkpoints' /
    'segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'
)
DEFAULT_OUTPUT_DIR = ROOT / 'outputs' / 'fusion_video_demo'

ROAD_CLASS_ID = 0
YOLO_COCO_KEEP = {
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'bench'
}


def ensure_imports() -> None:
    mmseg_path = str(MMSEG_ROOT)
    if mmseg_path not in sys.path:
        sys.path.insert(0, mmseg_path)
    env_bin = Path(sys.prefix) / 'bin'
    if env_bin.exists():
        os.environ['PATH'] = str(env_bin) + os.pathsep + os.environ.get('PATH', '')


def clamp_box(xyxy: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(round, xyxy)
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2


def risk_from_box(
    xyxy: list[float],
    road_mask: np.ndarray,
    image_shape: tuple[int, int, int],
) -> tuple[str, float]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = clamp_box(xyxy, width, height)
    if x2 <= x1 or y2 <= y1:
        return 'safe', 0.0

    roi = road_mask[y1:y2, x1:x2]
    road_overlap = float(roi.sum()) / max(float(roi.size), 1.0)
    box_height_ratio = (y2 - y1) / height
    box_bottom_ratio = y2 / height

    if road_overlap >= 0.50 and box_bottom_ratio > 0.75 and box_height_ratio > 0.18:
        return 'danger', road_overlap
    if road_overlap >= 0.30 and box_bottom_ratio > 0.55:
        return 'warning', road_overlap
    if road_overlap >= 0.10:
        return 'notice', road_overlap
    return 'safe', road_overlap


def draw_overlay(image: np.ndarray, road_mask: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    overlay = image.copy()
    road_color = np.array([80, 210, 80], dtype=np.uint8)
    overlay[road_mask] = (
        overlay[road_mask].astype(np.float32) * (1.0 - alpha) +
        road_color.astype(np.float32) * alpha
    ).astype(np.uint8)
    return overlay


def draw_detection(
    image: np.ndarray,
    xyxy: list[float],
    label: str,
    confidence: float,
    risk: str,
    road_overlap: float,
) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clamp_box(xyxy, width, height)
    colors = {
        'safe': (120, 120, 120),
        'notice': (0, 200, 255),
        'warning': (0, 140, 255),
        'danger': (0, 0, 255),
    }
    color = colors.get(risk, (255, 255, 255))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = f'{label} {confidence:.2f} {risk} road={road_overlap:.2f}'
    cv2.putText(
        image,
        text,
        (x1, max(18, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def frame_risk(objects: list[dict]) -> str:
    order = {'safe': 0, 'notice': 1, 'warning': 2, 'danger': 3}
    best = 'safe'
    for obj in objects:
        if order[obj['risk_level']] > order[best]:
            best = obj['risk_level']
    return best


def process_frame(
    frame: np.ndarray,
    frame_id: int,
    seg_model,
    yolo_model,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    from mmseg.apis import inference_model

    seg_result = inference_model(seg_model, frame)
    seg_mask = seg_result.pred_sem_seg.data[0].cpu().numpy()
    road_mask = seg_mask == args.road_class_id

    yolo_result = yolo_model.predict(frame, conf=args.conf, verbose=False)[0]
    annotated = draw_overlay(frame, road_mask)

    objects = []
    for box in yolo_result.boxes:
        cls_id = int(box.cls[0])
        class_name = yolo_model.names[cls_id]
        if args.coco_filter and class_name not in YOLO_COCO_KEEP:
            continue

        confidence = float(box.conf[0])
        xyxy = [float(v) for v in box.xyxy[0].cpu().numpy().tolist()]
        risk, road_overlap = risk_from_box(xyxy, road_mask, frame.shape)
        draw_detection(annotated, xyxy, class_name, confidence, risk, road_overlap)
        objects.append({
            'class_id': cls_id,
            'class_name': class_name,
            'confidence': round(confidence, 4),
            'bbox_xyxy': [round(v, 2) for v in xyxy],
            'road_overlap': round(road_overlap, 4),
            'risk_level': risk,
        })

    record = {
        'frame_id': frame_id,
        'frame_risk': frame_risk(objects),
        'objects': objects,
    }
    cv2.putText(
        annotated,
        f'frame {frame_id} risk: {record["frame_risk"]}',
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    return annotated, record


def run(args: argparse.Namespace) -> None:
    ensure_imports()
    from mmseg.apis import init_model
    from ultralytics import YOLO

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / 'result.mp4'
    output_log = output_dir / 'risk_log.jsonl'

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {args.video}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = args.max_frames if args.max_frames > 0 else total

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f'Could not create output video: {output_video}')

    seg_model = init_model(args.seg_config, args.seg_checkpoint, device=args.device)
    yolo_model = YOLO(args.yolo_model)

    print(f'Input video: {args.video}')
    print(f'Output video: {output_video}')
    print(f'Output log:   {output_log}')
    print(f'Processing frames: {limit if limit else total}')

    frame_id = 0
    with output_log.open('w', encoding='utf-8') as log_file:
        while frame_id < limit:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, record = process_frame(frame, frame_id, seg_model, yolo_model, args)
            writer.write(annotated)
            log_file.write(json.dumps(record, ensure_ascii=False) + '\n')
            frame_id += 1
            if frame_id % 10 == 0:
                print(f'processed {frame_id}/{limit}')

    cap.release()
    writer.release()
    print(f'Done. Frames written: {frame_id}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fuse YOLO detections with a SegFormer road mask on a video.')
    parser.add_argument('--video', default=str(DEFAULT_VIDEO), help='Input video path.')
    parser.add_argument('--yolo-model', default='yolo11n.pt', help='YOLO model path/name.')
    parser.add_argument('--seg-config', default=str(DEFAULT_SEG_CONFIG), help='MMSeg config path.')
    parser.add_argument('--seg-checkpoint', default=str(DEFAULT_SEG_CHECKPOINT), help='SegFormer checkpoint path.')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR), help='Output directory.')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 or cpu.')
    parser.add_argument('--road-class-id', type=int, default=ROAD_CLASS_ID, help='Road class id in segmentation mask.')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO confidence threshold.')
    parser.add_argument('--max-frames', type=int, default=100, help='Maximum frames to process; <=0 means all frames.')
    parser.add_argument('--no-coco-filter', dest='coco_filter', action='store_false',
                        help='Keep all COCO detections instead of road-relevant classes.')
    parser.set_defaults(coco_filter=True)
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
