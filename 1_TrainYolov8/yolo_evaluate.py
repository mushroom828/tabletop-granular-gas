"""Evaluate matched YOLO boxes on the validation split used for localization errors.

Pass --weights for a numbered training run such as runs/train-2/weights/best.pt.
The optional --split test evaluates the reserved test images instead. Center
and size errors below describe matched boxes; missed detections are not assigned
a pixel error. Matching remains one-to-one and restricted to the same class.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--weights", type=Path, default=BASE_DIR / "runs/train/weights/best.pt",
                    help="Trained weights; set this to the actual training run's best.pt")
parser.add_argument("--split", choices=("val", "test"), default="val",
                    help="Evaluation split (default: val, as used for localization errors)")
args = parser.parse_args()
MODEL_PATH = args.weights.expanduser().resolve()
IMG_DIR = BASE_DIR / "dataset" / args.split / "images"
LABEL_DIR = BASE_DIR / "dataset" / args.split / "labels"

# A same-class pair must have IoU strictly greater than this threshold.
IOU_THRESH = 0.5

# Inference settings match velocity extraction; IOU_THRESH above is GT matching.
DETECTION_CONFIDENCE = 0.25
DETECTION_NMS_IOU = 0.2

# Supported image extensions; each image requires a matching .txt label.
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']


def match_boxes_by_class(iou_values, gt_classes, pred_classes, threshold):
    """Return a maximum-cardinality, one-to-one matching above the IoU threshold."""
    candidate_preds = []
    for gt_idx, gt_cls in enumerate(gt_classes):
        indices = np.flatnonzero((pred_classes == gt_cls) & (iou_values[gt_idx] > threshold))
        candidate_preds.append(sorted(indices.tolist(), key=lambda idx: iou_values[gt_idx, idx], reverse=True))

    matched_pred_to_gt = {}

    def match_gt(gt_idx, visited_preds):
        for pred_idx in candidate_preds[gt_idx]:
            if pred_idx in visited_preds:
                continue
            visited_preds.add(pred_idx)
            previous_gt = matched_pred_to_gt.get(pred_idx)
            if previous_gt is None or match_gt(previous_gt, visited_preds):
                matched_pred_to_gt[pred_idx] = gt_idx
                return True
        return False

    for gt_idx in sorted(range(len(gt_classes)), key=lambda idx: len(candidate_preds[idx])):
        match_gt(gt_idx, set())

    return [(gt_idx, pred_idx) for pred_idx, gt_idx in matched_pred_to_gt.items()]


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file does not exist: {MODEL_PATH}")
if not os.path.isdir(IMG_DIR):
    raise FileNotFoundError(f"Image folder does not exist: {IMG_DIR}")
if not os.path.isdir(LABEL_DIR):
    raise FileNotFoundError(f"Label folder does not exist: {LABEL_DIR}")

# Load model
print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Collect all image files
img_files = sorted(f for f in os.listdir(IMG_DIR)
                   if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS)
print(f"Found {len(img_files)} {args.split} images")

# Initialize error lists (unit: pixels)
center_dx_list = []       # Center x offset (pixels)
center_dy_list = []       # Center y offset (pixels)
width_error_list = []     # Width absolute error (pixels)
height_error_list = []    # Height absolute error (pixels)
iou_list = []             # Matched IoU
cls_matched = []          # Class of matched boxes

# Store errors by class (pixels)
per_class_errors = {
    0: {'dx': [], 'dy': [], 'dw': [], 'dh': [], 'iou': []},
    1: {'dx': [], 'dy': [], 'dw': [], 'dh': [], 'iou': []}
}

total_gt = 0
matched_gt = 0
total_pred = 0

# Process each image
for img_name in img_files:
    img_path = os.path.join(IMG_DIR, img_name)
    base = os.path.splitext(img_name)[0]
    label_path = os.path.join(LABEL_DIR, base + ".txt")

    # Read ground truth boxes
    if not os.path.exists(label_path):
        print(f"Warning: Label file missing: {label_path}, skipping this image")
        continue

    gt_boxes_norm = []   # Store normalized cx,cy,w,h
    gt_cls_list = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id, cx, cy, w, h = map(float, parts)
            gt_cls_list.append(int(cls_id))
            gt_boxes_norm.append([cx, cy, w, h])

    if len(gt_boxes_norm) == 0:
        continue

    # Model prediction
    results = model(img_path, conf=DETECTION_CONFIDENCE,
                    iou=DETECTION_NMS_IOU, verbose=False)[0]
    pred_boxes_xyxy = results.boxes.xyxy.cpu().numpy()      # Absolute pixel coordinates xyxy
    pred_cls = results.boxes.cls.cpu().numpy().astype(int)
    img_h, img_w = results.orig_shape

    # Convert ground truth boxes to pixel xyxy
    gt_boxes_pixel = []
    for (cx, cy, w, h) in gt_boxes_norm:
        x1 = (cx - w/2) * img_w
        y1 = (cy - h/2) * img_h
        x2 = (cx + w/2) * img_w
        y2 = (cy + h/2) * img_h
        gt_boxes_pixel.append([x1, y1, x2, y2])
    gt_boxes_pixel = np.array(gt_boxes_pixel)

    total_gt += len(gt_boxes_pixel)
    total_pred += len(pred_boxes_xyxy)

    if len(pred_boxes_xyxy) == 0:
        continue

    # Compute IoU matrix
    gt_tensor = torch.tensor(gt_boxes_pixel, dtype=torch.float32)
    pred_tensor = torch.tensor(pred_boxes_xyxy, dtype=torch.float32)
    iou_matrix = box_iou(gt_tensor, pred_tensor)  # shape: [M, N]

    # Find a one-to-one matching among same-class boxes above the IoU threshold.
    # Allow reassignment so an earlier ground truth does not block a later one
    # that could use that prediction while the earlier one has an alternative.
    for gt_idx, pred_idx in match_boxes_by_class(iou_matrix.numpy(), gt_cls_list, pred_cls, IOU_THRESH):
        gt_cls = gt_cls_list[gt_idx]
        max_iou = iou_matrix[gt_idx, pred_idx]
        matched_gt += 1

        gt_box = gt_boxes_pixel[gt_idx]
        pred_box = pred_boxes_xyxy[pred_idx]

        # Absolute pixel differences for center
        gt_cx = (gt_box[0] + gt_box[2]) / 2
        gt_cy = (gt_box[1] + gt_box[3]) / 2
        pred_cx = (pred_box[0] + pred_box[2]) / 2
        pred_cy = (pred_box[1] + pred_box[3]) / 2
        dx = abs(pred_cx - gt_cx)
        dy = abs(pred_cy - gt_cy)
        center_dx_list.append(dx)
        center_dy_list.append(dy)

        # Absolute pixel differences for width and height
        gt_w = gt_box[2] - gt_box[0]
        gt_h = gt_box[3] - gt_box[1]
        pred_w = pred_box[2] - pred_box[0]
        pred_h = pred_box[3] - pred_box[1]
        dw = abs(pred_w - gt_w)
        dh = abs(pred_h - gt_h)
        width_error_list.append(dw)
        height_error_list.append(dh)

        iou_list.append(max_iou.item())
        cls_matched.append(gt_cls)

        # Store errors by class
        per_class_errors[gt_cls]['dx'].append(dx)
        per_class_errors[gt_cls]['dy'].append(dy)
        per_class_errors[gt_cls]['dw'].append(dw)
        per_class_errors[gt_cls]['dh'].append(dh)
        per_class_errors[gt_cls]['iou'].append(max_iou.item())

# Output results
print("\n" + "=" * 55)
print("          Bounding Box Offset Error Analysis (matched by class)")
print("=" * 55)
print(f"Total ground truth boxes: {total_gt}")
print(f"Total predicted boxes: {total_pred}")
print(f"Successfully matched box pairs (same class IoU > {IOU_THRESH}): {matched_gt}")

if len(center_dx_list) == 0:
    print("\nWarning: No matched box pairs found. Please check if IOU_THRESH is too high or the model performance.")
else:
    center_dx = np.array(center_dx_list)
    center_dy = np.array(center_dy_list)
    width_err = np.array(width_error_list)
    height_err = np.array(height_error_list)
    ious = np.array(iou_list)

    print("\n[Overall metrics] (unit: pixels)")
    print(f"  Center X offset: mean={center_dx.mean():.2f}, median={np.median(center_dx):.2f}, std={center_dx.std():.2f}")
    print(f"  Center Y offset: mean={center_dy.mean():.2f}, median={np.median(center_dy):.2f}, std={center_dy.std():.2f}")
    print(f"  Width absolute error: mean={width_err.mean():.2f}, median={np.median(width_err):.2f}, std={width_err.std():.2f}")
    print(f"  Height absolute error: mean={height_err.mean():.2f}, median={np.median(height_err):.2f}, std={height_err.std():.2f}")
    print(f"  Matched IoU: mean={ious.mean():.4f}, median={np.median(ious):.4f}, min={ious.min():.4f}")

    # Output by class
    for cid, name in {0: 'ball', 1: 'container'}.items():
        err = per_class_errors[cid]
        n = len(err['dx'])
        if n == 0:
            print(f"\n[{name}] No matches")
            continue
        print(f"\n[{name}] Number of matches = {n}")
        print(f"  X offset mean = {np.mean(err['dx']):.2f} px, Y offset mean = {np.mean(err['dy']):.2f} px")
        print(f"  Width error mean = {np.mean(err['dw']):.2f} px, Height error mean = {np.mean(err['dh']):.2f} px")
        print(f"  IoU mean = {np.mean(err['iou']):.4f}")
