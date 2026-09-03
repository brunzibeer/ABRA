# file: bg_box_generator.py

import json
import random
import numpy as np

############################################ ANGEL MODE ############################################
# ------------------------
# Box utilities
# ------------------------
def iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0


def covers_too_much(new_box, gt_box, max_frac=0.1):
    """Check if new_box overlaps GT more than max_frac of GT area."""
    xA = max(new_box[0], gt_box[0])
    yA = max(new_box[1], gt_box[1])
    xB = min(new_box[0] + new_box[2], gt_box[0] + gt_box[2])
    yB = min(new_box[1] + new_box[3], gt_box[1] + gt_box[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    gtArea = gt_box[2] * gt_box[3]
    return (interArea / gtArea) > max_frac if gtArea > 0 else False


def contained_fraction_gt(new_box, gt_box):
    """Compute fraction of new_box area contained in GT box."""
    xA = max(new_box[0], gt_box[0])
    yA = max(new_box[1], gt_box[1])
    xB = min(new_box[0] + new_box[2], gt_box[0] + gt_box[2])
    yB = min(new_box[1] + new_box[3], gt_box[1] + gt_box[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    newArea = new_box[2] * new_box[3]
    return interArea / newArea if newArea > 0 else 0


# ------------------------
# Generate candidate background boxes
# ------------------------
def generate_candidate_bg_boxes(data, image_id, num_boxes=1000,
                               min_size=50, max_size=300,
                               iou_thresh=0.1, max_gt_frac=0.1,
                               max_contained_frac=0.3):
    """
    Generate candidate background boxes for a specific image from COCO data.
    """
    img_info = next(img for img in data["images"] if img["id"] == image_id)
    img_w, img_h = img_info["width"], img_info["height"]
    gt_boxes = [ann["bbox"] for ann in data["annotations"] if ann["image_id"] == image_id]

    bg_boxes = []
    attempts = 0
    while len(bg_boxes) < num_boxes and attempts < num_boxes * 10:
        attempts += 1
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        x = random.randint(0, img_w - w)
        y = random.randint(0, img_h - h)
        new_box = [x, y, w, h]

        bad = any(
            iou(new_box, gt) > iou_thresh or
            covers_too_much(new_box, gt, max_gt_frac) or
            contained_fraction_gt(new_box, gt) > max_contained_frac
            for gt in gt_boxes
        )

        if not bad:
            bg_boxes.append(new_box)

    return bg_boxes


# ------------------------
# Non-Maximum Suppression (NMS)
# ------------------------
def nms_boxes(boxes, iou_thresh=0.05):
    """
    Apply Non-Maximum Suppression to a list of boxes using area-based scoring.
    Returns a filtered list of boxes.
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]
    order = areas.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter_area = inter_w * inter_h
        iou_vals = inter_area / (areas[i] + areas[order[1:]] - inter_area)
        order = order[1:][iou_vals <= iou_thresh]

    return boxes[keep].tolist()


# ------------------------
# Utility function for external use
# ------------------------
def get_background_boxes_for_image(data, image_id, num_boxes=1000,
                                   min_size=50, max_size=300,
                                   iou_thresh=0.1, max_gt_frac=0.1,
                                   max_contained_frac=0.3,
                                   nms_iou_thresh=0.05):
    """
    Generate candidate background boxes for an image and apply NMS.
    Returns a list of filtered background boxes.
    """
    candidates = generate_candidate_bg_boxes(
        data, image_id, num_boxes=num_boxes,
        min_size=min_size, max_size=max_size,
        iou_thresh=iou_thresh,
        max_gt_frac=max_gt_frac,
        max_contained_frac=max_contained_frac
    )
    final_boxes = nms_boxes(candidates, iou_thresh=nms_iou_thresh)
    return final_boxes


# ------------------------
# Optional helper to load COCO JSON
# ------------------------
def load_coco_json(anno_path):
    """Load COCO JSON annotation file."""
    with open(anno_path, 'r') as f:
        return json.load(f)

