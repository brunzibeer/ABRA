############################################ ENVER MODE ############################################
import json
import random
import numpy as np

# ------------------------
# Helper functions
# ------------------------
def load_coco_annotations(json_path):
    """Load COCO-style annotations from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_image_and_boxes(image_path, ann_path):
    """
    Given an image path and annotation file path:
    - Extract its GT bounding boxes.
    - Get image dimensions from the JSON.
    """
    ann_data = load_coco_annotations(ann_path)

    # Find the image entry in JSON
    file_name = image_path.split('/')[-1]
    image_entry = next((img for img in ann_data['images'] if img['file_name'] == file_name), None)
    if image_entry is None:
        raise ValueError(f"Image {file_name} not found in annotation file.")

    image_id = image_entry['id']
    img_w, img_h = image_entry['width'], image_entry['height']

    # Extract annotations for this image
    boxes = [ann['bbox'] for ann in ann_data['annotations'] if ann['image_id'] == image_id]

    return (img_w, img_h), boxes

def fraction_covered(box_small, box_large):
    """Compute fraction of box_small covered by box_large."""
    xA = max(box_small[0], box_large[0])
    yA = max(box_small[1], box_large[1])
    xB = min(box_small[0] + box_small[2], box_large[0] + box_large[2])
    yB = min(box_small[1] + box_small[3], box_large[1] + box_large[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    smallArea = box_small[2] * box_small[3]
    return interArea / float(smallArea + 1e-6)

def is_fully_contained(inner, outer):
    """Check if box `inner` is fully contained within box `outer`."""
    x_in, y_in, w_in, h_in = inner
    x_out, y_out, w_out, h_out = outer
    return (x_in >= x_out and
            y_in >= y_out and
            x_in + w_in <= x_out + w_out and
            y_in + h_in <= y_out + h_out)
    
def is_compliant(box):
    """Check that box follows x0,y0,x1,y1 standard (x1>=x0, y1>=y0)."""
    x, y, w, h = box
    return (w >= 0) and (h >= 0)

def generate_background_boxes(gt_boxes, img_w, img_h, K, aspect_tolerance=0.2, min_dim=20, max_dim=200):
    """
    Generate K background bounding boxes that:
    - Are not fully contained inside GT boxes.
    - Do not fully contain GT boxes.
    - Neither covers more than 30% of the other (bbb vs GT).
    - Have aspect ratio similar to GT boxes (± tolerance).
    - Width and height between min_dim and max_dim.
    """

    background_boxes = []

    # Compute average aspect ratio of GT boxes
    gt_aspects = [(w / h) for (_, _, w, h) in gt_boxes if h > 0]
    avg_aspect = np.mean(gt_aspects) if gt_aspects else 1.0

    # Adaptive parameters
    max_attempts = 1000
    relax_steps = [
        {'aspect_tolerance': aspect_tolerance, 'overlap': 0.3, 'min_dim': min_dim, 'max_dim': max_dim},
        {'aspect_tolerance': aspect_tolerance * 1.5, 'overlap': 0.4, 'min_dim': min_dim * 0.8, 'max_dim': max_dim * 1.2},
        {'aspect_tolerance': aspect_tolerance * 2, 'overlap': 0.5, 'min_dim': min_dim * 0.5, 'max_dim': max_dim * 1.5},
        {'aspect_tolerance': 999, 'overlap': 1.0, 'min_dim': 1, 'max_dim': max(img_w, img_h)},
    ]
    step = 0
    attempts = 0

    while len(background_boxes) < K:
        if attempts > max_attempts:
            # Relax constraints
            step = min(step + 1, len(relax_steps) - 1)
            attempts = 0
        params = relax_steps[step]

        # Random width/height scaled from GT box sizes
        ref_box = random.choice(gt_boxes)
        w = random.uniform(0.5, 1.5) * ref_box[2]
        h = random.uniform(0.5, 1.5) * ref_box[3]

        # Dimension constraints
        if not (params['min_dim'] <= w <= params['max_dim'] and params['min_dim'] <= h <= params['max_dim']):
            attempts += 1
            continue

        # Random top-left corner
        x = random.uniform(0, img_w - w)
        y = random.uniform(0, img_h - h)
        candidate = [x, y, w, h]

        # Check aspect ratio
        aspect = w / (h + 1e-6)
        if not (avg_aspect * (1 - params['aspect_tolerance']) <= aspect <= avg_aspect * (1 + params['aspect_tolerance'])):
            attempts += 1
            continue

        # Check overlap, containment and coverage with all GT boxes
        valid = True
        for gt in gt_boxes:
            if fraction_covered(candidate, gt) > params['overlap']:
                valid = False
                break
            if fraction_covered(gt, candidate) > params['overlap']:
                valid = False
                break
            if is_fully_contained(candidate, gt):
                valid = False
                break
            if is_fully_contained(gt, candidate):
                valid = False
                break

        if not is_compliant(candidate):
            valid = False

        if valid:
            background_boxes.append(candidate)
        attempts += 1

    # ---
    # Alternative: Mask-based approach stub (for future extension)
    # 1. Create a binary mask of the image, marking GT boxes as occupied.
    # 2. Sample candidate boxes only from unoccupied regions.
    # 3. This is more efficient, but for many small GT boxes, feasible space may be fragmented.
    # 4. If not enough boxes are found, fallback to adaptive random sampling above.
    # ---
    return background_boxes
