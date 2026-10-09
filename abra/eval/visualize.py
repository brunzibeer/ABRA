import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# Distinct colours per category (BGR)
PALETTE = [
    (0, 200, 255),   # yellow-orange
    (0, 255, 128),   # green
    (255, 80,  80),  # blue
    (200,  0, 255),  # magenta
    (0, 165, 255),   # orange
    (255, 255,  0),  # cyan
    (128,   0, 255), # purple
    (0, 255, 255),   # yellow
]

THICKNESS = 15
FONT_SCALE = 0.8


def visualize_predictions(
    targets: list,
    results: list,
    cat_list: list,
    output_dir: Path,
    start_idx: int,
    batch_size: int,
    dataset,
    confidence_threshold: float = 0.3
):
    """
    Saves visual representations of model predictions with bounding boxes.
    
    Args:
        targets: List of target dictionaries containing ground truth annotations
        results: List of prediction dictionaries containing boxes, labels, and scores
        cat_list: List of category names
        output_dir: Path object pointing to the epoch-specific visualization directory
        start_idx: Starting index for naming saved images
        batch_size: Number of images in the current batch
        dataset: Dataset object with COCO API access
        confidence_threshold: Minimum confidence score to visualize a prediction
    """

    # Create images subdirectory
    save_dir = output_dir / "images"
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0

    for i in range(batch_size):
        # Check if we have a valid target for this index
        if i >= len(targets):
            continue

        # Get image information
        image_id = targets[i]['image_id'].item()

        try:
            # Load image information from COCO API
            img_info = dataset.coco.loadImgs(image_id)[0]
            image_path = Path(dataset.root) / img_info['file_name']

            # Load and convert image
            if not image_path.exists():
                print(f"WARNING: Image file not found: {image_path}")
                continue

            original_image = Image.open(image_path).convert('RGB')
            # Convert RGB -> BGR for OpenCV
            canvas = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
            h, w = canvas.shape[:2]

        except (FileNotFoundError, AttributeError, KeyError, IndexError) as e:
            print(f"ERROR: Unable to load image for image_id {image_id}: {e}")
            continue

        # Filter predictions by confidence threshold
        pred_scores = results[i]['scores'].cpu().numpy()
        pred_boxes  = results[i]['boxes'].cpu().numpy()
        pred_labels = results[i]['labels'].cpu().numpy()

        keep_indices = pred_scores >= confidence_threshold

        pred_boxes_filtered  = pred_boxes[keep_indices]
        pred_labels_filtered = pred_labels[keep_indices]
        pred_scores_filtered = pred_scores[keep_indices]

        # Draw predictions
        for box, label, score in zip(pred_boxes_filtered, pred_labels_filtered, pred_scores_filtered):
            x1, y1, x2, y2 = box

            # Clip coordinates to image boundaries
            x1, x2 = int(np.clip(x1, 0, w)), int(np.clip(x2, 0, w))
            y1, y2 = int(np.clip(y1, 0, h)), int(np.clip(y2, 0, h))

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Assign a stable colour to this category
            color = PALETTE[int(label) % len(PALETTE)]

            # Draw bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, THICKNESS)

        # Save image
        save_path = save_dir / f'prediction_{start_idx + i + 1:04d}.png'
        cv2.imwrite(str(save_path), canvas)

        saved_count += 1

    return 1