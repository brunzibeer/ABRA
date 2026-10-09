import json 
from datetime import datetime

def save_predictions_as_coco(targets, results, cat_list, dataset, output_path, confidence_threshold=0.3):
    """
    Saves the model's predictions to a COCO format annotation file,
    using the original image filenames.

    Args:
        targets (list): A list of dictionaries containing ground truth information.
        results (list): A list of dictionaries containing the model's predictions.
        cat_list (list): A list of category names.
        dataset: The dataset object (e.g., data_loader.dataset) which contains the COCO API.
        output_path (str): The path for the output JSON file.
        confidence_threshold (float): The confidence score threshold to filter predictions.
    """
    
    info = {
        "year": datetime.now().year,
        "version": "1.0",
        "description": "Predicted annotations from Grounding DINO model",
        "contributor": "Grounding DINO",
    }

    categories = [{"id": i, "name": cat_name, "supercategory": "none"} 
                  for i, cat_name in enumerate(cat_list)]

    images_info = []
    annotations = []
    annotation_id = 1
    processed_image_ids = set()

    print(f"Starting the creation of COCO annotations with a confidence threshold > {confidence_threshold}...")

    for target, result in zip(targets, results):
        image_id = target['image_id'].item()

        if image_id not in processed_image_ids:
            try:
                img_info = dataset.coco.loadImgs(image_id)[0]
                file_name = img_info['file_name']
                width = img_info['width']
                height = img_info['height']
            except (AttributeError, KeyError, IndexError) as e:
                print(f"Warning: Could not find filename for image_id {image_id} via dataset.coco. Error: {e}. Using a placeholder.")
                orig_size = target['orig_size'].cpu().numpy()
                width = int(orig_size[1])
                height = int(orig_size[0])
                file_name = f"image_{image_id:012d}.jpg"

            images_info.append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": file_name,
            })
            processed_image_ids.add(image_id)

        pred_scores = result['scores'].cpu().numpy()
        keep_indices = pred_scores >= confidence_threshold

        pred_boxes = result['boxes'].cpu().numpy()[keep_indices]
        pred_labels = result['labels'].cpu().numpy()[keep_indices]
        
        for box, label in zip(pred_boxes, pred_labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(label),
                "area": int(width * height),
                "bbox": [int(x1), int(y1), int(width), int(height)],
                "iscrowd": 0,
            })
            annotation_id += 1

    coco_output = {
        "info": info,
        "images": images_info,
        "annotations": annotations,
        "categories": categories,
    }

    with open(output_path, 'w') as f:
        json.dump(coco_output, f, indent=4)

    print(f"COCO annotations successfully saved to: {output_path}")