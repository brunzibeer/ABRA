import json
import argparse
from pathlib import Path
from collections import defaultdict

from box_utils import get_background_boxes_for_image, load_coco_json
from tqdm import tqdm

def list_domains(annotations_folder: Path):
    '''
    Given the annotation folder, list all the domains available.
    Each domain is represented by a single JSON file.
    '''
    return [f for f in annotations_folder.glob("*.json")]

def generate_bbb_for_domain(domain: Path, args):
    gt_annotations = load_coco_json(domain)
    images = gt_annotations["images"]

    bbb_annotations = defaultdict(list)
    bbb_annotations["images"] = images
    bbb_annotations["categories"] = [{"supercategory": "none", "id": 999, "name": "background"}]
    for img in tqdm(images, desc=f"Processing {domain.stem}"):
        img_id = img["id"]
        bbb = get_background_boxes_for_image(
            gt_annotations, img_id,
            num_boxes=args.num_boxes, min_size=args.min_size, max_size=args.max_size,
            iou_thresh=args.iou_thresh, max_gt_frac=args.max_gt_frac, max_contained_frac=args.max_contained_frac,
            nms_iou_thresh=args.nms_iou_thresh
        )

        for i, box in enumerate(bbb):
            x, y, w, h = box
            bbb_annotations["annotations"].append({
                "segmentation": [],
                "area": w * h,
                "ignore": 0,
                "iscrowd": 0,
                "image_id": img_id,
                "bbox": [x, y, w, h],
                "category_id": 999,
                "id": i
            })
    return bbb_annotations

def save_bbb_as_json(bbb: dict, output_path: Path):
    with open(output_path, 'w') as f:
        json.dump(bbb, f, indent=4)

def main(args):
    domain_list = list_domains(args.annotations)
    for domain in domain_list:
        new_bbb = generate_bbb_for_domain(domain, args)
        output_path = args.output / f"{domain.stem}_bbb.json"
        save_bbb_as_json(new_bbb, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Shift category ids to start from 0")
    parser.add_argument("--annotations", "-a", required=False, type=Path, help="Percorso alla cartella contenente le annotazioni")
    parser.add_argument("--output", "-o", required=True, type=Path, help="Percorso alla cartella in cui salvare le nuove annotazioni delle bbb")
    parser.add_argument("--min_size", type=int, default=120, help="Minimum size of the background boxes")
    parser.add_argument("--max_size", type=int, default=500, help="Maximum size of the background boxes")
    parser.add_argument("--num_boxes", type=int, default=5000, help="Number of candidate background boxes to generate per image")
    parser.add_argument("--iou_thresh", type=float, default=0.1, help="Maximum IoU threshold with ground truth boxes")
    parser.add_argument("--max_gt_frac", type=float, default=0.1, help="Maximum fraction of the background box that can overlap with ground truth boxes")
    parser.add_argument("--max_contained_frac", type=float, default=0.3, help="Maximum fraction of the background box that can be contained within ground truth boxes")
    parser.add_argument("--nms_iou_thresh", type=float, default=0.05, help="IoU threshold for non-maximum suppression")
    args = parser.parse_args()

    main(args)