import argparse
import jsonlines
from tqdm import tqdm

def coco_to_xyxy(bbox):
    x, y, width, height = bbox
    x1 = round(x, 2)
    y1 = round(y, 2)
    x2 = round(x + width, 2)
    y2 = round(y + height, 2)
    return [x1, y1, x2, y2]


def coco2odvg(args):
    from pycocotools.coco import COCO
    coco = COCO(args.input)

    cats = coco.loadCats(coco.getCatIds())
    id_to_name = {cat["id"]: cat["name"] for cat in cats}
    
    if args.idmap == "sequential":
        sorted_ids = sorted(id_to_name.keys())
        idmap = {cat_id: idx for idx, cat_id in enumerate(sorted_ids)}
    else:
        idmap = {cat_id: cat_id for cat_id in id_to_name.keys()}

    metas = []

    for img_id, img_info in tqdm(coco.imgs.items()):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        instance_list = []
        for ann_id in ann_ids:
            ann = coco.anns[ann_id]
            bbox = ann["bbox"]
            bbox_xyxy = coco_to_xyxy(bbox)
            label = ann["category_id"]

            if label not in idmap:
                continue

            label_trans = idmap[label]
            category = id_to_name[label]

            instance_list.append(
                {
                    "bbox": bbox_xyxy,
                    "label": label_trans,
                    "category": category,
                }
            )
        metas.append(
            {
                "filename": img_info["file_name"],
                "height": img_info["height"],
                "width": img_info["width"],
                "detection": {
                    "instances": instance_list,
                },
            }
        )
    print("  == dump meta ...")
    with jsonlines.open(args.output, mode="w") as writer:
        writer.write_all(metas)
    print("  == done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("coco to odvg format.", add_help=True)
    parser.add_argument("--input", '-i', required=True, type=str, help="COCO input annotation json")
    parser.add_argument("--output", '-o', required=True, type=str, help="Output JSONLines file")
    parser.add_argument("--idmap", default='identity', choices=["identity", "sequential"], help="How to map category ids: keep original (identity) or remap to 0..N-1 (sequential)")
    args = parser.parse_args()

    coco2odvg(args)
