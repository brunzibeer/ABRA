"""M3FD annotations at a glance (RGB -> thermal extension), CPU only, standard library. Reads the unpacked
``M3FD_Detection.zip`` (``Annotation/*.xml`` Pascal VOC, ``Ir/*.png``, ``Vis/*.png``) and prints what the converter
needs to know: class names with box and image counts, object flags, the xml tags in use, boxes outside their image,
xml sizes that differ from the PNGs, the numbering of the files (gaps hint at scene boundaries), and one raw xml.

    python tools/m3fd_summary.py datasets/m3fd/raw | tee datasets/m3fd/annotation_summary.txt
"""
import argparse
import collections
import struct
import xml.etree.ElementTree as ET
from pathlib import Path


def png_size(path):
    with path.open("rb") as f:
        return struct.unpack(">II", f.read(24)[16:24])


def summary(raw):
    anns = sorted((raw / "Annotation").glob("*.xml"))
    if not anns:
        raise SystemExit(f"no xml in {raw / 'Annotation'}")
    tags, names, images_with, flags = (collections.Counter() for _ in range(4))
    per_image, problems = collections.Counter(), collections.Counter()
    for a in anns:
        r = ET.parse(a).getroot()
        tags.update(el.tag for el in r.iter())
        if Path(r.findtext("filename") or "").stem != a.stem:
            problems["filename differs from the xml name"] += 1
        w, h = (int(float(r.findtext(f"size/{k}") or 0)) for k in ("width", "height"))
        for modality in ("Ir", "Vis"):
            image = raw / modality / f"{a.stem}.png"
            if not image.exists():
                problems[f"no {modality} image"] += 1
            elif png_size(image) != (w, h):
                problems[f"size in xml != {modality} png"] += 1
        objs = r.findall("object")
        per_image[len(objs)] += 1
        images_with.update({o.findtext("name") for o in objs})
        for o in objs:
            names[o.findtext("name")] += 1
            flags.update(f"{k}={o.findtext(k)}" for k in ("difficult", "truncated", "pose"))
            x1, y1, x2, y2 = (float(o.findtext(f"bndbox/{k}")) for k in ("xmin", "ymin", "xmax", "ymax"))
            if x2 <= x1 or y2 <= y1:
                problems["empty box (max <= min)"] += 1
            elif x2 - x1 < 1 or y2 - y1 < 1:
                problems["box thinner than 1 px"] += 1
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                problems["box outside the image"] += 1

    stems = [a.stem for a in anns]
    print(f"{len(anns)} xml files; names {stems[:3]} ... {stems[-3:]}")
    numbers = sorted(int(s) for s in stems if s.isdigit())
    if numbers:
        gaps = [(a, b) for a, b in zip(numbers, numbers[1:]) if b != a + 1]
        print(f"numeric names {numbers[0]}..{numbers[-1]}, {len(gaps)} gaps" + (f", e.g. {gaps[:5]}" if gaps else ""))
    print("boxes per class:  " + ", ".join(f"{k} {v}" for k, v in names.most_common()))
    print("images per class: " + ", ".join(f"{k} {v}" for k, v in images_with.most_common()))
    print(f"boxes: {sum(names.values())}; images without boxes: {per_image[0]}; max boxes in one image: {max(per_image)}")
    print("object flags: " + ", ".join(f"{k} {v}" for k, v in sorted(flags.items())))
    print("xml tags: " + ", ".join(f"{k} {v}" for k, v in tags.most_common()))
    print("problems: " + (", ".join(f"{k} {v}" for k, v in problems.items()) or "none"))
    print(f"\n{anns[0].name}:\n{anns[0].read_text()[:1200]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("raw", type=Path, help="unpacked M3FD_Detection.zip (holds Annotation/, Ir/, Vis/)")
    summary(p.parse_args().raw)
