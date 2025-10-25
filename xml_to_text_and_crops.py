# -*- coding: utf-8 -*-
"""
نسخه اصلاح‌شده و مقاوم‌تر:
- تلاش می‌کند تصویر متناظر را از داخل همان فولدر یا با basename xml پیدا کند.
- خروجی‌ها:
    labels_txt/<split>.csv
    chars_dataset/<char_label>/*.jpg
    yolo_labels/<split>/*.txt  (خام px coords)
"""
import os
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm

ROOT = "LicensePlate"  # مسیر پایه که شامل train/validation/test هست
SPLITS = ["train", "validation", "test"]
OUT_TXT_DIR = "labels_txt"
CHARS_OUT = "chars_dataset"
YOLO_LABELS_OUT = "yolo_labels"
CLASSES_FILE = "char_classes.txt"

os.makedirs(OUT_TXT_DIR, exist_ok=True)
os.makedirs(CHARS_OUT, exist_ok=True)
os.makedirs(YOLO_LABELS_OUT, exist_ok=True)

possible_exts = [".jpg", ".jpeg", ".png", ".bmp"]

unique_chars = set()

def find_image_for_xml(xml_path, xml_root, split_dir):
    """
    Try several strategies to find the image corresponding to this xml:
    1) Use <filename> inside xml if present.
    2) Try basename(xml) + common extensions.
    3) Search split_dir for file names that start with basename.
    Returns full image path or None.
    """
    # 1) filename tag
    filename_tag = xml_root.find('filename')
    if filename_tag is not None and filename_tag.text:
        candidate = filename_tag.text.strip()
        # If candidate contains path parts, take basename
        cand_basename = os.path.basename(candidate)
        cand_path = os.path.join(split_dir, cand_basename)
        for ext in possible_exts + [""]:  # try as given and fallback add exts
            p = cand_path if ext == "" else (cand_path if cand_path.lower().endswith(ext) else cand_path + ext)
            if os.path.exists(p):
                return p
    # 2) try xml basename + exts
    xml_base = os.path.splitext(os.path.basename(xml_path))[0]
    for ext in possible_exts:
        p = os.path.join(split_dir, xml_base + ext)
        if os.path.exists(p):
            return p
    # 3) search for files starting with xml_base in split_dir (recursive)
    for root_dir, dirs, files in os.walk(split_dir):
        for f in files:
            if f.lower().startswith(xml_base.lower()):
                lower = f.lower()
                if any(lower.endswith(e) for e in [".jpg",".jpeg",".png",".bmp"]):
                    return os.path.join(root_dir, f)
    # 4) not found
    return None

def parse_xml_and_extract(split):
    split_dir = os.path.join(ROOT, split)
    if not os.path.isdir(split_dir):
        print(f"⚠️ Split folder not found: {split_dir}")
        return
    xml_files = [f for f in os.listdir(split_dir) if f.lower().endswith('.xml')]
    rows = []
    yolo_labels_split_dir = os.path.join(YOLO_LABELS_OUT, split)
    os.makedirs(yolo_labels_split_dir, exist_ok=True)
    miss_count = 0

    for fname in tqdm(xml_files, desc=f"Processing {split}"):
        xml_path = os.path.join(split_dir, fname)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"⚠️ Failed to parse XML {xml_path}: {e}")
            continue

        img_path = find_image_for_xml(xml_path, root, split_dir)
        if img_path is None:
            # report missing image but continue
            print(f"⚠️ Image for {xml_path} not found in {split_dir}")
            miss_count += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Failed to read image {img_path}")
            miss_count += 1
            continue
        H, W = img.shape[:2]

        objects = []
        for obj in root.findall('object'):
            name_tag = obj.find('name')
            bnd = obj.find('bndbox')
            if name_tag is None or bnd is None:
                continue
            label = name_tag.text.strip()
            try:
                xmin = int(float(bnd.find('xmin').text))
                ymin = int(float(bnd.find('ymin').text))
                xmax = int(float(bnd.find('xmax').text))
                ymax = int(float(bnd.find('ymax').text))
            except Exception:
                continue
            # clamp coordinates
            xmin = max(0, min(xmin, W-1))
            ymin = max(0, min(ymin, H-1))
            xmax = max(0, min(xmax, W-1))
            ymax = max(0, min(ymax, H-1))
            if xmax <= xmin or ymax <= ymin:
                continue
            objects.append((xmin, ymin, xmax, ymax, label))
            unique_chars.add(label)

        if len(objects) == 0:
            # no labeled objects
            continue

        # sort objects left->right by xmin
        objects = sorted(objects, key=lambda x: x[0])
        text = "".join([o[4] for o in objects])
        img_name = os.path.basename(img_path)
        rows.append((img_name, text))

        # save character crops
        for i,(xmin,ymin,xmax,ymax,label) in enumerate(objects):
            crop = img[ymin:ymax, xmin:xmax]
            out_dir = os.path.join(CHARS_OUT, label)
            os.makedirs(out_dir, exist_ok=True)
            out_fname = f"{os.path.splitext(img_name)[0]}_{i}.jpg"
            cv2.imwrite(os.path.join(out_dir, out_fname), crop)

        # write raw yolo label (pixel coords) for later conversion
        base = os.path.splitext(img_name)[0]
        yolo_label_path = os.path.join(yolo_labels_split_dir, base + ".txt")
        with open(yolo_label_path, "w", encoding="utf-8") as f:
            for (xmin,ymin,xmax,ymax,label) in objects:
                f.write(f"{label} {xmin} {ymin} {xmax} {ymax}\n")

    out_csv = os.path.join(OUT_TXT_DIR, f"{split}.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        for img_name, text in rows:
            f.write(f"{img_name},{text}\n")
    print(f"➡️ Written {out_csv} with {len(rows)} entries (missed {miss_count} xmls)")
    return

if __name__ == "__main__":
    for s in SPLITS:
        parse_xml_and_extract(s)

    # write classes file
    classes = sorted(list(unique_chars))
    with open(CLASSES_FILE, "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")
    print(f"✅ Extracted {len(classes)} unique classes, saved to {CLASSES_FILE}")
    print("Outputs:")
    print(" - per-split CSVs: labels_txt/<split>.csv")
    print(" - chars crops: chars_dataset/<char_label>/*.jpg")
    print(" - rough yolo labels (text coordinates): yolo_labels/<split>/*.txt")
