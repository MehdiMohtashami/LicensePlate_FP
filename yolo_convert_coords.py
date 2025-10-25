# -*- coding: utf-8 -*-
import os
from PIL import Image

YOLO_RAW_DIR = "yolo_labels"
OUT_DIR = "yolo_labels_norm"
CLASSES_FILE = "char_classes.txt"  # ساخته شده از اسکریپت قبل

os.makedirs(OUT_DIR, exist_ok=True)

# load classes -> map label->idx
with open(CLASSES_FILE, "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f if line.strip()]
label2idx = {c:i for i,c in enumerate(classes)}

for split in os.listdir(YOLO_RAW_DIR):
    in_split = os.path.join(YOLO_RAW_DIR, split)
    out_split = os.path.join(OUT_DIR, split)
    os.makedirs(out_split, exist_ok=True)
    # assume images are in LicensePlate/<split>/
    img_dir = os.path.join("LicensePlate", split)
    for fname in os.listdir(in_split):
        if not fname.endswith(".txt"): continue
        txt_path = os.path.join(in_split, fname)
        base = os.path.splitext(fname)[0]
        # find image
        for ext in ['.jpg','.png','.jpeg']:
            img_path = os.path.join(img_dir, base + ext)
            if os.path.exists(img_path):
                break
        if not os.path.exists(img_path):
            print("Image not found for", base); continue
        from PIL import Image
        W,H = Image.open(img_path).size

        out_lines = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5: continue
                label, xmin, ymin, xmax, ymax = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                if label not in label2idx:
                    print("Unknown label", label); continue
                cls = label2idx[label]
                x_center = (xmin + xmax) / 2.0 / W
                y_center = (ymin + ymax) / 2.0 / H
                w = (xmax - xmin) / W
                h = (ymax - ymin) / H
                out_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        with open(os.path.join(out_split, fname), "w", encoding="utf-8") as fo:
            fo.write("\n".join(out_lines))
    print("Converted split:", split)
print("✅ YOLO normalized labels written to", OUT_DIR)
