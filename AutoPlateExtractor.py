import cv2
import os
import numpy as np
from ultralytics import YOLO

# مسیر عکس‌های ورودی
input_dir = "images_car"
output_images = "datasets/plates/images/train"
output_labels = "datasets/plates/labels/train"

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# مدل YOLO عمومی برای تشخیص خودرو
model = YOLO("yolo11x.pt")  # یا yolov11n.pt

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for idx, filename in enumerate(image_files):
    path = os.path.join(input_dir, filename)
    img = cv2.imread(path)

    results = model(img, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        if model.names[cls] != "car":
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # فقط ناحیه ماشین
        car_crop = img[y1:y2, x1:x2]

        # تبدیل به grayscale و threshold برای حدس پلاک
        gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 100, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        possible_plates = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            ratio = w / float(h)
            area = w * h

            # شرط هندسی پلاک
            if 2 < ratio < 6 and 2000 < area < 30000:
                possible_plates.append((x, y, w, h))

        # اگر چند پلاک پیدا شد، بزرگترین رو انتخاب کن
        if possible_plates:
            x, y, w, h = max(possible_plates, key=lambda a: a[2] * a[3])
            plate_crop = car_crop[y:y + h, x:x + w]

            # ذخیره تصویر
            out_img_path = os.path.join(output_images, f"plate_{idx}.jpg")
            cv2.imwrite(out_img_path, plate_crop)

            # ذخیره label YOLO برای پلاک
            img_h, img_w = img.shape[:2]
            x_center = (x + w / 2 + x1) / img_w
            y_center = (y + h / 2 + y1) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            label_path = os.path.join(output_labels, f"plate_{idx}.txt")
            with open(label_path, "w") as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            print(f"✅ Saved plate {idx} from {filename}")
        else:
            print(f"⚠️ No plate detected in {filename}")
