import torch
import cv2
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np
from train_char_clf import SimpleOCR  # مدل CNN قبلی که ساختی
import os

# مسیرها
MODEL_YOLO_PATH = "Model/yolov8n.pt"  # یا مسیر مدل YOLO خودت
MODEL_OCR_PATH = "Model/ocr_farsi_cnn.pth"

# لود مدل YOLO
yolo_model = YOLO(MODEL_YOLO_PATH)

# لود مدل OCR فارسی
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ocr_model = SimpleOCR(num_classes=len(os.listdir("Iranis Dataset Files_2"))).to(device)
ocr_model.load_state_dict(torch.load(MODEL_OCR_PATH, map_location=device))
ocr_model.eval()

# پیش‌پردازش مشابه آموزش
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# مسیر عکس ماشین
image_path = "images_car/2.jpg"
img = cv2.imread(image_path)

# YOLO Detect
results = yolo_model(img)[0]

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    cls = int(box.cls[0])

    # Crop پلاک
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    # نمایش پلاک شناسایی‌شده
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, "Plate", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # حالا OCR روی پلاک اعمال کن
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # در این مرحله فرض می‌کنیم هر کاراکتر جدا نیست — تست ساده روی پلاک کامل
    # در نسخه نهایی با segment کردن حروف دقیق‌تر می‌شه.
    pil_img = Image.fromarray(gray_crop)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    output = ocr_model(input_tensor)
    pred = torch.argmax(output, dim=1).item()

    # نام label از دایرکتوری دیتاست
    labels = sorted(os.listdir("Iranis Dataset Files_2"))
    detected_label = labels[pred]

    # نمایش label روی تصویر
    cv2.putText(img, detected_label, (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# ذخیره یا نمایش
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
