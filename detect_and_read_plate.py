# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO  # pip install ultralytics
import torchvision.transforms as T
import torch.nn as nn

# ------------- تنظیمات -------------
YOLO_WEIGHTS = "yolo11x.pt"   # مسیر مدل YOLO که پلاک رو تشخیص میده
CHAR_CLF_WEIGHTS = "Model/char_clf.pth"  # مسیر مدل classifier کاراکتر
IMAGES_DIR = "images_car"
RESULTS_DIR = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# لیست حروف/اعداد مطابق با Iranis folder names (ترتیب ایندکس clf)
CLASSES = ['0','1','2','3','4','5','6','7','8','9','A','B','D','Gh','H','J','L','M','N','P','PuV','PwD','Sad','Sin','T','Taxi','V','Y']
# map index->char
IDX2CHAR = {i:CLASSES[i] for i in range(len(CLASSES))}
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------- مدل classifier نمونه (باید مشابه معماری ذخیره‌سازی باشه) -------------
class SimpleCharCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128,256), nn.ReLU(),
            nn.Linear(256,n_classes)
        )
    def forward(self,x): return self.net(x)

# load classifier
char_clf = SimpleCharCNN().to(DEVICE)
if os.path.exists(CHAR_CLF_WEIGHTS):
    print("Loading char classifier weights...")
    state = torch.load(CHAR_CLF_WEIGHTS, map_location=DEVICE)
    char_clf.load_state_dict(state)
else:
    print("Warning: char classifier weights not found. Predictions will be random.")

char_clf.eval()

# load yolo
yolo = YOLO(YOLO_WEIGHTS)

# transforms for classifier
tf = T.Compose([
    T.Resize((40,24)),  # ارتفاع کمتر عرض بیشتر؛ ممکنه نیاز به تنظیم داشته باشه
    T.ToTensor(),
    T.Normalize(0.5,0.5)
])

def segment_characters(plate_img_gray):
    """
    ورودی: تصویر پلاک خاکستری (numpy)
    خروجی: لیست از bbox های هر کاراکتر (x,y,w,h) به ترتیب از چپ به راست
    روش: threshold + contours + filter by size + sort by x
    """
    # تقویت کنتراست و threshold
    img = plate_img_gray.copy()
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # پاکسازی نویز
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    # پیدا کردن کانتورها
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    H, W = img.shape[:2]
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        # فیلتر اندازه: حذف اجسام خیلی کوچک یا خیلی بزرگ نسبت به پلاک
        if w < 5 or h < 10: continue
        if h < 0.3*H or h > 0.95*H:  # ولی پارامترها بسته به پلاک
            pass  # اجازه میدیم کمی بزرگ باشه، ولی میتونی stricter باشی
        boxes.append((x,y,w,h))
    # مرتب‌سازی از چپ به راست
    boxes = sorted(boxes, key=lambda b: b[0])
    return boxes

def predict_char(img_crop_gray):
    # img_crop_gray: numpy gray
    pil = Image.fromarray(img_crop_gray).convert('L')
    tensor = tf(pil).unsqueeze(0).to(DEVICE)  # 1CH
    with torch.no_grad():
        out = char_clf(tensor)
        pred = out.argmax(dim=1).item()
    return IDX2CHAR.get(pred, '?'), pred

# main loop
for fname in os.listdir(IMAGES_DIR):
    if not fname.lower().endswith(('.jpg','.png','.jpeg')): continue
    path = os.path.join(IMAGES_DIR, fname)
    img = cv2.imread(path)
    h,w = img.shape[:2]
    # YOLO predict
    results = yolo.predict(source=path, conf=0.25, device=DEVICE)
    # results is list; iterate
    annotated = img.copy()
    final_texts = []
    for r in results:
        # each r.boxes -> xyxy etc.
        if not hasattr(r, 'boxes') or r.boxes is None: continue
        boxes = r.boxes
        for i in range(len(boxes)):
            # xyxy absolute
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            cls = int(boxes.cls[i].cpu().numpy()) if hasattr(boxes, 'cls') else 0
            conf = float(boxes.conf[i].cpu().numpy())
            x1,y1,x2,y2 = xyxy
            # crop plate
            pad = 3
            x1p = max(0,x1-pad); y1p = max(0,y1-pad); x2p = min(w-1,x2+pad); y2p = min(h-1,y2+pad)
            plate = img[y1p:y2p, x1p:x2p]
            plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            # segment characters
            char_boxes = segment_characters(plate_gray)
            plate_text = ""
            # draw plate box
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
            for (cx,cy,cw,ch) in char_boxes:
                char_crop = plate_gray[cy:cy+ch, cx:cx+cw]
                # optional: pad to consistent size
                char_resized = cv2.resize(char_crop, (24,40), interpolation=cv2.INTER_LINEAR)
                label, idx = predict_char(char_resized)
                plate_text += label
                # draw char bbox relative to original image
                cv2.rectangle(annotated, (x1+cx, y1+cy), (x1+cx+cw, y1+cy+ch), (255,0,0), 1)
            # write recognized text above plate
            cv2.putText(annotated, plate_text, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            final_texts.append(plate_text)
    # save result
    out_path = os.path.join(RESULTS_DIR, fname)
    cv2.imwrite(out_path, annotated)
    print(f"Processed {fname} -> {final_texts}")
