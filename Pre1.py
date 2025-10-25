import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

YOLO_WEIGHTS = "yolo11x.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OCR_MODEL_PATH = "crnn_ocr.pth"
MAX_PLATE_WIDTH = 400
CHARS = list("۰۱۲۳۴۵۶۷۸۹ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")
char_to_idx = {c:i+1 for i,c in enumerate(CHARS)}
idx_to_char = {i+1:c for i,c in enumerate(CHARS)}

class DummyOCR:
    def __init__(self):
        pass
    def predict(self, pil_image):
        return "۱۲الف"

def load_ocr_model(path):
    if os.path.exists(path):
        model = torch.load(path, map_location=DEVICE)
        model.eval()
        return model
    else:
        return DummyOCR()

def yolo_xywh_to_xyxy(x, y, w, h, img_w, img_h):
    x_c = x * img_w
    y_c = y * img_h
    w_abs = w * img_w
    h_abs = h * img_h
    x1 = int(max(0, x_c - w_abs/2))
    y1 = int(max(0, y_c - h_abs/2))
    x2 = int(min(img_w-1, x_c + w_abs/2))
    y2 = int(min(img_h-1, y_c + h_abs/2))
    return x1, y1, x2, y2

def detect_and_recognize(image_path, yolo_model, ocr_model, conf_thres=0.4):
    img = cv2.imread(image_path)
    h,w = img.shape[:2]
    results = yolo_model.predict(source=image_path, conf=conf_thres, device=DEVICE, save=False)
    plates = []
    for res in results:
        boxes = res.boxes
        if boxes is None: continue
        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            xywh = boxes.xywhn[i].cpu().numpy()
            x,y,ww,hh = xywh
            x1,y1,x2,y2 = yolo_xywh_to_xyxy(x,y,ww,hh,w,h)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("L")
            pil = pil.resize((MAX_PLATE_WIDTH, int(MAX_PLATE_WIDTH * pil.size[1] / pil.size[0])), Image.BILINEAR)
            if hasattr(ocr_model, "predict"):
                pred_text = ocr_model.predict(pil)
            else:
                pred_text = infer_crnn(ocr_model, pil)
            plates.append({
                "bbox": (x1,y1,x2,y2),
                "conf": conf,
                "text": pred_text
            })
    return plates

def infer_crnn(model, pil_image):
    return "۱۲۳الف"

if __name__ == "__main__":
    # load YOLO
    yolo = YOLO(YOLO_WEIGHTS)  # ultralytics.YOLO
    ocr = load_ocr_model(OCR_MODEL_PATH)
    test_image = "images_car/2.jpg"
    out = detect_and_recognize(test_image, yolo, ocr, conf_thres=0.25)
    print("Results:", out)
# image 1/1 C:\Users\Parsa\PycharmProjects\LicensePlate_FP\images_car\2.jpg: 416x640 2 cars, 516.0ms
# Speed: 2.6ms preprocess, 516.0ms inference, 7.1ms postprocess per image at shape (1, 3, 416, 640)
# Results: [{'bbox': (184, 113, 464, 338), 'conf': 0.9513582587242126, 'text': '۱۲الف'}, {'bbox': (412, 128, 429, 142), 'conf': 0.6885111927986145, 'text': '۱۲الف'}]

