# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader
# import os
#
# # مسیر دیتاست
# DATASET_DIR = "Iranis Dataset Files_2"
#
# # تنظیمات پیش‌پردازش
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
#
#
# # مدل ساده CNN برای OCR فارسی
# class SimpleOCR(nn.Module):
#     def __init__(self, num_classes):
#         super(SimpleOCR, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout = nn.Dropout(0.25)
#         self.fc1 = nn.Linear(64 * 14 * 14, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#
#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.max_pool2d(x, 2)
#         x = torch.relu(self.conv2(x))
#         x = torch.max_pool2d(x, 2)
#         x = torch.flatten(x, 1)
#         x = self.dropout(x)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#
# # -------------------------------------------------------------
# # 👇 تمام کد اجرا باید داخل این بخش باشد
# # -------------------------------------------------------------
# if __name__ == "__main__":
#     # لود دیتاست
#     dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
#     num_classes = len(dataset.classes)
#
#     # تقسیم دیتاست به train/test
#     train_size = int(0.8 * len(dataset))
#     test_size = len(dataset) - train_size
#     train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
#
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)  # مهم: num_workers=0 در ویندوز
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
#
#     # تنظیم مدل، لاک‌لاس، و اپتیمایزر
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = SimpleOCR(num_classes).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#     # آموزش
#     epochs = 5
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for imgs, labels in train_loader:
#             imgs, labels = imgs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(imgs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {total_loss / len(train_loader):.4f}")
#
#     # ارزیابی
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for imgs, labels in test_loader:
#             imgs, labels = imgs.to(device), labels.to(device)
#             outputs = model(imgs)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     acc = 100 * correct / total
#     print(f"Accuracy: {acc:.2f}%")
#
#     # ذخیره مدل
#     os.makedirs("Model", exist_ok=True)
#     torch.save(model.state_dict(), "Model/ocr_farsi_cnn.pth")
#     print("✅ مدل ذخیره شد در مسیر: Model/ocr_farsi_cnn.pth")



# -*- coding: utf-8 -*-
import os, random
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm import tqdm

DATA_ROOT = "chars_dataset"  # produced by xml_to_text_and_crops
SAVE_PATH = "Model/char_clf.pth"
BATCH = 64
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT,d))])
CLS2IDX = {c:i for i,c in enumerate(CLASSES)}
print("Classes:", CLASSES)

class CharDataset(Dataset):
    def __init__(self, files, tf):
        self.files = files
        self.tf = tf
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        p, label = self.files[idx]
        img = Image.open(p).convert('L')
        img = self.tf(img)
        return img, label

# gather
all_files = []
for cls in CLASSES:
    cls_dir = os.path.join(DATA_ROOT, cls)
    for f in os.listdir(cls_dir):
        if f.lower().endswith(('.png','.jpg','.jpeg')):
            all_files.append((os.path.join(cls_dir,f), CLS2IDX[cls]))

train_files, val_files = train_test_split(all_files, test_size=0.15, random_state=42, stratify=[l for _,l in all_files])
print(f"Train samples: {len(train_files)}  Val samples: {len(val_files)}")

tf = transforms.Compose([
    transforms.Resize((40,24)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_ds = CharDataset(train_files, tf)
val_ds = CharDataset(val_files, tf)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

class SimpleCharCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128,256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,n_classes)
        )
    def forward(self,x): return self.net(x)

model = SimpleCharCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

best_val = 0.0
for epoch in range(EPOCHS):
    model.train()
    tloss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(DEVICE); labels = labels.to(DEVICE)
        opt.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward(); opt.step()
        tloss += loss.item()
    print(f"Epoch {epoch+1} train_loss={tloss/len(train_loader):.4f}")

    # val
    model.eval()
    correct=0; total=0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE); labels = labels.to(DEVICE)
            out = model(imgs)
            preds = out.argmax(dim=1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)
    acc = correct/total
    print(f"Val acc: {acc:.4f}")
    if acc > best_val:
        best_val = acc
        torch.save(model.state_dict(), SAVE_PATH)
        print("Saved best model", SAVE_PATH)

print("Training finished. Best val:", best_val)



# C:\Users\Parsa\PycharmProjects\LicensePlate_FP\.venv\Scripts\python.exe C:\Users\Parsa\PycharmProjects\LicensePlate_FP\train_char_clf.py
# Classes: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'الف', 'ب', 'ت', 'ث', 'ج', 'د', 'ز', 'س', 'ش', 'ص', 'ط', 'ع', 'ق', 'ل', 'م', 'ن', 'ه\u200d', 'و', 'پ', 'ژ (معلولین و جانبازان)', 'ی']
# Train samples: 160491  Val samples: 28322
# Epoch 1/15: 100%|██████████| 2508/2508 [36:51<00:00,  1.13it/s]
# Epoch 1 train_loss=0.7115
# Val acc: 0.9549
# Saved best model Model/char_clf.pth
# Epoch 2/15: 100%|██████████| 2508/2508 [01:50<00:00, 22.70it/s]
# Epoch 2 train_loss=0.1510
# Val acc: 0.9643
# Saved best model Model/char_clf.pth
# Epoch 3/15: 100%|██████████| 2508/2508 [01:45<00:00, 23.85it/s]
# Epoch 3 train_loss=0.1085
# Val acc: 0.9722
# Saved best model Model/char_clf.pth
# Epoch 4/15: 100%|██████████| 2508/2508 [01:45<00:00, 23.80it/s]
# Epoch 4 train_loss=0.0882
# Val acc: 0.9766
# Saved best model Model/char_clf.pth
# Epoch 5/15: 100%|██████████| 2508/2508 [01:45<00:00, 23.83it/s]
# Epoch 5 train_loss=0.0766
# Epoch 6/15:   0%|          | 0/2508 [00:00<?, ?it/s]Val acc: 0.9799
# Saved best model Model/char_clf.pth
# Epoch 6/15: 100%|██████████| 2508/2508 [01:48<00:00, 23.09it/s]
# Epoch 6 train_loss=0.0692
# Val acc: 0.9807
# Saved best model Model/char_clf.pth
# Epoch 7/15: 100%|██████████| 2508/2508 [01:46<00:00, 23.48it/s]
# Epoch 7 train_loss=0.0625
# Val acc: 0.9816
# Saved best model Model/char_clf.pth
# Epoch 8/15: 100%|██████████| 2508/2508 [01:44<00:00, 23.90it/s]
# Epoch 8 train_loss=0.0584
# Epoch 9/15:   0%|          | 0/2508 [00:00<?, ?it/s]Val acc: 0.9829
# Saved best model Model/char_clf.pth
# Epoch 9/15: 100%|██████████| 2508/2508 [01:43<00:00, 24.20it/s]
# Epoch 9 train_loss=0.0553
# Epoch 10/15:   0%|          | 0/2508 [00:00<?, ?it/s]Val acc: 0.9831
# Saved best model Model/char_clf.pth
# Epoch 10/15: 100%|██████████| 2508/2508 [01:45<00:00, 23.88it/s]
# Epoch 10 train_loss=0.0521
# Val acc: 0.9835
# Saved best model Model/char_clf.pth
# Epoch 11/15: 100%|██████████| 2508/2508 [01:48<00:00, 23.15it/s]
# Epoch 11 train_loss=0.0491
# Epoch 12/15:   0%|          | 0/2508 [00:00<?, ?it/s]Val acc: 0.9810
# Epoch 12/15: 100%|██████████| 2508/2508 [01:44<00:00, 23.98it/s]
# Epoch 12 train_loss=0.0469
# Epoch 13/15:   0%|          | 0/2508 [00:00<?, ?it/s]Val acc: 0.9814
# Epoch 13/15: 100%|██████████| 2508/2508 [01:43<00:00, 24.21it/s]
# Epoch 13 train_loss=0.0444
# Val acc: 0.9845
# Saved best model Model/char_clf.pth
# Epoch 14/15: 100%|██████████| 2508/2508 [01:44<00:00, 23.99it/s]
# Epoch 14 train_loss=0.0424
# Val acc: 0.9853
# Saved best model Model/char_clf.pth
# Epoch 15/15: 100%|██████████| 2508/2508 [01:47<00:00, 23.38it/s]
# Epoch 15 train_loss=0.0404
# Val acc: 0.9825
# Training finished. Best val: 0.9853470800084739
#
# Process finished with exit code 0
