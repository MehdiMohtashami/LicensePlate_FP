import cv2 as cv
import numpy as np
import time
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
import nltk
from nltk.corpus import stopwords

def draw_line(path):
    #cv.line(image, start_point, end_point, color, thickness)
    image= cv.imread(path)
    start_p= (0,0)
    end_p= (100,500)
    color= (255,0,0)
    thick= 10
    img= cv.line(image,start_p,end_p,color,thick)
    return img

def draw_box(path):
    # cv.rectangle(image,start_point, end_point, color, thickness)
    img = cv.imread(path)
    start_p = (0,0)
    end_p=(500,500)
    color= (0,0,255)
    thick=10
    img = cv.rectangle(img,start_p,end_p,color,thick)
    return img

def draw_circle(path):
    img= cv.imread(path)
    center=(200,200)
    radios = 200
    color = (0,255,0)
    thick = -1
    img=cv.circle(img,center,radios, color,thick)
    return img

def display_image():
    path="images/DP_Course.jpeg"
    # img= draw_line(path)
    # img_b=draw_box(path)
    img= draw_circle(path)
    cv.imshow("img",img)
    # cv.imshow("img", img)
    cv.waitKey(0)

def process_video():
    path = "video/Traffic.mp4"
    vs = cv.VideoCapture(path)
    model= YOLO("Model/yolov8n.pt")
    while True:
        (grabbed, frame)= vs.read() #Tupple
        # print(grabbed) Boolean
        if not grabbed:
            break

        model.predict(frame, stream=False, classes=[2], save=True)
        time.sleep(1)


sample_image = "images/Students.jpg"

model_v11x = YOLO("yolo11x.pt")
# result = model_v11x.predict()
# result = model_v11x.predict(source=sample_image,save=True)
selected_class=[0]
results = model_v11x.predict(source=sample_image,classes=selected_class,save=True, conf=0.1)
