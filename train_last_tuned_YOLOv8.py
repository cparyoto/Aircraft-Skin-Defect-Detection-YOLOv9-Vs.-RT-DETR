########## import libraries ###################################################
from ultralytics import YOLO
import os

os.chdir('/home/users/sutd/1007677/scratch/AirplaneDefectDetection/')

## define the model
# model = YOLO("yolov8x.pt")
model = YOLO("runs/detect/tune9/weights/last.pt")

## train the model
model.train(data="data.yaml", epochs=300, \
batch=16, imgsz=640, optimizer='auto', save_period=10, \
device=[0, 1], patience=100)
