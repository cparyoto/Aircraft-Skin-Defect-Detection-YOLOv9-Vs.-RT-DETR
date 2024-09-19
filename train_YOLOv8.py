########## import libraries ##########

from ultralytics import YOLO
import os


########## change working directory ##########

os.chdir('/home/users/industry/hpcic/hpcic32/scratch/')


########## run code ##########

## define the model
model = YOLO("yolov8x.pt")
# model = YOLO("runs/detect/tune9/weights/last.pt")

## train the model
model.train(data="datasets/data.yaml", \
epochs=300, imgsz=640, device=[0,1])
