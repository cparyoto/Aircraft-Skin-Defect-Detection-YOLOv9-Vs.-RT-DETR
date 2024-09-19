########## import libraries ##########

from ultralytics import YOLO
import os


########## change working directory ##########

os.chdir('/home/users/industry/hpcic/hpcic32/scratch/')


########## run code ##########

## define the model
model = YOLO("yolov8x.pt")

## train the model
model.tune(data="datasets/data.yaml", \
epochs=300, imgsz=640, device=[0, 1, 2, 3], \
plots=False, save=False)

