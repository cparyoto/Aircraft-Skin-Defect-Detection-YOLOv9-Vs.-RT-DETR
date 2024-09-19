########## import libraries ##########

from ultralytics import YOLO
import os


########## change working directory ##########

os.chdir('/home/users/industry/hpcic/hpcic32/scratch/')


########## run code ##########

## define the model
model = YOLO("yolov9e-seg.pt")

## train the model
model.train(data="Airplane_Defect_Detection_Seg_YOLOv9/data.yaml", \
epochs=500, device=[0,1])
