########## import libraries ##########

from ultralytics import YOLO
import os


########## change working directory ##########

os.chdir('/home/users/industry/hpcic/hpcic32/scratch/')


########## run code ##########

## define the model
model = YOLO("yolov9e.pt")
# model = YOLO("runs/detect/tune2/weights/last.pt")

## train the model
model.train(data="datasets_2/data.yaml", \
	epochs=150, device=[0,1,2,3], \
        patience=100, close_mosaic=0)
