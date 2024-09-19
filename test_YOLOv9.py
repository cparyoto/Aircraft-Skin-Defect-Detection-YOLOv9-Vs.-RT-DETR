from ultralytics import YOLO
import os

os.chdir('/home/users/industry/hpcic/hpcic32/scratch/')

## get the trained model
model = YOLO("runs/detect/train917/weights/best.pt")

## train the model
model.val(data="datasets_2/test.yaml", device=[0])


