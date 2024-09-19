from ultralytics import YOLO
import os

os.chdir('/home/users/industry/hpcic/hpcic32/scratch/')

## get the trained model
model = YOLO("runs/segment/train3/weights/best.pt")

## train the model
model.val(data="Airplane_Defect_Detection_Seg_YOLOv9/test.yaml")
