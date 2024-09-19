from ultralytics import RTDETR
import os

os.chdir('/home/users/industry/hpcic/hpcic32/scratch/')

## get the trained model
model = RTDETR("runs/detect/train873/weights/best.pt")

## train the model
model.val(data="datasets_2/test.yaml", device='cuda:0')


