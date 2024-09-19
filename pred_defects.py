from ultralytics import YOLO

# Load a model
model = YOLO('defect_detection_yolov9_tuned_best.pt')

# Define path to directory containing images and videos for inference
source = 'Bebop_Drone_2020-09-21T124026+0700_BA5969.mp4'

# Run inference on the source
results = model(source, save=True, stream=True)

