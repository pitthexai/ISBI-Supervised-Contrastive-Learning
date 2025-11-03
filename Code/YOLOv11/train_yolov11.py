from ultralytics import YOLO
from PIL import Image

# Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="xray_knee.yaml", epochs=50)

# Optional quick check on image mode (kept from original)
# img = Image.open('C:/kl_grading/yolov11/datasets/xray_datasets/yolo_datasets/full_images/9999878.jpg')
# print(img.mode)
