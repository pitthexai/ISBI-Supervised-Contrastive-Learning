# Tutorial:
Followed these instructions to train the yolo11 model
1. https://docs.ultralytics.com/models/yolo11/#usage-examples
2. https://docs.ultralytics.com/tasks/detect/#export

## Train yolo11 on custom datasets:
Prepare the data in formated folder and run this command to kick off model training
```
yolo train model=yolo11n.pt data=xray_knee.yaml epochs=10
```
## Inference the trained yolo11 model on new datasets, also crop the detected box:
```
python train.py
```
