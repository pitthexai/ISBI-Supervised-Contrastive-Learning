from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
from PIL import Image

# Load a model
model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("C:/kl_grading/yolov11/runs/detect/train/weights/best.pt")  # load a custom model

# Predict with the model
image_path = "C:/kl_grading/yolov11/datasets/xray_datasets/yolo_datasets/full_images/9000099.jpg"
source_folder = "C:/kl_grading/yolov11/datasets/xray_datasets/yolo_datasets/full_images/"
output_folder = "C:/kl_grading/yolov11/datasets/xray_datasets/yolo_datasets/cropped_images/"
results = model(source=source_folder, conf=0.8, save=True, save_txt=True, name="test_infer")  # predict on images
# results = model(image_path, conf=0.5, save=True, save_txt=True, name="test_infer")

# Process results list
for result in results:
    boxes = result.boxes.xyxy.cpu().tolist()
    w = result.boxes.orig_shape[1]
    image_path = result.path
    image_name = image_path.split("\\")[-1]
    image_name = image_name.split('.')[0]
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Iterate through the bounding boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        # Crop the object using the bounding box coordinates
        ultralytics_crop_object = img[int(y1):int(y2), int(x1):int(x2)]
        # Save the cropped object as an image
        if (x1 + x2) / 2 < w / 2:
            side = 'r'
        else:
            side = 'l'

        cv2.imwrite(output_folder + image_name + '_' + side + '.jpg', ultralytics_crop_object)

    # if boxes is not None:
    #     for box, cls in zip(boxes, clss):
    #         idx += 1
    #         annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])
    #         crop_obj = im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
    #         cv2.imwrite(os.path.join(crop_dir_name, str(idx) + ".png"), crop_obj)

    # boxes = result.boxes  # Boxes object for bounding box outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    # obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    # result.save(filename="result.jpg")  # save to disk
