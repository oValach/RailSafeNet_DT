from ultralyticsplus import YOLO, render_result
import cv2

# load model
model = YOLO('ultralyticsplus/yolov8s')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# set image
#image = cv2.imread('RailNet_DT/railway_dataset/media/images/44aabd7ea3e4a32e034f/frame_132280.png')
image = cv2.imread('RailNet_DT/rs19_val/jpgs/rs19_val/rs00135.jpg')
results = model.predict(image)

# observe results
print(results[0].boxes)
render = render_result(model=model, image=image, result=results[0])
render.show()