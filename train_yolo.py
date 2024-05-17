import torch
import os
import ultralytics
ultralytics.checks()
import comet_ml; comet_ml.init()

logger = 'Comet' #@param ['Comet', 'TensorBoard']

os.system('yolo train model=yolov8s.pt data=coco8.yaml epochs=50 imgsz=640')