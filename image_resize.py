import cv2
import os

PATH_jpgs = 'RailNet_DT/rs19_val/jpgs/vozovna'
for filename_img in os.listdir(PATH_jpgs):

    image = cv2.imread(os.path.join(PATH_jpgs, filename_img))
    resized_image = cv2.resize(image, (1920, 1080))
    cv2.imwrite('RailNet_DT/rs19_val/jpgs/vozovna/{}_res.png'.format(filename_img), resized_image)
