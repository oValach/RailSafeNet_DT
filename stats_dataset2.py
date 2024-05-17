import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import os

mask_path = "RailNet_DT\\railway_dataset\media\images\mask"
img_path = "RailNet_DT\\railway_dataset\media\images"
eda_path = "RailNet_DT\\railway_dataset\eda_table.table.json"

data = json.load(open(eda_path, 'r'))

vehicles = {}
daytimes = {}
weathers = {}

plot = 1
counter_masked = 0
counter_notmasked = 0
for item in enumerate(data["data"]):
    img_name = item[1][1]["path"]
    img_sha = item[1][1]["sha256"][0:20]
    mask_name = item[1][1]["masks"]["ground_truth"]["path"]
    vehicle = item[1][4]
    daytime = item[1][7]
    weather = item[1][8]
    
    print(img_name)
    image = cv2.imread(os.path.join('RailNet_DT\\railway_dataset', img_name))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()
    
    if vehicle not in vehicles:
        vehicles[vehicle] = 1
    else:
        vehicles[vehicle] += 1
    if daytime not in daytimes:
        daytimes[daytime] = 1
    else:
        daytimes[daytime] += 1
    if weather not in weathers:
        weathers[weather] = 1
    else:
        weathers[weather] += 1    

    if os.path.isfile(os.path.join(img_path,img_sha)) and os.path.isfile(os.path.join("RailNet_DT\\railway_dataset",mask_name)):
        counter_masked+=1
    else:
        counter_notmasked+=1
print("Masked: {}, Not masked: {}".format(counter_masked, counter_notmasked))

plot = 1
counter_rails = 0
counter_siderails = 0
for filename in os.listdir(mask_path):
    image = cv2.imread(os.path.join(mask_path, filename))
    unique = np.unique(image)
    if (0 in unique) and (1 in unique):
        counter_rails += 1
    if 2 in unique:
        counter_siderails += 1
    
    if plot:
        plt.imshow(image[:,:,0],cmap="gray")
        plt.show()

print("Rails: {}, Side rails: {}".format(counter_rails, counter_siderails))