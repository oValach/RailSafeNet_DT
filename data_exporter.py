import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import imageio
from rs19_val.example_vis import config_to_rgb

rs19_label2bgr = {"buffer-stop": (70,70,70),
                "crossing": (128,64,128),
                "guard-rail": (0,254,0),
                "train-car" :  (100,80,0),
                "platform" : (232,35,244),
                "rail": (254,254,0),
                "switch-indicator": (127,254,0),
                "switch-left": (254,254,0),
                "switch-right": (127,127,0),
                "switch-unknown": (191,191,0),
                "switch-static": (0,254,127),
                "track-sign-front" : (0,220,220),
                "track-signal-front" : (30,170,250),
                "track-signal-back" : (0,85,125),
                #rail occluders
                "person-group" : (60,20,220),
                "car" : (142,0,0),
                "fence" : (153,153,190),
                "person" : (60,20,220),
                "pole" : (153,153,153),
                "rail-occluder" : (254,254,254),
                "truck" : (70,0,0)
                }

PATH_UINT = "rs19_val/uint8/rs19_val"
PATH_CONFIG = "rs19_val/rs19-config.json"
PATH_JPGS = "rs19_val/jpgs/rs19_val"
PATH_MASKS = "rs19_val/masks"
PATH_OBJECTS = "rs19_val/masks/rails"

def get_color_map(cur_dir):
    im_id_map = cv2.imread(cur_dir,cv2.IMREAD_GRAYSCALE) #get semantic label map
    im_id_col = np.zeros((im_id_map.shape[0], im_id_map.shape[1], 3), np.uint8)
    lut_bgr = config_to_rgb(PATH_CONFIG, default_col = [255,255,255])[::-1] #swap color channels as opencv uses BGR
    for c in range(3):
        im_id_col[:,:,c] = lut_bgr[c][im_id_map] #apply color coding    

    #plt.imshow(im_id_col)
    #plt.show()
    return im_id_col

def export_segmented_labels(file_idx):
    file_path = os.path.join(PATH_UINT, f"rs{str(file_idx).zfill(5)}.png")

    label_colors = np.array(list(rs19_label2bgr.values()))
    color_map = get_color_map(file_path)
    reshaped_image = color_map.reshape(-1, 3)
    reshaped_channels = np.array(label_colors).reshape(-1, 3)

    distances = np.linalg.norm(reshaped_image[:, np.newaxis] - reshaped_channels, axis=2)
    lowest_distances_indices = np.argmin(distances, axis=1)
    filtered_indices = np.where(distances[np.arange(distances.shape[0]), lowest_distances_indices] == 0, lowest_distances_indices, -1)
    filtered_indices = filtered_indices.reshape(1080, 1920)
    
    #np.savez_compressed(os.path.join(PATH_MASKS,str(i)+'.npz'), filtered_indices)

    for label_idx in range(21):
        if label_idx == 5:
            filtered_indicess = filtered_indices == label_idx
            filtered_label = np.where(filtered_indicess, 0, 254).astype(np.uint8)
            imageio.imwrite(os.path.join(PATH_OBJECTS,'rs0'+str(file_idx)+'.png'), filtered_label)

    #jpg_path = file_path.replace('uint8', 'jpgs')
    #jpg_path = jpg_path.replace('png', 'jpg')

    #jpg_file = cv2.imread(jpg_path)

    #return jpg_file, filtered_indices

def get_label(label_keys, channel_index):
    channel_label = label_keys[channel_index] if channel_index != -1 else None
    return channel_label

if __name__ == "__main__":
    #loaded_data = np.load(os.path.join(PATH_MASKS,str(0)+'.npz'))['arr_0'] #ulozeno v arr_*cislo masky*.npz
    for i in range(8500):
        print(i)
        path = os.path.join(PATH_OBJECTS,'rs0',str(i)+'.png')
        if os.path.exists(os.path.join(PATH_OBJECTS,'rs0',str(i)+'.png')):
            continue
        if i == 4337: # corrupted file
            continue
        else:
            export_segmented_labels(i)