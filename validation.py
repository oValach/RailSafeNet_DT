import numpy as np
import pandas as pd
import torch
import cv2
import os
import torch.nn.functional as F
from mAP import compute_map_cls
from rs19_val.example_vis import rs19_label2bgr

PATH_jpgs = 'RailNet_DT/rs19_val/jpgs/test'
PATH_jpg = 'RailNet_DT/rs19_val/jpgs/test/rs07700.jpg'
PATH_mask = 'RailNet_DT/rs19_val/uint8/test/rs07700.png'
PATH_masks = 'RailNet_DT/rs19_val/uint8/test'
PATH_model = 'RailNet_DT/models/model_300_0.01_13_16_wh.pth'

def load(filename):
    im_jpg = cv2.imread(os.path.join(PATH_jpgs, filename))
    mask_pth = os.path.join(PATH_masks, filename).replace('.jpg', '.png')
    mask_gr = cv2.imread(mask_pth, cv2.IMREAD_GRAYSCALE)

    # LOAD THE IMAGE
    im_jpg = cv2.resize(im_jpg, (224, 224), interpolation=cv2.INTER_NEAREST)
    image = torch.tensor(im_jpg, dtype=torch.float32)
    image_norm = torch.div(image.permute(2, 0, 1), 254) # input normalization
    image_norm = image_norm.unsqueeze(0)
    
    # LOAD THE MASK
    id_map_gt = cv2.resize(mask_gr, (224, 224), interpolation=cv2.INTER_NEAREST)
    mask = torch.tensor(id_map_gt, dtype=torch.float32).long()

    # LOAD THE MODEL
    model = torch.load(PATH_model, map_location=torch.device('cpu'))
    model, image_norm = model.cpu(), image_norm.cpu()
    model.eval()
    
    return image_norm, image, mask, id_map_gt, model

def remap_ignored_clss(id_map):
    ignore_list = [0,1,2,6,8,9,15,16,19,20]
    for cls in ignore_list:
        id_map[id_map==cls] = 255

    ignore_set = set(ignore_list)
    cls_remaining = [num for num in range(0, 22) if num not in ignore_set]

    # renumber the remaining classes 0-number of remaining classes
    for idx, cls in enumerate(cls_remaining):
        id_map[id_map==cls] = idx

    id_map[id_map==255] = 12 # background
    
    return id_map

def prepare_for_display(mask, image, id_map, rs19_label2bgr):
    # Mask + prediction preparation
    mask = mask + 1
    mask[mask==256] = 0
    mask = remap_ignored_clss(mask)
    mask = (mask + 100).detach().numpy().astype(np.uint8)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Opacity channel addition to both mask and img
    alpha_channel = np.full((mask.shape[0], mask.shape[1]), 255, dtype=np.uint8)
    back_ids = mask==100
    alpha_channel[back_ids] = 0
    rgba_mask = cv2.merge((mask_rgb, alpha_channel))

    image = np.array(image.cpu().detach().numpy(), dtype=np.uint8)
    rgba_img = cv2.merge((image, alpha_channel))
    
    # Label colors + background
    rgbs = list(rs19_label2bgr.values())
    rgbs.append((255,255,255))

    blend_sources = np.zeros((224, 224, 3), dtype=np.uint8)
    for class_id in range(21):
        class_pixels = id_map == class_id
        rgb_color = np.array(rgbs[class_id])

        for i in range(3):
            blend_sources[:,:,i] = blend_sources[:,:,i] + (rgb_color[i] * class_pixels).astype(np.uint8)

    # Opacity channel for the rgb class mask and merge with input mask
    alpha_channel_blend = np.full((blend_sources.shape[0], blend_sources.shape[1]), 150, dtype=np.uint8)
    rgba_blend = cv2.merge((blend_sources , alpha_channel_blend))
    blend_sources = (rgba_blend * 0.1 + rgba_img * 0.9).astype(np.uint8)
    
    return(rgba_mask, rgba_blend, blend_sources)
    
def visualize(rgba_blend, rgba_mask):
    # CV2 VIZUALISATION
    image1 = rgba_blend
    image2 = rgba_mask

    initial_opacity1 = 0.05
    initial_opacity2 = 0.95

    cv2.namedWindow('{} | mAP:{:.3f} | MmAP:{:.3f} '.format(filename, map, Mmap), cv2.WINDOW_NORMAL)
    cv2.resizeWindow('{} | mAP:{:.3f} | MmAP:{:.3f} '.format(filename, map, Mmap), 1000, 1000)

    while True:
        overlay_image = image1.copy()
        overlay_image[:, :, 3] = (image1[:, :, 3] * initial_opacity1).astype(np.uint8)

        # Blend the modified image with image2 using the specified opacity
        alpha = (image2[:, :, 3] * initial_opacity2).astype(float)
        beta = 1.0 - alpha / 255.0

        # Perform element-wise multiplication with reshaped alpha
        blended_image = np.empty_like(overlay_image)
        blended_image[:, :, :3] = (overlay_image[:, :, :3] * alpha[:, :, np.newaxis] + image2[:, :, :3] * beta[:, :, np.newaxis]).astype(np.uint8)
        blended_image[:, :, 3] = (overlay_image[:, :, 3] * alpha + image2[:, :, 3] * beta).astype(np.uint8)

        blended_image = (image1 * initial_opacity1 + image2 * initial_opacity2).astype(np.uint8)

        cv2.imshow('{} | mAP:{:.3f} | MmAP:{:.3f} '.format(filename, map, Mmap), blended_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            initial_opacity1 += 0.1
            initial_opacity1 = min(initial_opacity1, 1.0)
        elif key == ord('s'):
            initial_opacity1 -= 0.1
            initial_opacity1 = max(initial_opacity1, 0.0)
        elif key == ord('z'):
            initial_opacity2 += 0.1
            initial_opacity2 = min(initial_opacity2, 1.0)
        elif key == ord('x'):
            initial_opacity2 -= 0.1
            initial_opacity2 = max(initial_opacity2, 0.0)

    cv2.destroyAllWindows()

mAPs = list()
MmAPs = list()
classes_ap = {}
classes_Map = {}
counter = 0
for filename in os.listdir(PATH_jpgs):
    counter += 1
    
    image_norm, image, mask, id_map_gt, model = load(filename)

    # INFERENCE + SOFTMAX
    output = model(image_norm)['out']
    confidence_scores = F.softmax(output, dim=1).cpu().detach().numpy().squeeze()
    id_map = np.argmax(confidence_scores, axis=0).astype(np.uint8)

    # mAP
    id_map_gt = remap_ignored_clss(id_map_gt)
    map, classes_ap  = compute_map_cls(id_map_gt, id_map, classes_ap)
    Mmap, classes_Map = compute_map_cls(id_map_gt, id_map, classes_Map, major = True)
    
    print('{} | mAP:{:.3f} | MmAP:{:.3f} '.format(filename, map, Mmap))
    mAPs.append(map)
    MmAPs.append(Mmap)
    
    #if counter > 100:
    #    break
    
    vis = False
    if vis:
        rgba_mask, rgba_blend, blend_sources = prepare_for_display(mask, image, id_map, rs19_label2bgr)        
        visualize(rgba_blend, rgba_mask)

mAPs_avg = np.nanmean(mAPs)
MmAPs_avg = np.nanmean(MmAPs)
print('All | mAP: {:.3f} | MmAP: {:.3f}'.format(mAPs_avg, MmAPs_avg))

for cls, value in classes_ap.items():
    classes_ap[cls] = value[0] / value[1]
    
for cls, value in classes_Map.items():
    classes_Map[cls] = value[0] / value[1]

df_ap = pd.DataFrame(list(classes_ap.items()), columns=['Class', 'mAP'])
df_Map = pd.DataFrame(list(classes_Map.items()), columns=['Class', 'MmAP'])
df_merged = pd.merge(df_ap, df_Map, on='Class', how='outer')
print(df_merged)
