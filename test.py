import numpy as np
import pandas as pd
import torch
import cv2
import os
import time
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from mAP import compute_map_cls, compute_IoU, image_morpho
from rs19_val.example_vis import rs19_label2bgr

PATH_jpgs = 'RailNet_DT/rs19_val/jpgs/test'
PATH_jpg = 'RailNet_DT/rs19_val/jpgs/test/rs07700.jpg'
PATH_mask = 'RailNet_DT/rs19_val/uint8/test/rs07700.png'
PATH_masks = 'RailNet_DT/rs19_val/uint8/test'
PATH_model = 'RailNet_DT/models/modelchp_85_100_0.0002865237576874738_2_0.606629.pth'
#model_300_0.001_13_16_dd_adamw.pth, model_300_0.005_13_32_fp_adamw.pth, model_300_0.01_13_16_wh.pth
#modelchp_170_300_0.001_32_0.671144_aug.pth!, modelchp_105_200_0.001_32_0.725929_rf.pth, modelchp_185_200_0.001_32_0.788379_robustfire_noaug_480x480.pth

def load(filename, PATH_jpgs, path_model, input_size=[224,224], dataset_type='rs19val', item = None):
    #transform_resize = A.Compose([
    #                A.RandomResizedCrop(height=input_size[0], width=input_size[1], scale=(0.8, 1.0)),
    #                ])
    transform_img = A.Compose([
                    A.Resize(height=input_size[0], width=input_size[1], interpolation=cv2.INTER_NEAREST),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                    ToTensorV2(p=1.0),
                    ])
    transform_mask = A.Compose([
                    A.Resize(height=input_size[0], width=input_size[1], interpolation=cv2.INTER_NEAREST),
                    ToTensorV2(p=1.0),
                    ])  
    
    if dataset_type == 'pilsen':
        mask_pth = item[1][1]["masks"]["ground_truth"]["path"]
        mask_pth = os.path.join(PATH_jpgs, mask_pth)
    else:
        mask_pth = os.path.join(PATH_masks, filename).replace('.jpg', '.png')
    image = cv2.imread(os.path.join(PATH_jpgs, filename))
    mask = cv2.imread(mask_pth, cv2.IMREAD_GRAYSCALE)

    # LOAD THE IMAGE
    #im_jpg = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST)
    #image = torch.tensor(im_jpg, dtype=torch.float32)
    #image_norm = torch.div(image.permute(2, 0, 1), 254) # input normalization
    #image_norm = image_norm.unsqueeze(0)
    
    # LOAD THE MASK
    #id_map_gt = cv2.resize(mask_gr, (224, 224), interpolation=cv2.INTER_NEAREST)
    #mask = torch.tensor(id_map_gt, dtype=torch.float32).long()

    #transformed = transform_resize(image=image, mask=mask)
    #image = transformed['image']
    #mask = transformed['mask']
    
    image_tr = transform_img(image=image)['image']
    image_tr = image_tr.unsqueeze(0)
    image_vis = transform_mask(image=image)['image']
    mask = transform_mask(image=mask)['image']
    mask_id_map = np.array(mask.cpu().detach().numpy(), dtype=np.uint8)
    
    # LOAD THE MODEL
    model = torch.load(path_model, map_location=torch.device('cpu'))
    model, image_tr = model.cpu(), image_tr.cpu()
    model.eval()
    
    return image_tr, image_vis, mask, mask_id_map, model

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

def prepare_for_display(mask, image, id_map, rs19_label2bgr, image_size = [224,224]):
    # Mask + prediction preparation
    mask = mask + 1
    mask[mask==256] = 0
    mask = remap_ignored_clss(mask)
    mask = (mask + 100).detach().numpy().squeeze().astype(np.uint8)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Opacity channel addition to both mask and img
    alpha_channel = np.full((mask.shape[0], mask.shape[1]), 255, dtype=np.uint8)
    back_ids = mask==100
    alpha_channel[back_ids] = 0
    rgba_mask = cv2.merge((mask_rgb, alpha_channel))

    image = np.array(image.cpu().detach().numpy(), dtype=np.uint8)
    rgba_img = cv2.merge((image.transpose(1, 2, 0), alpha_channel))
    
    # Label colors + background
    rgbs = list(rs19_label2bgr.values())
    rgbs.append((255,255,255))

    blend_sources = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
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
    # Load two smaller images
    small_image1 = cv2.resize(image1, (300, 300), interpolation=cv2.INTER_NEAREST)
    small_image2 = cv2.resize(image2, (300, 300), interpolation=cv2.INTER_NEAREST)
        
    # Create a blank canvas for the combined visualization
    combined_image = np.zeros((600, 900, 4), dtype=np.uint8)  # Adjust the size as needed

    # Main loop for adjusting opacity and displaying the images
    cv2.namedWindow('{} | mAP:{:.3f} | MmAP:{:.3f} '.format(filename, map, Mmap), cv2.WINDOW_NORMAL)
    cv2.resizeWindow('{} | mAP:{:.3f} | MmAP:{:.3f} '.format(filename, map, Mmap), 900, 600)  # Adjust the size as needed

    while True:
        
        overlay_image = image1.copy()
        overlay_image[:, :, 3] = (image1[:, :, 3] * initial_opacity1).astype(np.uint8)

        alpha = (image2[:, :, 3] * initial_opacity2).astype(float)
        beta = 1.0 - alpha / 255.0

        blended_image = np.empty_like(overlay_image)
        blended_image[:, :, :3] = (overlay_image[:, :, :3] * alpha[:, :, np.newaxis] + image2[:, :, :3] * beta[:, :, np.newaxis]).astype(np.uint8)
        blended_image[:, :, 3] = (overlay_image[:, :, 3] * alpha + image2[:, :, 3] * beta).astype(np.uint8)

        blended_image = (image1 * initial_opacity1 + image2 * initial_opacity2).astype(np.uint8)

        blended_image_resized = cv2.resize(blended_image, (600, 600))  # Adjust the size as needed
        combined_image[:, :600, :] = blended_image_resized

        # Copy the smaller images to the right portion of the canvas
        combined_image[0:300, 600:900, :] = small_image1[:, :, :]
        combined_image[300:600, 600:900, :] = small_image2[:, :, :]

        cv2.imshow('{} | mAP:{:.3f} | MmAP:{:.3f} '.format(filename, map, Mmap), combined_image)

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

def stats_mean_and_reorder(classes_ap,classes_Map,classes_stats,classes_Mstats):
    for cls, value in classes_ap.items():
        classes_ap[cls] = np.divide(value[0], value[1])
    classes_ap['all']= np.mean(np.array(list(classes_ap.values())), axis=0)

    for cls, value in classes_Map.items():
        classes_Map[cls] = np.divide(value[0], value[1])
    classes_Map['all']= np.mean(np.array(list(classes_Map.values())), axis=0)

    for cls, value in classes_stats.items():
        classes_stats[cls] = np.divide(value[0], value[1])
    classes_stats['all']= np.mean(np.array(list(classes_stats.values()))[:, :4], axis=0)

    for cls, value in classes_Mstats.items():
        classes_Mstats[cls] = np.divide(value[0], value[1])
    classes_Mstats['all']= np.mean(np.array(list(classes_Mstats.values()))[:, :4], axis=0)

    for cls, value in classes_Mstats.items():
        classes_stats[cls] = np.insert(classes_stats[cls], 1, value[0])
        classes_stats[cls] = np.insert(classes_stats[cls], 3, value[1])
        classes_stats[cls] = np.insert(classes_stats[cls], 5, value[2])
        classes_stats[cls] = np.insert(classes_stats[cls], 7, value[3])

    return classes_ap,classes_Map,classes_stats,classes_Mstats

def process(model, input_img, mask, model_type):
    if model_type == "segformer":
        outputs = model(input_img) # segformer
    elif model_type == "deeplab":
        outputs = model(input_img)['out'] # deeplab resnet
    
    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=mask.shape[-2:],
        mode="bilinear",
        align_corners=False
    )
    
    output  = upsampled_logits.float()
        
    confidence_scores = F.softmax(output, dim=1).cpu().detach().numpy().squeeze()
    id_map = np.argmax(confidence_scores, axis=0).astype(np.uint8)
    id_map = image_morpho(id_map)
    
    return id_map

if __name__ == "__main__":
    mAPs,MmAPs,IoUs,MIoUs,accs,Maccs,precs,Mprecs,recs,Mrecs= list(),list(),list(),list(),list(),list(),list(),list(),list(),list()
    classes_ap,classes_Map,classes_stats,classes_Mstats = {},{},{},{}
    images_computed = 0
    
    for filename in os.listdir(PATH_jpgs):
        images_computed += 1
        
        vis = True
        to_break = False
        image_size = [1024,1024]
        
        if to_break:
            if images_computed > 50:
                break
        
        model_type = "segformer" #"deeplab"
        dataset_type = 'rs19val'
        #filename = 'rs07848.jpg'
        image_norm, image, mask, id_map_gt, model = load(filename, PATH_jpgs, PATH_model, image_size, dataset_type)

        # INFERENCE + SOFTMAX
        id_map = process(model, image_norm, mask, model_type)
        
        import matplotlib.pyplot as plt
        plt.imshow(id_map)
        plt.show()
        
        
        # mAP
        id_map_gt = remap_ignored_clss(id_map_gt)
        map,classes_ap  = compute_map_cls(id_map_gt, id_map, classes_ap)
        Mmap,classes_Map = compute_map_cls(id_map_gt, id_map, classes_Map, major = True)
        IoU,acc,prec,rec,classes_stats = compute_IoU(id_map_gt, id_map, classes_stats)
        MIoU,Macc,Mprec,Mrec,classes_Mstats = compute_IoU(id_map_gt, id_map, classes_Mstats, major=True)
        
        print('{} | mAP:{:.3f}/{:.3f} | IoU:{:.3f}/{:.3f} | prec:{:.3f}/{:.3f} | rec:{:.3f}/{:.3f} | acc:{:.3f}/{:.3f}'.format(filename,map,Mmap,IoU,MIoU,prec,Mprec,rec,Mrec,acc,Macc))
        mAPs.append(map)
        MmAPs.append(Mmap)
        IoUs.append(IoU)
        MIoUs.append(MIoU)
        accs.append(acc)
        Maccs.append(Macc)
        precs.append(prec)
        Mprecs.append(Mprec)
        recs.append(rec)
        Mrecs.append(Mrec)
        
        if vis:
            rgba_mask, rgba_blend, blend_sources = prepare_for_display(mask, image, id_map, rs19_label2bgr, image_size)        
            visualize(rgba_blend, rgba_mask)

    mAPs_avg, MmAPs_avg = np.nanmean(mAPs), np.nanmean(MmAPs)
    IoUs_avg, MIoUs_avg = np.nanmean(IoUs), np.nanmean(MIoUs)
    accs_avg, Maccs_avg = np.nanmean(accs), np.nanmean(Maccs)
    precs_avg, Mprecs_avg = np.nanmean(precs), np.nanmean(Mprecs)
    recs_avg, Mrecs_avg = np.nanmean(recs), np.nanmean(Mrecs)

    print('All         | mAP:{:.3f}/{:.3f} | IoU:{:.3f}/{:.3f} | prec:{:.3f}/{:.3f} | rec:{:.3f}/{:.3f} | acc:{:.3f}/{:.3f}'.format(mAPs_avg,MmAPs_avg,IoUs_avg,MIoUs_avg,precs_avg,Mprecs_avg,recs_avg,Mrecs_avg,accs_avg,Maccs_avg))
    print('mAP: {:.3f}-{:.3f} | MmAP: {:.3f}-{:.3f} | IoU: {:.3f}-{:.3f} | MIoU: {:.3f}-{:.3f}'.format(np.nanmin(mAPs), np.nanmax(mAPs), np.nanmin(MmAPs), np.nanmax(MmAPs),np.nanmin(IoUs), np.nanmax(IoUs), np.nanmin(MIoUs), np.nanmax(MIoUs)))

    classes_ap,classes_Map,classes_stats,classes_Mstats = stats_mean_and_reorder(classes_ap,classes_Map,classes_stats,classes_Mstats)

    df_ap = pd.DataFrame(list(classes_ap.items()), columns=['Class', 'mAP'])
    df_Map = pd.DataFrame(list(classes_Map.items()), columns=['Class', 'MmAP'])

    classes_stats_flat = [(key, *value) for key, value in classes_stats.items()]
    df_stats = pd.DataFrame(classes_stats_flat, columns=['Class','IoU','MIoU', 'acc','Macc', 'precision','Mprecision','recall','Mrecall'])

    df_merged = pd.merge(df_ap, df_Map, on='Class', how='outer')
    df_merged = pd.merge(df_merged, df_stats, on='Class', how='outer')

    print(df_merged)
