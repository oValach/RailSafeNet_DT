import numpy as np
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor
import matplotlib.pyplot as plt
import cv2

rs19_label2bgr = {"buffer-stop": (70,70,70),
                "crossing": (128,64,128),
                "guard-rail": (0,255,0),
                "train-car" :  (100,80,0),
                "platform" : (232,35,244),
                "rail": (255,255,0),
                "switch-indicator": (127,255,0),
                "switch-left": (255,255,0),
                "switch-right": (127,127,0),
                "switch-unknown": (191,191,0),
                "switch-static": (0,255,127),
                "track-sign-front" : (0,220,220),
                "track-signal-front" : (30,170,250),
                "track-signal-back" : (0,85,125),
                #rail occluders
                "person-group" : (60,20,220),
                "car" : (142,0,0),
                "fence" : (153,153,190),
                "person" : (60,20,220),
                "pole" : (153,153,153),
                "rail-occluder" : (255,255,255),
                "truck" : (70,0,0)
                }

PATH_jpg = 'RailNet_DT/rs19_val/jpgs/rs19_val/rs00035.jpg'
PATH_mask = 'RailNet_DT/rs19_val/uint8/rs19_val/rs00035.png'
PATH_model = 'RailNet_DT/models/modelb_300_0.01_22_32_il.pth'


with open(PATH_jpg, "rb") as image_file, open(PATH_mask, "rb") as mask_file:

    # LOAD THE IMAGE
    im_jpg = cv2.imread(image_file.name)
    im_jpg = cv2.resize(im_jpg, (224, 224), interpolation=cv2.INTER_NEAREST)
    image = torch.tensor(im_jpg, dtype=torch.float32)
    image_norm = torch.div(image.permute(2, 0, 1), 254) # input normalization
    image_norm = image_norm.unsqueeze(0)
    
    # LOAD THE MASK
    id_map = cv2.imread(mask_file.name, cv2.IMREAD_GRAYSCALE)
    id_map = cv2.resize(id_map, (224, 224), interpolation=cv2.INTER_NEAREST)
    mask = torch.tensor(id_map, dtype=torch.float32).long()

    # LOAD THE MODEL
    model = torch.load(PATH_model, map_location=torch.device('cpu'))
    model, image_norm = model.cpu(), image_norm.cpu()
    model.eval()

    # INFERENCE + SOFTMAX
    output = model(image_norm)['out']
    confidence_scores = F.softmax(output, dim=1).cpu().detach().numpy().squeeze()
    #normalized_results = output.softmax(dim=1)
    id_map = np.argmax(confidence_scores, axis=0).astype(np.uint8)

    # Mask + prediction preparation
    mask = mask + 1
    mask[mask==256] = 0
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

    # Plot image and final mask
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap='gray')
    plt.title('Ground truth')

    plt.subplot(1, 2, 2) 
    plt.imshow(blend_sources, cmap='gray')
    plt.title('Output')
    plt.show()

    # CV2 VIZUALISATION
    image1 = rgba_blend
    image2 = rgba_mask

    initial_opacity1 = 0.05
    initial_opacity2 = 0.95

    cv2.namedWindow('Segmentation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Segmentation', 224, 224)

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

        cv2.imshow('Segmentation', blended_image)

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


#for o in range(21):
#    plt.figure()
#    plt.subplot(1, 2, 1)
#    plt.imshow(mask, cmap='gray')
#    plt.title('Ground truth')

#   plt.subplot(1, 2, 2) 
#    out = (output[0][o])#<0.5)
#    plt.imshow(out, cmap='gray')
#    plt.title('Output {}'.format(o))
#    plt.show()


#dataset = CustomDataset(subset = 'Val')
#dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
#outputs_test = []
#i = 1
#for inputs_test, masks in tqdm(dataloader):
#    outputs_test = model(inputs_test)
#    outputs_test = outputs_test['out'].detach().numpy()
#    np.save(os.path.join('models','output_{}'.format(i)), outputs_test)
#    i += 1