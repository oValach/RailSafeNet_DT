import numpy as np
from sklearn.metrics import average_precision_score
from skimage import morphology

def image_morpho(mask_prediction):
    selem2 = morphology.disk(2)
    closed = morphology.closing(mask_prediction, selem2)
    
    return closed

def remap_mask(mask1, mask2): # not used
    classes_mask1 = np.unique(mask1)
    classes_mask2 = np.unique(mask2)

    # mapping between class indices in mask2 to mask1
    class_mapping = {}
    for class1 in classes_mask1:
        intersection = list()
        for class2 in classes_mask2:
            intersection.append((np.sum((mask1 == class1) & (mask2 == class2)), class1, class2))
            
        most_sim_cls = max(intersection, key=lambda x: x[0])
        class_mapping[class1] = most_sim_cls[2]

    # remapping the prediction mask
    for class_old, class_new in class_mapping.items():
        mask2[mask2==class_new] = class_old

    return mask1, mask2

def compute_ap_for_cls(gt_mask, pred_mask, cls_id):
    # flatten to binary
    gt_flat = (gt_mask.flatten() == cls_id).astype(int)
    pred_flat = (pred_mask.flatten() == cls_id).astype(int)

    #if there is no occurences return 0 for the class
    if np.sum(gt_flat) == 0:
        return 0
    else:
        ap = average_precision_score(gt_flat, pred_flat)
        return ap

def compute_map_cls(gt_mask, pred_mask, classes_ap, major = False, treshold=150):
    
    if major: # compute mAP just from the major classes in the image (more than 150 pixels in each mask)
        classes = get_major_classes(gt_mask, pred_mask, treshold)
    else:
        classes = np.unique(np.concatenate((np.unique(gt_mask),np.unique(pred_mask))))
        classes = classes[np.isin(classes, np.unique(gt_mask)) & np.isin(classes, np.unique(pred_mask))]
    
    if np.all(classes==21):
        return 0, classes_ap
    
    # compute the AP for individual classes
    ap_values = []
    dict_ap_values = {}
    for class_index in classes:
        if class_index != 21: # exclude background class 21
            ap = compute_ap_for_cls(gt_mask, pred_mask, class_index)
            ap_values.append(ap) # save for per picture evaluation
            dict_ap_values[class_index] = ap # save for per class evaluation

    # add the values for the per class evaluation
    for cls, value in dict_ap_values.items():
        if cls not in classes_ap:
            classes_ap[cls] = [value,1]
        else:
            classes_ap[cls] = np.add(classes_ap[cls], [value,1])

    return np.mean(ap_values), classes_ap


def get_major_classes(gt_mask,pred_mask,treshold):
    classes = np.unique(np.concatenate((np.unique(gt_mask),np.unique(pred_mask))))
    
    occurrences1 = {num: np.count_nonzero(gt_mask == num) for num in np.unique(gt_mask)}
    occurrences2 = {num: np.count_nonzero(pred_mask == num) for num in np.unique(pred_mask)}
    
    classes = classes[[i for i, key in enumerate(classes) if occurrences1.get(key, 0) > treshold and occurrences2.get(key, 0) > treshold]]
    
    return classes

def compute_IoU(gt_mask, pred_mask, classes_stats, major=False, treshold=144):
    if major: # compute mAP just from the major classes in the image (more than 144 pixels in each mask)
        classes = get_major_classes(gt_mask, pred_mask, treshold)
    else:
        classes = np.unique(np.concatenate((np.unique(gt_mask),np.unique(pred_mask))))
        classes = classes[np.isin(classes, np.unique(gt_mask)) & np.isin(classes, np.unique(pred_mask))]

    if np.all(classes==21):
        return(0, 0, 0, 0, classes_stats)
    
    stats_image = {}
    
    for cls in classes:
        if cls != 21: # excluding background
            
            intersection = np.sum((gt_mask == cls) & (pred_mask == cls))
            union = np.sum((gt_mask == cls) | (pred_mask == cls))
            IoU = intersection / union
            
            tp = np.sum((gt_mask == cls) & (pred_mask == cls))
            tn = np.sum((gt_mask != cls) & (pred_mask != cls))
            fp = np.sum((gt_mask != cls) & (pred_mask == cls))
            fn = np.sum((gt_mask == cls) & (pred_mask != cls))
            
            acc = (tp+tn)/(tp+tn+fp+fn)
            if tp != 0:
                precision = tp/(tp+fp)
                recall = tp/(tp+fn)
            else:
                precision = 0
                recall = 0
                
            stats_image[cls] = [IoU,acc,precision,recall]
    
    for cls, value in stats_image.items():
        if cls not in classes_stats:
            classes_stats[cls] = [value,1]
        else:
            classes_stats[cls][0] = np.add(classes_stats[cls][0], value)
            classes_stats[cls][1] = np.add(classes_stats[cls][1], 1)
    
    stats = list(stats_image.values())
    stats = np.array(stats).reshape(-1, len(stats[0]))

    IoU_img = np.mean(stats[:,0])
    acc_img = np.mean(stats[:,1])
    precision_img = np.mean(stats[:,2])
    recall_img = np.mean(stats[:,3])
    
    return(IoU_img, acc_img, precision_img, recall_img, classes_stats)
