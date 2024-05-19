from scripts.dataloader_RailSem19 import CustomDataset
from scripts.metrics_filtered_cls import compute_map_cls, compute_IoU
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torch.optim import SGD, Adam, Adagrad
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torchsummary import summary
import torch.nn as nn
import albumentations as A
import torch
import numpy as np
import os
import cv2
import wandb
from tqdm import tqdm
import time
import copy

def get_image_4_wandb(path, input_size = [224,224]):
    transform_img = A.Compose([
                    A.Resize(height=input_size[0], width=input_size[1], interpolation=cv2.INTER_NEAREST),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                    A.pytorch.ToTensorV2(p=1.0),
                    ])
    image = cv2.imread(next((os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))), None))
    image = transform_img(image=image)['image']
    image = image.unsqueeze(0)
    image = image.cpu()
    
    return image

def wandb_init(num_epochs, lr, batch_size, outputs, optimizer, scheduler):
    wandb.init(
        project="DP_train_full",
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "outputs": outputs,
            "optimizer": optimizer,
            "scheduler": scheduler,
        }
    )

LIGHT = True
WANDB = False

if not LIGHT:
    PATH_JPGS = "RailNet_DT/rs19_val/jpgs/rs19_val"
    PATH_MASKS = "RailNet_DT/rs19_val/uint8/rs19_val"
else:
    PATH_JPGS = "RailNet_DT/rs19_val_light/jpgs/rs19_val"
    PATH_MASKS = "RailNet_DT/rs19_val_light/uint8/rs19_val"

PATH_MODELS = "RailNet_DT/models"
PATH_LOGS = "RailNet_DT/logs"

def create_model(output_channels=1):
    model = models.segmentation.deeplabv3_resnet50(weight=True, progress=True)

    model.classifier = DeepLabHead(2048, output_channels)
    model.train()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model

def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.train()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model

def train(model, num_epochs, batch_size, image_size, optimizer, criterion):
    start = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    loss = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        print('-' * 20)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        
        # Epoch
        train_loss = 0 # for the wandb logging
        val_loss = 0 # --||--
        val_MmAP, val_mAP, val_IoU, val_MIoU = list(), list(), list(), list()
        classes_MAP, classes_AP, classes_IoU, classes_MIoU = {},{},{},{}
        
        dl_lentrain = 0
        dl_lenval = 0
        
        for phase in ['Train', 'Valid']:

            dataset = CustomDataset(PATH_JPGS, PATH_MASKS, image_size, subset=phase, val_fraction=0.5)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            
            if phase == 'Train':
                model.train()
                dl_lentrain = len(dataloader)
                
                for inputs, masks in tqdm(dataloader):
                    inputs = inputs.to(device)
                    masks = masks.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    outputs = model(inputs)
                    outputs = outputs['out']
                    loss = criterion(torch.squeeze(outputs), masks)

                    loss.backward()  # gradients
                    optimizer.step()  # update parameters

                    train_loss += loss
                    
            elif phase == 'Valid':
                model.eval()
                dl_lenval = len(dataloader)
                with torch.no_grad():
                    for inputs, masks in tqdm(dataloader):
                        inputs = inputs.to(device)
                        masks = masks.to(device)

                        outputs = model(inputs)['out']
                        loss = criterion(torch.squeeze(outputs), masks)

                        val_loss += loss
                        predicted_masks = outputs
                        gt_masks = masks.cpu().detach().numpy().squeeze()
                        for prediction, gt in zip(predicted_masks, gt_masks):
                            prediction = F.softmax(prediction, dim=0).cpu().detach().numpy().squeeze()
                            prediction = np.argmax(prediction, axis=0).astype(np.uint8)
                            
                            mAP,classes_AP = compute_map_cls(gt, prediction, classes_AP)
                            Mmap,classes_MAP = compute_map_cls(gt, prediction, classes_MAP, major = True)
                            IoU,_,_,_,classes_IoU = compute_IoU(gt, prediction, classes_IoU)
                            MIoU,_,_,_,classes_MIoU = compute_IoU(gt, prediction, classes_MIoU, major=True)
                            val_mAP.append(mAP)
                            val_MmAP.append(Mmap)
                            val_IoU.append(IoU)
                            val_MIoU.append(MIoU)

        # Compute the epoch mAP and IoU
        val_MmAP, val_mAP = np.nanmean(val_MmAP), np.nanmean(val_mAP)
        val_MIoU, val_IoU = np.nanmean(val_MIoU), np.nanmean(val_IoU)
        for cls, value in classes_MAP.items():
            classes_MAP[cls] = np.divide(value[0], value[1])
        classes_MmAP_all= np.mean(np.array(list(classes_MAP.values())), axis=0)
        for cls, value in classes_AP.items():
            classes_AP[cls] = np.divide(value[0], value[1])
        classes_mAP_all= np.mean(np.array(list(classes_AP.values())), axis=0)
        for cls, value in classes_IoU.items():
            classes_IoU[cls] = np.divide(value[0], value[1])
        classes_IoU_all= np.mean(np.array(list(classes_IoU.values()))[:, :4], axis=0)
        for cls, value in classes_MIoU.items():
            classes_MIoU[cls] = np.divide(value[0], value[1])
        classes_MIoU_all= np.mean(np.array(list(classes_MIoU.values()))[:, :4], axis=0)

        # LROnPlateau
        # scheduler.step(classes_MIoU_all[0])
        # current_lr = scheduler._last_lr[0]
        # linearLRdecay
        if epoch > 200:
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print epoch summary
        print('Epoch {}/{}: Train loss: {:.4f} | Val loss: {:.4f} | lr: {:.4f} | mAP: {:.4f} | MmAP: {:.4f} | IoU: {:.4f} | MIoU: {:.4f}'.format(epoch + 1,num_epochs,train_loss/dl_lentrain,val_loss/dl_lenval,current_lr,classes_mAP_all,classes_MmAP_all,classes_IoU_all[0],classes_MIoU_all[0]))
        
        with open(os.path.join(PATH_LOGS, 'log_{}_{}.txt'.format(num_epochs, lr)), 'a') as log_file:
            log_file.write('Epoch {}/{}: Train loss: {:.4f} | Val loss: {:.4f} | lr: {:.4f} | mAP: {:.4f} | MmAP: {:.4f} | IoU: {:.4f} | MIoU: {:.4f}'.format(epoch + 1,num_epochs,train_loss/dl_lentrain,val_loss/dl_lenval,current_lr,classes_mAP_all,classes_MmAP_all,classes_IoU_all[0],classes_MIoU_all[0]))

        # Save model checkpoint every 5 epochs
        if epoch > 1 and epoch % 5 == 0 and phase == 'Valid':
            torch.save(model, os.path.join(PATH_MODELS,'modelchp_{}_{}_{}_{}_{:3f}.pth'.format(epoch, epochs, lr, batch_size,classes_MIoU_all[0])))
            print('Saving checkpoint for epoch {} as: modelchp_{}_{}_{}_{}_{:3f}.pth'.format(epoch, epoch, epochs, lr, batch_size,classes_MIoU_all[0]))

        # Save the best model based on validation loss
        if phase == 'Valid' and (val_loss/dl_lenval) < best_loss:
            best_loss = (val_loss/dl_lenval)
            best_model = copy.deepcopy(model.state_dict())
            print('Saving model for epoch {} as the best so far: modelb_{}_{}_{}_{}_{:3f}.pth'.format(epoch, epoch, epochs, lr, batch_size,classes_MIoU_all[0]))
            
        if WANDB:
            normalized_results = outputs[0].softmax(dim=0).cpu().detach().numpy().squeeze()
            id_map = np.argmax(normalized_results, axis=0).astype(np.uint8)
            id_map = np.divide(id_map,np.max(id_map))

            im_classes = []
            for class_id in range(outs-1):
                im_classes.append(wandb.Image((outputs[0][class_id].cpu()).detach().numpy(), caption="Prediction of a class {}".format(class_id+1)))
            im_classes.append(wandb.Image((outputs[0][-1].cpu()).detach().numpy(), caption="Background"))
            id_log = wandb.Image(id_map, caption="Predicted ID map")

            mask_log = masks[0].cpu().detach().numpy() + 1
            mask_log[mask_log==256] = 0
            mask_log = (mask_log / outs)
            mask_log = wandb.Image(mask_log, caption="Input mask")

            wandb.log({
                "train_loss" : train_loss,
                "val_loss" : val_loss,
                "lr" : current_lr,
                "mAP" : classes_mAP_all,
                "MmAP" : classes_MmAP_all,
                "IoU" : classes_IoU_all[0],
                "MIoU" : classes_MIoU_all[0],
                "Input, predicted mask, background" : [mask_log, id_log, im_classes[-1]],
                "Classes": im_classes[0:-2]
                })

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    final_model = model
    model.load_state_dict(best_model)
    return final_model, model

if __name__ == "__main__":
    epochs = 500
    lr = 0.001
    batch_size = 4
    outs = 13
    image_size = [224,224]
    
    model = create_model(outs)
    #model = load_model('RailNet_DT/models/modelchp_105_300_0.001_32.pth')

    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True,
                                                #threshold=0.005, threshold_mode='abs')
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

    if WANDB:
        wandb_init(epochs, lr, batch_size, outs, str(optimizer.__class__), str(scheduler.__class__))

    model_final, best_model = train(model, epochs, batch_size, image_size, optimizer, loss_function)

    torch.save(model_final, os.path.join(PATH_MODELS, 'model_{}_{}_{}_{}.pth'.format(epochs, lr, outs, batch_size)))
    torch.save(best_model, os.path.join(PATH_MODELS, 'modelb_{}_{}_{}_{}.pth'.format(epochs, lr, outs, batch_size)))
    print('Saved as: model_{}_{}_{}_{}.pth'.format(epochs, lr, outs, batch_size))
    if WANDB:
        wandb.finish()
