from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torch.optim import Adadelta, SGD, Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import os
import wandb
from tqdm import tqdm
import time
import copy
from dataloader import CustomDataset

# torch.set_num_threads(6)

def wandb_init(num_epochs, lr, batch_size, outputs, optimizer):
    wandb.init(
        project="DP_train_full",
        # name=f"experiment_{run}",
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "outputs": outputs,
            "optimizer": optimizer,
        }
    )

LIGHT = True
WANDB = True

if not LIGHT:
    PATH_JPGS = "DP_work/rs19_val/jpgs/rs19_val"
    PATH_MASKS = "DP_work/rs19_val/uint8/rs19_val"  # /rails
    PATH_MODELS = "DP_work/models"
    PATH_LOGS = "DP_work/logs"
else:
    PATH_JPGS = "DP_work/rs19_val_light/jpgs/rs19_val"
    PATH_MASKS = "DP_work/rs19_val_light/uint8/rs19_val"
    PATH_MODELS = "DP_work/models"
    PATH_LOGS = "DP_work/logs"


def create_model(output_channels=1):
    model = models.segmentation.deeplabv3_resnet50(weight=True, progress=True)
    model.classifier = DeepLabHead(2048, output_channels)

    model.train()
    return model


def train(model, num_epochs, batch_size, optimizer, criterion):
    start = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    loss = 0
    #device = "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        print('-' * 20)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        
        # Epoch
        train_loss = 0 # for the wandb logging
        test_loss = 0 # --||--
        outputs_c = None
        for phase in ['Train', 'Test']:

            if phase == 'Train':
                model.train()
            else:
                model.eval()

            # Iterate over data
            dataset = CustomDataset(PATH_JPGS, PATH_MASKS, seed=True, subset=phase, test_val_fraction=0.1)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
            for inputs, masks in tqdm(dataloader):

                if device == 'cpu':
                    inputs, masks = inputs.cpu(), masks.cpu()
                else:
                    inputs, masks = inputs.cuda(), masks.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs_c = model(inputs)
                loss = criterion(torch.squeeze(outputs_c['out']), masks)

                if phase == 'Train':
                    train_loss = loss
                    loss.backward()  # gradients
                    optimizer.step()  # update parameters
                else:
                    test_loss = loss

            epoch_loss = loss

            print('{} Loss: {:.4f}'.format(phase, loss))

            with open(os.path.join(PATH_LOGS, 'log_{}_{}.txt'.format(num_epochs, lr)), 'a') as log_file:
                log_file.write('Epoch {}: {} Loss: {:.4f}\n'.format(epoch, phase, epoch_loss))

            # save the better model
            if phase == 'Test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = copy.deepcopy(model.state_dict())

        if WANDB:
            im_classes = []
            #for class_id in range(outputs):
            im_class = wandb.Image((outputs_c["out"][0][0].cpu()).detach().numpy(), caption="Prediction of a class 0")
            
            wandb.log({
                "train_loss" : train_loss,
                "test_loss" : test_loss,
                "Class 1": im_class
                })

        print('Epoch {} done with loss: {:4f}'.format(epoch, epoch_loss))

    time_elapsed = time.time() - start

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    final_model = model
    model.load_state_dict(best_model)

    return final_model, model


if __name__ == "__main__":
    epochs = 5
    lr = 0.01
    batch_size = 4
    outputs = 21
    model = create_model(outputs)

    #optimizer = Adadelta(model.parameters(), lr = lr)
    optimizer = Adam(model.parameters(), lr=lr) #zkusit adamw?
    #optimizer = SGD(model.parameters(), lr = lr)

    loss_function = nn.CrossEntropyLoss(ignore_index=255) # ignore not classified pixels

    if WANDB:
        wandb_init(epochs, lr, batch_size, outputs, str(optimizer.__class__))

    model_final, best_model = train(model, epochs, batch_size, optimizer, loss_function)

    torch.save(model_final, os.path.join(PATH_MODELS, 'model_{}_{}_{}.pth'.format(epochs, lr, outputs)))
    torch.save(best_model, os.path.join(PATH_MODELS, 'modelb_{}_{}_{}.pth'.format(epochs, lr, outputs)))
    print('Saved as: model_{}_{}_{}.pth'.format(epochs, lr, outputs))
    if WANDB:
        wandb.finish()
