from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torch.optim import Adadelta, SGD, Adam
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import time, copy, torch
from dataloader_onelabel import CustomDataset
from torch.utils.data import DataLoader

torch.set_num_threads(6)

PATH_MODELS = 'models'

def create_model(output_channels=1):
    model = models.segmentation.deeplabv3_resnet50(weight=True, progress=True)
    model.classifier = DeepLabHead(2048, output_channels)
    
    model.train()
    return model

def train(model, num_epochs, optimizer, criterion):
    start = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    loss = 0
    #device = "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Epoch
        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()
            else:
                model.eval()

            # Iterate over data
            dataset = CustomDataset(subset = phase)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            for inputs, masks in tqdm(dataloader):
                
                # load 1 data sample
                if device.type == 'cuda':
                    inputs, masks = inputs.cuda(), masks.cuda()
                else:
                    inputs, masks = inputs.cpu(), masks.cpu()

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs['out'], masks)

                if phase == 'Train':
                    loss.backward() # gradients
                    optimizer.step() # update parameters

            epoch_loss = loss

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            with open(os.path.join(PATH_MODELS, 'log_{}_{}.txt'.format(num_epochs, lr)), 'a') as log_file:
                log_file.write('Epoch {}: {} Loss: {:.4f}\n'.format(epoch, phase, epoch_loss))
            # save the better model
            if phase == 'Test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = copy.deepcopy(model.state_dict())

        print('Epoch {} done with loss: {:4f}'.format(epoch, epoch_loss))
        
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model)
    return model
    
if __name__ == "__main__":
    outputs = 2
    epochs = 5
    lr = 0.1
    model = create_model(outputs)

    loss_function = nn.CrossEntropyLoss()
    #optimizer = Adadelta(model.parameters(), lr = lr)
    optimizer = SGD(model.parameters(), lr = lr)
    #optimizer = Adam(model.parameters(), lr = lr)
    model_trained = train(model, epochs, optimizer, loss_function)

    torch.save(model_trained, os.path.join(PATH_MODELS,'model_{}_{}'.format(epochs, lr)))