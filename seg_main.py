import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from utils import *
from tqdm import tqdm
from model.UNet import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epochs = 20
batch_size = 4
num_epoch_for_val = 5 # How often to compute val loss

train_dataset = ImageMaskDataset(dataset_type='train')
val_dataset = ImageMaskDataset(dataset_type='val')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = UNet(49).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=0.01,steps_per_epoch=len(train_dataloader),epochs=num_epochs)

def evaluate_model_on_val_set(model, val_dataloader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    return val_loss / len(val_dataloader)



for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_iou = 0.0

    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, data in progress_bar:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        #calculate iou
        probabilities = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probabilities, dim=1)
        # iou = iou_NHW(pred.cpu(), labels.cpu()).item()
        iou = 0
        running_iou += iou

        # upgrade progress bar
        average_loss = running_loss / (i + 1)
        average_iou = running_iou / (i + 1)
        progress_bar.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
        progress_bar.set_postfix(loss=average_loss, iou=average_iou)

    epoch_loss = running_loss / len(train_dataloader)
    epoch_iou = running_iou / len(train_dataloader)

    # validate every num_epoch_for_val epochs or at the last epoch
    if (epoch + 1) % num_epoch_for_val == 0 or (epoch + 1) == num_epochs:
        val_loss = evaluate_model_on_val_set(model, val_dataloader, device)
        print(f"Epoch {epoch + 1}/{num_epochs} finished, Train Loss: {epoch_loss:.4f}, Avg Train IoU: {epoch_iou:.4f}, Val Loss: {val_loss:.4f}")
    else:
        print(f"Epoch {epoch + 1}/{num_epochs} finished, Train Loss: {epoch_loss:.4f}, Avg Train IoU: {epoch_iou:.4f}")


torch.save(model.state_dict(), 'weight/unet_20.pth')
