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
from model.SimVP_classification import *
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_dim = 49
batch_size = 4

train_dataset = MaskMaskDataset('Train+Unlabeled')
val_dataset = MaskMaskDataset('Val')
# train_dataset = ImageImageDataset('Train',mode="11to11")
# val_dataset = ImageImageDataset('Val',mode="11to11")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 获取训练数据集的第一个批次
train_inputs, train_labels = next(iter(train_dataloader))
print("Train DataLoader:")
print("Inputs Shape:", train_inputs.shape)
print("Labels Shape:", train_labels.shape)

# 获取验证数据集的第一个批次
val_inputs, val_labels = next(iter(val_dataloader))
print("\nValidation DataLoader:")
print("Inputs Shape:", val_inputs.shape)
print("Labels Shape:", val_labels.shape)
# plot_timestep_images(convert_mask_to_gray_image(val_inputs))
print(len(train_dataset))

print("START")
epochs = 100
num_epoch_for_val = 50

# input is T,channel,H,W
scaler = GradScaler()
model = SimVP((11, input_dim, 160, 240)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=1e-2,
                                                steps_per_epoch=len(train_dataloader),
                                                epochs=epochs)

def validate_model(model, val_dataloader, device):
    model.eval()
    total_val_loss = 0.0

    val_dataloader_tqdm = tqdm(val_dataloader, desc="Validation")
    with torch.no_grad():
        for inputs, labels in val_dataloader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = convert_to_one_hot(inputs)
            # inputs, labels = convert_mask_to_gray_image(inputs), convert_mask_to_gray_image(labels)

            outputs = model(inputs)
            outputs_permute = outputs.permute(0, 2, 1, 3, 4)
            loss = criterion(outputs_permute, labels.long())
            total_val_loss += loss.item()

            val_dataloader_tqdm.set_postfix(val_loss=loss.item())

    average_val_loss = total_val_loss / len(val_dataloader)
    val_dataloader_tqdm.close()

    return average_val_loss


train_loss = 0
# 训练循环
for epoch in range(epochs):
    model.train()
    total_train_loss = 0

    train_dataloader_tqdm = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{epochs}]")
    with autocast():
        for i, (inputs, labels) in enumerate(train_dataloader_tqdm):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = convert_to_one_hot(inputs)

            # inputs,labels=convert_mask_to_gray_image(inputs),convert_mask_to_gray_image(labels)
            optimizer.zero_grad()
            outputs = model(inputs)

            outputs_permute = outputs.permute(0, 2, 1, 3, 4)
            loss = criterion(outputs_permute, labels.long())
            train_loss = loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            train_dataloader_tqdm.set_postfix(train_loss=loss.item())

    average_train_loss = total_train_loss / len(train_dataloader)
    train_dataloader_tqdm.close()

    # 每 num_epoch_for_val 个 epoch 验证一次
    if (epoch + 1) % num_epoch_for_val == 0 or (epoch + 1) == epochs:
        average_val_loss = validate_model(model, val_dataloader, device)
        tqdm.write(
            f"Epoch Summary - Epoch [{epoch + 1}/{epochs}]: Train Loss: {average_train_loss}, Val Loss: {average_val_loss}")
    else:
        tqdm.write(
            f"Epoch Summary - Epoch [{epoch + 1}/{epochs}]: Train Loss: {average_train_loss}")

time.sleep(0.1)

torch.save(model.state_dict(), 'SimVP_checkpoint.pth')
checkpoint_save(epochs, model, optimizer, scheduler)
print("done")
