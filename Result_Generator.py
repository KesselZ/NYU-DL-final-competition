import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from utils import *
from model.ConvLSTM import *
from model.SimVP_classification import *
from model.SimVP2 import *
from torchmetrics import JaccardIndex
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_dim = 49
hidden_dim = 64
kernel_size = 3
num_layers = 2
output_dim = 3  # 替换为实际的类别数
batch_size = 4

train_dataset = MaskMaskDataset('Train')
val_dataset = MaskMaskDataset('Val')
hidden_dataset = MaskMaskDataset('hidden')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
hidden_dataloader = DataLoader(hidden_dataset, batch_size=batch_size, shuffle=False)

model = SimVP_Model((11, input_dim, 160, 240)).to(device)
saved_state_dict = torch.load('weight/SimVP_check_44.08.pth', map_location=device)
model.load_state_dict(saved_state_dict)

print(len(val_dataloader))


preds = []
with torch.no_grad():
    for i, (inputs, _) in enumerate(tqdm(hidden_dataloader, desc="Validating")):
        # draw_timestep_masks(inputs)
        # draw_timestep_masks(labels)

        model.eval()
        inputs = inputs.to(device)
        inputs_onehot = convert_to_one_hot(inputs)
        outputs = model(inputs_onehot)

        probabilities = torch.softmax(outputs, dim=2)
        pred = torch.argmax(probabilities, dim=2).cpu()
        pred = pred[:, -1, :, :]
        # draw_timestep_masks(pred.cpu().numpy())
        pred=filter_pred_by_input(pred, inputs[:, -1, :, :].cpu().long())
        jaccard = JaccardIndex(task="multiclass", num_classes=49).to(device)
        preds.append(pred)


result = torch.cat(preds, dim=0)
result = result.to(torch.uint8)
torch.save(result, 'result.pt')

print("getting val")
val_dataloader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
inputs, labels = next(iter(val_dataloader))
labels = labels[:, -1, :, :]

# 计算 Jaccard Index
print("Jaccard Index:")
with torch.no_grad():
    jaccard = JaccardIndex(task="multiclass", num_classes=49).to(device)
    print(jaccard(result.to(device), labels.to(device)))
