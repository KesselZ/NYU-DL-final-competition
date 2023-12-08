import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from utils import *
from model.SimVP_classification import *
from torchmetrics import JaccardIndex

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_dim = 49
batch_size = 1

train_dataset = MaskMaskDataset('Train')
val_dataset = MaskMaskDataset('Val')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


model = SimVP((11, input_dim, 160, 240)).to(device)
saved_state_dict = torch.load('weight/SimVP_check_42.98.pth', map_location=device)
model.load_state_dict(saved_state_dict)

with torch.no_grad():
    for i, (inputs, labels) in enumerate(val_dataloader):
        draw_timestep_masks(inputs)
        draw_timestep_masks(labels)

        model.eval()
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = convert_to_one_hot(inputs)
        outputs = model(inputs)

        probabilities = torch.softmax(outputs, dim=2)
        pred= torch.argmax(probabilities, dim=2).cpu()

        draw_timestep_masks(pred.cpu().numpy())

        jaccard = JaccardIndex(task="multiclass", num_classes=49).to(device)
        print(jaccard(pred.to(device), labels.to(device)))
        print(torch.unique(pred),torch.unique(labels))
        break