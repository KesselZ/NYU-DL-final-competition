from torch.utils.data import DataLoader
from utils import *
from model.UNet import *
import torch
import torchmetrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1

train_dataset = ImageMaskDataset(dataset_type='train')
val_dataset = ImageMaskDataset(dataset_type='val')
unlabel_dataset = ImageMaskDataset(dataset_type='unlabeled')
hidden_dataset = ImageMaskDataset(dataset_type='hidden')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
unlabel_dataloader = DataLoader(unlabel_dataset, batch_size=batch_size, shuffle=False)
hidden_dataloader = DataLoader(hidden_dataset, batch_size=batch_size, shuffle=False)

# print(len(val_dataloader))
# print(len(train_dataloader))
# print(len(unlabel_dataloader))

model = UNet(49).to(device)
saved_state_dict =  torch.load('weight/unet_20.pth', map_location=device)
model.load_state_dict(saved_state_dict)

total_iou = 0.0
total_iou2 = 0.0
with torch.no_grad():  # 在评估过程中不计算梯度
    for inputs, labels in hidden_dataloader:
        # draw_label_masks(labels.cpu())
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)

        # 选择概率最高的类别作为最终预测
        pred = torch.argmax(probabilities, dim=1).cpu()
        pred_remove=remove_rare_pixels(pred)

        batch_iou = iou_NHW(pred, labels,ignore_index=None).to(device)
        jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49).to(device)
        batch_iou2=jaccard(pred.to(device), labels)

        draw_label_masks(labels.cpu())
        draw_label_masks(pred)


        # draw_label_masks(pred)
        total_iou += batch_iou
        total_iou2 += batch_iou2


average_iou = total_iou / len(val_dataloader)
average_iou2 = total_iou2 / len(val_dataloader)
print("Average IoU:", average_iou,average_iou2)


# with torch.no_grad():  # 在评估过程中不计算梯度
#     for data in unlabel_dataloader:
#         inputs, _ = data
#         print(inputs.shape)
#         plot_images(inputs)
#         inputs = inputs.to(device)
#
#         outputs = model(inputs)
#         probabilities = torch.softmax(outputs, dim=1)
#         # 选择概率最高的类别作为最终预测
#         pred = torch.argmax(probabilities, dim=1).cpu()
#
#         draw_label_masks(pred)