import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def draw_label_masks(masks, num_classes=49):
    """
    以彩色方式展示语义分割掩膜，使用颜色渐变为每个类别分配颜色。

    参数:
    masks - 一个形状为 [B, H, W] 的语义分割掩膜张量。
    num_classes - 掩膜中的类别数量，默认为 49。
    """
    B, H, W = masks.shape

    # 使用 Matplotlib 的 colormap
    cmap = plt.get_cmap('viridis')  # 可以选择其他颜色映射如 'jet', 'plasma', 'inferno', 'magma'
    colors = [cmap(i / num_classes) for i in range(num_classes)]

    ncols = min(B, 4)
    nrows = (B + ncols - 1) // ncols

    plt.figure(figsize=(15, nrows * 5))

    for i in range(B):
        plt.subplot(nrows, ncols, i + 1)
        colored_mask = np.zeros((H, W, 3), dtype=np.float32)
        for cls in range(num_classes):
            colored_mask[masks[i] == cls] = colors[cls][:3]  # 取 RGB 部分，忽略 alpha
        plt.imshow(colored_mask)
        plt.title(f'Image {i + 1}')
        plt.axis('off')

    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_images(tensor):
    """
    绘制一个图像张量，每行最多四张图像。
    参数:
        tensor (torch.Tensor): 一个形状为 [B, C, H, W] 的图像张量，其中C为3。
    """
    # 确保tensor是numpy数组
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    # 调整数据范围到[0, 1]
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()

    tensor = np.transpose(tensor, (0, 2, 3, 1))  # 重排形状为[B, H, W, C]

    # 计算总行数
    num_images = tensor.shape[0]
    num_rows = np.ceil(num_images / 4).astype(int)

    # 设置绘图
    fig, axes = plt.subplots(num_rows, 4, figsize=(12, 3 * num_rows))
    axes = axes.flatten()

    # 绘制每张图像
    for i in range(num_images):
        img = tensor[i]
        axes[i].imshow(img)
        axes[i].axis('off')  # 关闭坐标轴

    # 隐藏剩余的坐标轴（如果有的话）
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def plot_timestep_images(tensor):
    """
    绘制一个图像张量，每个批次的图像按时间步展开，每行最多四张子图像。
    参数:
        tensor (torch.Tensor): 一个形状为 [B, T, C, H, W] 的图像张量，其中C可以是1或3。
    """
    # 确保tensor是numpy数组
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    # 调整数据范围到[0, 1]
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()

    # 获取批次大小、时间步数、通道数
    B, T, C, H, W = tensor.shape

    # 如果是灰度图像，重复通道以适应imshow
    if C == 1:
        tensor = np.repeat(tensor, 3, axis=2)

    # 重排形状为 [B, T, H, W, C]
    tensor = np.transpose(tensor, (0, 1, 3, 4, 2))

    # 设置绘图
    fig, axes = plt.subplots(B, T, figsize=(12, 3 * B))
    if B == 1:
        axes = np.expand_dims(axes, 0)  # 确保axes是二维的
    if T == 1:
        axes = np.expand_dims(axes, 1)  # 确保axes是二维的

    # 绘制每个批次的每个时间步的图像
    for b in range(B):
        for t in range(T):
            img = tensor[b, t]
            ax = axes[b, t]
            ax.imshow(img)
            ax.axis('off')  # 关闭坐标轴

    plt.tight_layout()
    plt.show()


def draw_timestep_masks(masks, num_classes=49):
    """
    展示语义分割掩膜，使用颜色渐变为每个类别分配颜色。

    参数:
    masks - 一个形状为 [B, T, H, W] 的语义分割掩膜张量。
    num_classes - 掩膜中的类别数量，默认为 49。
    """
    B, T, H, W = masks.shape

    # 使用 Matplotlib 的 colormap
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / num_classes) for i in range(num_classes)]

    for b in range(B):
        plt.figure(figsize=(15, (T // 4 + 1) * 5))
        for t in range(T):
            plt.subplot((T // 4) + 1, min(T, 4), t + 1)
            colored_mask = np.zeros((H, W, 3), dtype=np.float32)
            for cls in range(num_classes):
                colored_mask[masks[b, t] == cls] = colors[cls][:3]
            plt.imshow(colored_mask)
            plt.title(f'Batch {b + 1}, Time {t + 1}')
            plt.axis('off')

        plt.show()

def map_pixels_tensor(image_tensor, mapping_tensor):
    # 扩展 mapping_tensor 以进行广播
    mapping_tensor = mapping_tensor.view(-1, 1, 1)

    # 计算每个像素与映射列表中每个值的差的绝对值
    diffs = torch.abs(image_tensor - mapping_tensor)

    # 找到差异最小的索引
    min_diff_indices = torch.argmin(diffs, dim=0)

    # 使用最小差异的索引从映射列表中获取值
    mapped_array = mapping_tensor[min_diff_indices].squeeze()

    return mapped_array

def convert_gray_image_to_mask(inputs_gray_images, pred_gray_images):
    """
    Adjust the pixel values in pred_gray_images based on the pixel values in the last time step of inputs_gray_images.

    Parameters:
    inputs_gray_images (torch.Tensor): A tensor of shape NT1HW representing input grayscale images.
    pred_gray_images (torch.Tensor): A tensor of shape NT1HW representing predicted grayscale images.

    Returns:
    torch.Tensor: A tensor of shape NTHW representing the adjusted segmentation masks.
    """
    # Remove the single-channel dimension
    inputs_masks = inputs_gray_images.squeeze(2)  # Assuming channel dimension is the third dimension
    pred_masks = pred_gray_images.squeeze(2)

    batch_ts=[]
    for n in range(pred_masks.size(0)):  # Iterate over each batch
        # Get the unique pixel values from the last time step of inputs_gray_images for this batch
        unique_pixels = torch.unique(inputs_masks[n, -1])
        print(unique_pixels)

        new_ts=[]
        # Iterate over each time step for this batch
        for t in range(pred_masks.size(1)):  # Iterate over each time step
            new_t=map_pixels_tensor(pred_masks[n,t], unique_pixels)
            new_ts.append(new_t)

        batch_t = torch.stack(new_ts, dim=0)
        batch_ts.append(batch_t)

    pred_masks_new=torch.stack(batch_ts, dim=0)

    # Scale pixel values up to the range 0-48
    pred_masks_new =(pred_masks_new*48).clamp(0, 48)

    # Convert to integer
    pred_masks_new = pred_masks_new.int()

    return pred_masks_new


import torch

def convert_gray_image_to_mask_simple(pred_gray_images):
    """
    Convert a batch of grayscale image tensors to segmentation masks.

    Parameters:
    pred_gray_images (torch.Tensor): A tensor of shape NT1HW representing grayscale images,
                                     where T is the number of time steps.

    Returns:
    torch.Tensor: A tensor of shape NTHW representing the segmentation masks,
                  with pixel values between 0 and 48 (inclusive).
    """
    # Remove the single-channel dimension
    masks = pred_gray_images.squeeze(2)  # Assuming channel dimension is the third dimension

    # Scale pixel values to range 0-48 and clamp to ensure values are within 0-48
    masks = (masks * 48).clamp(0, 48)

    # Convert to integer
    masks = masks.int()
    unique_pixels = torch.unique(masks[0, -1])
    print(unique_pixels)
    return masks


def convert_mask_to_gray_image(mask):
    """
    Convert a semantic segmentation mask to a single-channel grayscale image.

    Parameters:
    mask (torch.Tensor): A tensor of shape NTHW, where N is the batch size,
                         T is the number of time steps, H and W are the height
                         and width of the image. The pixel values are between 0 and 48.

    Returns:
    torch.Tensor: A tensor of shape NTCHW, where C is 1 (single channel),
                  with pixel values normalized to the range 0-1.
    """
    # Ensure the input tensor is a floating point type (needed for division)
    mask = mask.float()

    # Normalize the pixel values to be between 0 and 1
    normalized_mask = mask / 48.0

    # Add an extra dimension for the channel (C)
    normalized_mask = normalized_mask.unsqueeze(2)  # New shape: NT1HW

    return normalized_mask

def convert_to_one_hot(labels, num_classes=49):
    """
    将标签编码的图像转换为 one-hot 编码格式。兼容 [N, T, H, W] 和 [N, H, W] 形状的输入。

    参数:
    labels - 标签编码的图像。
    num_classes - 类别总数。

    返回:
    one_hot_labels - one-hot 编码的图像。
    """
    # 检查输入维度
    if labels.dim() == 4:  # 形状 [N, T, H, W]
        N, T, H, W = labels.size()
        one_hot_labels = torch.zeros(N, T, num_classes, H, W, device=labels.device)
        labels = labels.long().unsqueeze(2)  # 形状变为 [N, T, 1, H, W]
        one_hot_labels.scatter_(2, labels, 1)
    elif labels.dim() == 3:  # 形状 [N, H, W]
        N, H, W = labels.size()
        one_hot_labels = torch.zeros(N, num_classes, H, W, device=labels.device)
        labels = labels.long()
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
    else:
        raise ValueError("Unsupported label shape: {}".format(labels.size()))

    return one_hot_labels


import torch


def convert_from_one_hot(one_hot_labels, dtype=torch.float32):
    """
    将 one-hot 编码的图像转换为标签编码格式。兼容 [N, C, H, W] 和 [N, T, C, H, W] 形状的输入。

    参数:
    one_hot_labels - one-hot 编码的图像。

    返回:
    labels - 标签编码的图像。
    """
    # 检查输入维度
    if one_hot_labels.dim() == 5:  # 形状 [N, T, C, H, W]
        labels = torch.argmax(one_hot_labels, dim=2)
    elif one_hot_labels.dim() == 4:  # 形状 [N, C, H, W]
        labels = torch.argmax(one_hot_labels, dim=1)
    else:
        raise ValueError("Unsupported one-hot label shape: {}".format(one_hot_labels.size()))

        # 转换数据类型
    labels = labels.to(dtype)
    return labels


# 示例使用
# 假设 one_hot_labels 是一个形状为 [N, C, H, W] 或 [N, T, C, H, W] 的张量
# labels = convert_from_one_hot(one_hot_labels)


class IoULoss(nn.Module):
    """
    计算基于交并比 (Intersection over Union, IoU) 的损失，常用于语义分割任务。
    IoU 损失用于评估预测掩码和真实掩码之间的重叠度。此实现提供了平滑项以避免除以零。

    参数:
        smooth (float, 可选): 平滑项，用以防止分母为零。默认为 1e-6。

    输入:
        inputs (Tensor): 模型的预测输出，形状为 [N, C, H, W]，其中
                         N 是批大小，C 是类别数，H 和 W 是图像的高度和宽度。
        targets (Tensor): 真实的标签（掩码），形状同 inputs。

    输出:
        loss (Tensor): 计算得到的 IoU 损失值，是一个标量，表示整个批次的平均损失。
    """

    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 如果输入和目标完全相同，输出 'equal'（用于调试）
        if torch.equal(inputs, targets):
            print('equal')

        # 计算交集和并集
        intersection = torch.sum(inputs * targets, dim=(2, 3))
        total = torch.sum(inputs + targets, dim=(2, 3))
        union = total - intersection

        # 计算 IoU
        IoU = (intersection + self.smooth) / (union + self.smooth)

        # 损失为 1 - IoU 的均值
        return 1 - IoU.mean()


def iou(inputs, targets, smooth=1e-6):
    """
    计算基于交并比 (IoU) 的损失，常用于语义分割任务。
    此函数计算预测掩码和真实掩码之间的 IoU 损失，忽略背景类别。

    参数:
        inputs (Tensor): 预测输出，形状为 [N, C, H, W]。
        targets (Tensor): 真实的标签（掩码），形状同 inputs。
        smooth (float): 平滑项，用以防止分母为零。默认为 1e-6。

    返回:
        loss (Tensor): 计算得到的 IoU 损失值，是一个标量，表示整个批次的平均损失。
    """

    # 忽略背景类别（类别 0）
    inputs = inputs[:, 1:, :, :]
    targets = targets[:, 1:, :, :]

    # 计算交集和并集
    intersection = torch.sum(inputs * targets, dim=(2, 3))
    total = torch.sum(inputs + targets, dim=(2, 3))
    union = total - intersection

    # 计算 IoU
    IoU = (intersection + smooth) / (union + smooth)

    return IoU.mean()

def iou_NHW(inputs, targets, ignore_index=0, smooth=1e-6):
    """
    计算基于交并比 (IoU) 的指标，常用于语义分割任务。
    此函数计算预测掩码和真实掩码之间的 IoU 指标，忽略特定类别（默认为背景）。

    参数:
        inputs (Tensor): 预测输出，形状为 [N, H, W]，每个像素的值为类别索引。
        targets (Tensor): 真实的标签（掩码），形状同 inputs。
        ignore_index (int): 要忽略的类别索引。默认为 0（背景）。
        smooth (float): 平滑项，用以防止分母为零。默认为 1e-6。

    返回:
        IoU (Tensor): 计算得到的 IoU 值，是一个标量，表示整个批次的平均 IoU。
    """

    # 确保输入和目标位于同一设备
    device = inputs.device
    inputs = inputs.to(device)
    targets = targets.to(device)

    # 计算每个类别的 IoU，并忽略背景类别
    unique_classes = torch.unique(targets)
    # print(unique_classes)
    ious = []
    for cls in unique_classes:
        if cls == ignore_index:
            continue

        # 确保cls也在相同的设备上
        cls = cls.to(device)

        # 提取特定类别的预测和真实标签
        inputs_cls = (inputs == cls).float()
        targets_cls = (targets == cls).float()
        # print(f"Class {cls.item()} - inputs_cls:\n{inputs_cls}")
        # print(f"Class {cls.item()} - targets_cls:\n{targets_cls}")
        # 计算交集和并集
        intersection = torch.sum(inputs_cls * targets_cls, dim=(1, 2)).sum()
        total = torch.sum(inputs_cls + targets_cls, dim=(1, 2)).sum()
        union = total - intersection
        #
        # print(f"Class {cls.item()} - Intersection: {intersection.item()}, Union: {union.item()}")
        # 计算 IoU
        IoU = (intersection + smooth) / (union + smooth)
        ious.append(IoU)
        # print(f"Class {cls.item()} - IoU: {IoU.item()}")
    # 如果没有有效的类别（所有像素都是背景），返回零损失
    if len(ious) == 0:
        return torch.tensor(0.0).to(device)

    # 计算平均 IoU
    iou_sum = torch.tensor(0.0).to(device)  # 确保iou_sum在正确的设备上
    for IoU in ious:
        iou_sum += IoU
    iou_mean = iou_sum / len(ious)

    return iou_mean

def iou_loss_NHW(inputs, targets, ignore_index=0, smooth=1e-6):
    iou = iou_NHW(inputs, targets, ignore_index=ignore_index, smooth=smooth)
    return 1 - iou

#21to1:训练集21个语义分割，标签1个语义分割（预测第22帧）
#11to11:训练集11个语义分割，标签11个语义分割（预测后11帧）

class MaskMaskDataset(Dataset):
    # 用21帧预测第22帧
    def __init__(self, dataset):
        if dataset == 'Train':
            initial_videos = [os.path.join('train', 'video_' + str(i)) for i in range(1000)]
            self.videos = []
            for video_path in initial_videos:
                mask_path = os.path.join(video_path, 'mask.npy')
                mask = np.load(mask_path)
                # 如果第11帧和最后一帧的类别相同，则保留该样本
                if set(np.unique(mask[21])).issubset(set(np.unique(mask[10]))):
                    self.videos.append(video_path)

        elif dataset == 'Train+Unlabeled':
            initial_videos = [os.path.join('train', 'video_' + str(i)) for i in range(1000)] + [os.path.join('unlabeled', 'video_' + str(i)) for i in range(2000, 15000)]
            self.videos = []
            for video_path in initial_videos:
                mask_path = os.path.join(video_path, 'mask.npy')
                mask = np.load(mask_path)
                # 如果第11帧和最后一帧的类别相同，则保留该样本
                if set(np.unique(mask[21])).issubset(set(np.unique(mask[10]))):
                    self.videos.append(video_path)

        elif dataset == 'hidden':
            initial_videos = [os.path.join('hidden', 'video_' + str(i)) for i in range(15000, 17000)]
            self.videos = []
            for video_path in initial_videos:
                self.videos.append(video_path)

        elif dataset == 'Train+Unlabeled-Full':
            initial_videos = [os.path.join('train', 'video_' + str(i)) for i in range(1000)] + [os.path.join('unlabeled', 'video_' + str(i)) for i in range(2000, 15000)]
            self.videos = []
            for video_path in initial_videos:
                self.videos.append(video_path)

        elif dataset == 'Val':
            self.videos = [os.path.join('val', 'video_' + str(i)) for i in range(1000, 2000)]
        else:
            raise ValueError("Dataset must be 'Train' or 'Val'or 'Train+Unlabeled'")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        mask_path = os.path.join(video_path, 'mask.npy')
        mask = np.load(mask_path)

        frames = mask[:11]
        label = mask[:11] if len(self.videos)==2000 else mask[11:22]

        # 转换为PyTorch张量
        frames = torch.from_numpy(frames).float()
        label = torch.from_numpy(label).float()

        return frames, label

class ImageImageDataset(Dataset):
    def __init__(self, dataset, mode="21to1"):
        self.mode = mode
        if dataset == 'Train':
            self.videos = [os.path.join('train', 'video_' + str(i)) for i in range(1000)]
            self.is_train = True
        elif dataset == 'Val':
            self.videos = [os.path.join('val', 'video_' + str(i)) for i in range(1000, 2000)]
            self.is_train = False
        else:
            raise ValueError("Dataset must be 'Train' or 'Val'")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]

        # 读取所有图片
        images = []
        for i in range(22):
            image_path = os.path.join(video_path, f'image_{i}.png')
            image = Image.open(image_path).convert('RGB')
            images.append(image)

        # 选择模式
        if self.is_train:
            frames = images[:21] if self.mode == "21to1" else images[:11]
            label = images[-1] if self.mode == "21to1" else images[11:22]
        else:
            frames = images[:11] if self.mode == "21to1" else images[:11]
            label = images[-1] if self.mode == "21to1" else images[11:22]

        # 图像转换为CHW格式
        transform = transforms.Compose([
            transforms.ToTensor(),  # 这个转换会自动将 HWC 转换为 CHW
        ])
        frames = torch.stack([transform(frame) for frame in frames])
        if isinstance(label, list):
            label = torch.stack([transform(l) for l in label])
        else:
            label = transform(label)

        return frames, label

class ImageMaskDataset(Dataset):
    def __init__(self, dataset_type='train'):
        self.is_hidden = False
        if dataset_type in ['train', 'val']:
            self.root_dir = dataset_type
            folder_range = range(1000) if dataset_type == 'train' else range(1000, 2000)
            self.is_labeled = True
        elif dataset_type == 'unlabeled':
            self.root_dir = 'unlabeled'
            folder_range = range(2000, 15000)
            self.is_labeled = False
        elif dataset_type == 'hidden':
            self.root_dir = 'hidden'
            folder_range = range(15000,17000)
            self.is_labeled = False
            self.is_hidden = True
        else:
            raise ValueError("dataset_type must be 'train', 'val', or 'unlabeled'")

        self.video_folders = [os.path.join(self.root_dir, f'video_{i}') for i in folder_range]
        self.image_paths = []
        self.folder_indices = []

        for index, folder in enumerate(self.video_folders):
            for i in range(11 if dataset_type == 'hidden' else 22):
                image_path = os.path.join(folder, f'image_{i}.png')
                self.image_paths.append(image_path)
                self.folder_indices.append(index)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()

        if self.is_labeled:
            folder_index = self.folder_indices[idx]
            mask_path = os.path.join(self.video_folders[folder_index], 'mask.npy')
            mask = np.load(mask_path)
            label = torch.from_numpy(mask[idx % 11 if self.is_hidden else 21]).long()
            # remind: change 21 to 11 if hidden, and change 11 to 21 if other
        else:
            label = torch.tensor(-1)

        return image, label

def checkpoint_save(epoch,model, optimizer,scheduler):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
    }, 'checkpoint.pth')


def filter_pred_by_input(pred, input):
    # 确保 pred 和 input 形状相同
    assert pred.shape == input.shape, "pred 和 input 的形状必须相同"

    N, H, W = pred.shape

    for n in range(N):
        # 获取当前批次的 unique 像素值
        unique_input_vals = torch.unique(input[n])
        unique_pred_vals = torch.unique(pred[n])

        # 检查是否相同，如果是，则继续下一个
        if torch.equal(unique_input_vals, unique_pred_vals):
            continue

        unique_input_vals = torch.unique(input[n])

        # 扩展 unique_input_vals 以进行广播
        unique_input_vals = unique_input_vals.view(-1, 1, 1)

        # 计算 pred 中的像素值是否存在于 input 的 unique 像素值中
        mask = torch.any(pred[n].unsqueeze(0) == unique_input_vals, dim=0)

        # 用 mask 更新 pred 中的像素值，如果 pred 的像素值不在 input 的 unique 像素值中，则设置为0
        pred[n] *= mask

    return pred


import torch


import torch

import torch


def remove_rare_pixels(pred, threshold=100):
    """
    将出现次数低于阈值的像素值设置为0。

    参数:
        pred (torch.Tensor): 输入的图像张量，形状为 N x H x W。
        threshold (int): 出现次数的阈值，低于此值的像素将被视为异常并设置为0。

    返回:
        torch.Tensor: 处理后的图像张量。
    """
    N, H, W = pred.shape

    # 展平 pred 以便于计算
    flat_pred = pred.view(N, -1)

    # 对每个批次进行处理
    for n in range(N):
        # 获取当前批次的像素值及其出现次数
        unique_vals, counts = torch.unique(flat_pred[n], return_counts=True)

        # 确定哪些像素值的出现次数低于阈值
        rare_vals = unique_vals[counts < threshold]

        # 将这些像素值设置为0
        for val in rare_vals:
            pred[n][pred[n] == val] = 0

    return pred

