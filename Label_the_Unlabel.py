from torch.utils.data import DataLoader
from utils import *
from model.UNet import *
import torch
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 11

unlabel_dataset = ImageMaskDataset(dataset_type='hidden')
unlabel_dataloader = DataLoader(unlabel_dataset, batch_size=batch_size, shuffle=False)

print(len(unlabel_dataloader))

model = UNet(49).to(device)
saved_state_dict = torch.load('weight/unet_20.pth', map_location=device)
model.load_state_dict(saved_state_dict)

# 确保输出目录存在
output_dir = "./hidden"
os.makedirs(output_dir, exist_ok=True)

video_index = 15000  # 起始索引

with torch.no_grad():
    for data in tqdm(unlabel_dataloader, desc="Processing unlabelled data"):
        inputs, _ = data
        inputs = inputs.to(device)

        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probabilities, dim=1).cpu()
        pred = remove_rare_pixels(pred)

        save_path = os.path.join(output_dir, f"video_{video_index}", "mask.npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # print(video_index)
        np.save(save_path, pred.numpy().astype(np.uint8))
        video_index += 1
