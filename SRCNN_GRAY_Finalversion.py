import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from SRCNN_function import SRCNN3

# 修改数据集加载和预处理函数
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_name).convert('L')  # 转换为灰度图像
        if self.transform:
            image_hr = self.transform[0](image)  # 高分辨率图像
            image_lr = self.transform[1](image)  # 低分辨率图像
            return image_hr, image_lr
        else:return image
