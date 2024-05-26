import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from matplotlib import pyplot as plt
import os
from torch.utils.data import DataLoader
from torchvision import transforms

# 定义 SRCNN 模型
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

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
            image_lr = self.transform[1](image_hr)  # 低分辨率图像
            return image_lr, image_hr
        return image

# 自定义转换函数用于训练图像
def train_transform_hr(image):
    transform_hr = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    return transform_hr(image)

def train_transform_lr(image):
    transform_lr = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    transform_lr(image)

class SRCNN1(nn.Module):
    def __init__(self):
        super(SRCNN1, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=5, padding=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        x=x.view(-1,1,256,256)
        return x



class SRCNN2(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN2, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')  # 第一次上采样，放大两倍
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')  # 第二次上采样，再放大两倍
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample1(x)  # 在卷积之前进行两倍上采样
        x = self.relu(self.conv1(x))
        x = self.upsample2(x)  # 在第1个卷积之后进行两倍上采样
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x

class SRCNN3(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN3, self).__init__()#它是nn.Module的子集，特殊情形，所以用super传参数
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # 添加上采样层，将输入图像尺寸增加为 256x256
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=3 // 2)
        self.conv4 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.upsample(x)  # 在第一次卷积之后进行上采样
        x = self.relu(self.conv2(x))
        x=self.upsample(x)  # 在第二次卷积之后进行上采样
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class SRCNN2_alter(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN2_alter, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=4, mode='nearest')  # 第一次上采样，放大4倍
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')  # 第二次上采样，再放大两倍
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample1(x)  # 在卷积之前进行四倍上采样
        x = self.relu(self.conv1(x))
        #x = self.upsample2(x)  # 在第1个卷积之后进行两倍上采样
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class SRCNN_resnets(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN_resnets, self).__init__()
        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')  # 添加上采样层，将输入图像尺寸增加为 256x256

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)

        self.res_block1 = ResidualBlock(32, 32)
        self.res_block2 = ResidualBlock(32, 32)
        self.res_block3 = ResidualBlock(32, 32)

        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = self.conv3(x)
        return x