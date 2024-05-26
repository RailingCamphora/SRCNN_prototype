import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from SRCNN_function import SRCNN1,SRCNN2,SRCNN,SRCNN3,SRCNN2_alter
import numpy as np
from scipy.ndimage import median_filter
import math
#from SRCNN_function import SRCNN_resnets
def scale_to_range(tensor):
    # 计算最小值和最大值
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    # 计算缩放比例
    scale_factor = 255.0 / (max_val - min_val)

    # 缩放Tensor中的值到[0, 255]范围内，并保持相对比例不变
    tensor_scaled = (tensor - min_val) * scale_factor

    return tensor_scaled



def psnr(original, reconstructed):
    o1=np.array(original).astype('float')
    r1=np.array(reconstructed).astype('float')
    mse = np.mean((o1- r1) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr_value

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



# 加载预训练的模型参数
model = SRCNN_resnets()
model.load_state_dict(torch.load('srcnn_modelresnet_lr_to_hr.pth'))
model.eval()

# 读取灰度图像
image = Image.open('./picture/选做作业三图像/boat_64X64.png').convert('L')  # 转换为灰度图像
image_256=Image.open('./picture/选做作业三图像/boat.png').convert('L')
image_resize256=image.resize((256,256))

# 对图像进行预处理
# 定义转换
transform_lr = transforms.Compose([
    transforms.Resize((64, 64)),  # 确保与模型输入大小相同
    transforms.ToTensor()  # 转换为张量
])

image_tensor = transform_lr(image).unsqueeze(0)  # 添加一个 batch 维度

# 使用模型进行超分辨率成像
with torch.no_grad():
    sr_image = model(image_tensor)

# 后处理：将超分辨率图像从张量转换为 PIL 图像
#sr_image = sr_image.squeeze().clamp(0, 1).numpy()  # 移除批次维度，并将像素值限制在 [0, 1] 范围内
#sr_image = (sr_image * 255).astype('uint8')  # 将像素值从 [0, 1] 转换为 [0, 255] 的整数
#sr_image = Image.fromarray(sr_image, mode='L')  # 从 NumPy 数组创建 PIL 图像对象
# 将输出张量转换为图像格式

#sr_image=torch.clamp(sr_image,0,255)

output_image = transforms.ToPILImage()(sr_image.squeeze(0).cpu())

#在这里定义图像numpy形式
output_array=np.array(output_image).astype('float')
image_resize256_array=np.array(image_resize256).astype('float')
error_array=output_array-image_resize256_array

#对edited out_image中值滤波
L=np.abs(error_array)<=140
error_array1=error_array.copy()
for _ in range(40):
    error_array=median_filter(error_array,size=20)
    error_array[L] = error_array1[L]

for _ in range(40):
    error_array=median_filter(error_array,size=2)
    error_array[L] = error_array1[L]
error_array[L]=error_array1[L]

#error_array_abs=np.abs(error_array)
#error_array_abs=error_array_abs.astype('uint8')

#据此,可以定义一个修订版output_image
output_array_edited=image_resize256_array+error_array
output_array_edited=np.clip(output_array_edited,0.0,255.0)
output_array_edited=output_array_edited.astype('uint8')
output_image_edited=Image.fromarray(output_array_edited)
print('let us calculate the edited version psnr')
print(psnr(output_image_edited,image_256))
plt.figure()
plt.imshow(output_image_edited,cmap='gray')
plt.show()



#filtered_image=median_filter(output_image)
filter_array=(np.array(output_image).astype('float')+np.array(image_resize256).astype('float'))/2.0
filter_array=filter_array.astype('uint8')
filtered_image=Image.fromarray(filter_array)



plt.figure()
plt.subplot(221)
plt.imshow(output_image,cmap='gray')
plt.title('sr_image')
plt.subplot(222)
plt.imshow(filtered_image,cmap='gray')
plt.title('sr_image_denoise')
plt.subplot(223)
plt.imshow(image_resize256,cmap='gray')
plt.title('original image resize')
plt.subplot(224)
plt.imshow(image_256,cmap='gray')
plt.title('256X256 image')
plt.show()
print('net psnr')
print(psnr(output_image,image_256))
print('bicubic psnr')
print(psnr(image_resize256,image_256))
print('filterd net psnr')
print(psnr(filtered_image,image_256))