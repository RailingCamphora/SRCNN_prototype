import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from SRCNN_function import SRCNN


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




# 加载预训练的模型参数
model = SRCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义数据预处理
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

transform_hr = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
transform_lr = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])



# 加载 BSDS300 数据集
train_dataset = CustomDataset(root_dir='./dataset/BSDS300/images/train', transform=(transform_hr,transform_lr))
test_dataset = CustomDataset(root_dir='./dataset/BSDS300/images/test', transform=(transform_hr,transform_lr))

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 修改损失函数
criterion = nn.MSELoss()

# 训练模型
num_epochs = 12
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data_hr, data_lr) in enumerate(train_loader):
        optimizer.zero_grad()
        output_hr = model(data_lr)
        loss = criterion(output_hr, data_hr)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch+1, num_epochs, batch_idx+1, len(train_loader), loss.item()))

# 在测试集上评估模型
model.eval()
test_loss = 0
with torch.no_grad():
    for data_hr, data_lr in test_loader:
        output_hr = model(data_lr)
        test_loss += criterion(output_hr, data_hr).item()
test_loss /= len(test_loader.dataset)
print('Test Loss: {:.4f}'.format(test_loss))

# 保存模型
torch.save(model.state_dict(), 'srcnn_model_gray.pth')