import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# 定义 SRCNN 模型
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# 自定义转换函数用于训练图像
def train_transform(image):
    transform = transforms.Compose([
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    return transform(image)

# 加载 BSDS300 数据集
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

# 定义数据预处理
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 加载 BSDS300 数据集
train_dataset = CustomDataset(root_dir='./dataset/BSDS300/images/train', transform=train_transform)
test_dataset = CustomDataset(root_dir='./dataset/BSDS300/images/test', transform=test_transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 实例化模型和优化器
model = SRCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义损失函数
criterion = nn.MSELoss()

# 训练模型
num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)  # 使用自身作为目标
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch+1, num_epochs, batch_idx+1, len(train_loader), loss.item()))

# 在测试集上评估模型
model.eval()
test_loss = 0
with torch.no_grad():
    for data in test_loader:
        output = model(data)
        test_loss += criterion(output, data).item()  # 使用自身作为目标
test_loss /= len(test_loader.dataset)
print('Test Loss: {:.4f}'.format(test_loss))

# 保存模型
torch.save(model.state_dict(), 'srcnn_model.pth')
