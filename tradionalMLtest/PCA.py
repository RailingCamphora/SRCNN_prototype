import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
# 下载 Fashion MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# 获取数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
dataiter = iter(trainloader)
images, labels = next(dataiter)

# 数据预处理
images = images.view(images.shape[0], -1).numpy()  # 将图像展平为一维数组

print(type(images))








# 对数据进行主成分分析
pca = PCA(n_components=3)
pca.fit(images)




# 可视化结果
components = pca.transform(images)

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

#scatter the result
ax.scatter(components[:, 0], components[:, 1], components[:,2],c=labels.numpy(), cmap='tab10', alpha=0.5,s=2 )
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

plt.title('Fashion MNIST PCA')
plt.show()
