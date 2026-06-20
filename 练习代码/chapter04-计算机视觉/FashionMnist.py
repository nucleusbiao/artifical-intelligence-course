

## 修改下面的代码，使其准确度达到95%以上

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 硬件加速配置 (GPU / CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 超参数设置
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

# 3. 数据预处理与加载
# 将图像转换为 Tensor 并归一化到 [-1, 1] 空间
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载并加载训练集与测试集
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 4. 构建卷积神经网络 (CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 输入通道 1 (灰度图)，输出通道 16，卷积核 3x3
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        )
        # 输入通道 16，输出通道 32，卷积核 3x3
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        )
        # 全连接层：32 * 7 * 7 = 1568 特征输入，10 个类别输出
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平特征图为一维向量
        x = self.fc(x)
        return x


model = SimpleCNN().to(device)

# 5. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 6. 训练网络
def train(model, loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计数据
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")


# 7. 测试网络性能
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 评估模式下不计算梯度
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"\nAccuracy on the 10000 test images: {100 * correct / total:.2f}%")


# 运行训练与评估
if __name__ == '__main__':
    print("--- Start Training ---")
    train(model, train_loader, criterion, optimizer, EPOCHS)
    print("--- Start Evaluation ---")
    evaluate(model, test_loader)