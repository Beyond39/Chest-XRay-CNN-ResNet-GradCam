import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from archive.v1_baseline.v1_dataset import ChestXRay
from archive.v1_baseline.v1_preprocess import build_transform
from archive.v1_baseline.v1_model import SimpleCNN


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集与 DataLoader
    train_dataset = ChestXRay(
        rootdir='data/raw',
        split='train',
        transform=build_transform()
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )

    # 模型、损失函数、优化器
    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5
    loss_history = []

    best_loss = float('inf')  # 用于保存 loss 最小的模型

    # 保存模型的路径
    save_path = "archive/v1_baseline/v1_models.pth"
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 确保父目录存在

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 每个 epoch 平均 loss
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        loss_history.append(epoch_loss)

        # 保存 loss 最小的模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model with lowest loss updated, saved to {save_path}")

    # 绘制 loss 曲线
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch Loss Curve")
    plt.show()


if __name__ == "__main__":
    main()


