#相较于第一版加入了对训练集的增强，效果又一些提高

import os
from torchvision import datasets , transforms
from torch.utils.data import DataLoader
from pathlib import Path

def get_data_loaders(data_dir, batch_size = 32):
    """
    此时的data_dir指向data/processed
    """
    # 转为 Path 对象
    data_dir = Path(data_dir)

    # 训练集 ：加入增强即可
    train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_and_test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 路径构建
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"
    test_dir  = data_dir / "test"

    # 路径检查
    for path in [train_dir, val_dir, test_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
    # 数据集对象
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_and_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_and_test_transforms)

    # Loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader , val_loader ,test_loader ,train_dataset.classes