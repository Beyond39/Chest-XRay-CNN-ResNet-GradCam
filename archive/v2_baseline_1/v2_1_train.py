import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

from archive.v2_baseline_1.v2_1_model import ChestCNN
from src.data.dataset import get_data_loaders


# 1. 路径工程化（彻底解决路径问题）
BASE_DIR = Path(__file__).resolve()
PROJECT_ROOT = BASE_DIR.parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SAVE_DIR = PROJECT_ROOT / "archive"
PIC_DIR = PROJECT_ROOT / "outputs"

SAVE_DIR.mkdir(parents=True, exist_ok=True)
PIC_DIR.mkdir(parents=True, exist_ok=True)

# 2. 训练参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR = 0.0005
EPOCHS = 15

# 3. 训练函数
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Train", leave=False)
    for inputs, labels in loop:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, prec, rec, cm, all_labels, all_preds


# 4. 主函数
def main():
    print(f"Using device: {DEVICE}")
    print(f"Data Dir: {DATA_DIR}")

    if not DATA_DIR.exists():
        raise FileNotFoundError("Processed data not found!")

    train_loader, val_loader, test_loader, classes = get_data_loaders(DATA_DIR, BATCH_SIZE)
    print(f"Classes: {classes}")

    model = ChestCNN(num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_prec': [], 'val_rec': []
    }

    print(f"\nStart Training ({EPOCHS} epochs)")
    start_time = time.time()

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # Train
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)

        # Val
        v_loss, v_acc, v_prec, v_rec, _, _, _ = evaluate(model, val_loader, criterion)

        # Record
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        history['val_prec'].append(v_prec)
        history['val_rec'].append(v_rec)

        print(f"[Train] Loss: {t_loss:.4f} | Acc: {t_acc:.4f}")
        print(f"[Val]   Loss: {v_loss:.4f} | Acc: {v_acc:.4f}")

        # Save best model
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), SAVE_DIR / "best_model.pth")
            print(">>> Best model saved.")

    print(f"\nTraining finished in {(time.time()-start_time)/60:.1f} mins")

    # 5. 测试阶段
    print("\nLoading Best Model for Final Test...")

    model.load_state_dict(torch.load(SAVE_DIR / "best_model.pth", map_location=DEVICE))

    test_loss, test_acc, test_prec, test_rec, test_cm, true_labels, pred_labels = evaluate(
        model, test_loader, criterion
    )

    print("\n[FINAL TEST RESULT]")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")

    # 混淆矩阵
    plt.figure(figsize=(6, 5))
    sns.heatmap(test_cm, annot=True, fmt='d',
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix (Test)")
    plt.savefig(PIC_DIR / "test_confusion_matrix.png")

    print("Test results saved.")


if __name__ == "__main__":
    main()
