import os
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, models
from datasets import CustomImageDataset
from augmentations import all_augs
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_transforms(train=True):
    base = [transforms.Resize((224, 224))]
    if train:
        base += [all_augs]
    base += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(base)

# ───────────────────────────────────────────────────────────────────── #
def train_model(epochs=20, lr=1e-3, batch_size=32, early_stop_patience=3):
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = CustomImageDataset('data/train', transform=get_transforms(train=True))
    val_ds = CustomImageDataset('data/val', transform=get_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, len(train_ds.get_class_names()))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    log_file = open("results/log.txt", "w", encoding='utf-8')

    for epoch in range(1, epochs + 1):
        # TRAIN
        model.train()
        total_loss = 0
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # VALIDATE
        model.eval()
        val_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out, y).item()
                preds = out.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        log_line = f"[{epoch:02d}/{epochs}] " \
                   f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, " \
                   f"Val Acc: {val_acc:.4f}"
        print(log_line)
        log_file.write(log_line + '\n')

        # EARLY STOPPING
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), "results/best_model.pth")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"⏹ Early stopping at epoch {epoch}")
                log_file.write("Early stopping triggered.\n")
                break

    log_file.close()

    # ─── SAVE METRICS ─────────────────────────────────── #
    with open("results/metrics.json", "w") as f:
        json.dump({
            "best_val_accuracy": best_val_acc,
            "last_val_accuracy": val_acc,
            "train_loss": history['train_loss'],
            "val_loss": history['val_loss'],
            "val_acc": history['val_acc']
        }, f, indent=4)

    # ─── PLOTS ─────────────────────────────────────────── #
    def save_plot(data, label, ylabel):
        plt.figure()
        plt.plot(data)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(label)
        plt.grid()
        plt.savefig(f"plots/{label.lower().replace(' ', '_')}.png")
        plt.close()

    save_plot(history['train_loss'], 'Train Loss', 'Loss')
    save_plot(history['val_loss'], 'Validation Loss', 'Loss')
    save_plot(history['val_acc'], 'Validation Accuracy', 'Accuracy')

    # ─── CONFUSION MATRIX ─────────────────────────────── #
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_ds.get_class_names())
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("plots/confusion_matrix.png")
    plt.close()




if __name__ == '__main__':
    train_model()
