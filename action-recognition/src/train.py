"""
Fine-tunes R(2+1)D-18 on the UCF101 class subset.

Usage:
    python src/train.py
"""

import os
import sys
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (  # noqa: E402
    BATCH_SIZE, CHECKPOINT_DIR, CLASSES, DATA_DIR, FRAME_SIZE,
    LEARNING_RATE, NUM_EPOCHS, NUM_FRAMES, NUM_WORKERS, SEED, WEIGHT_DECAY,
)
from src.dataset import VideoClipDataset  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import get_device, set_seed  # noqa: E402


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for clips, labels in tqdm(loader, desc="train" if train else "val", leave=False):
            clips, labels = clips.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
                with autocast(enabled=device.type == "cuda"):
                    outputs = model(clips)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast(enabled=device.type == "cuda"):
                    outputs = model(clips)
                    loss = criterion(outputs, labels)

            total_loss += loss.item() * clips.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += clips.size(0)

    return total_loss / total, correct / total


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_ds = VideoClipDataset(
        os.path.join(DATA_DIR, "train"), CLASSES, NUM_FRAMES, FRAME_SIZE, train=True
    )
    test_ds = VideoClipDataset(
        os.path.join(DATA_DIR, "test"), CLASSES, NUM_FRAMES, FRAME_SIZE, train=False
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Train clips: {len(train_ds)} | Test clips: {len(test_ds)}")

    model = build_model(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler(enabled=device.type == "cuda")

    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss, val_acc = run_epoch(model, test_loader, criterion, optimizer, scaler, device, train=False)
        scheduler.step()
        elapsed = time.time() - start

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.3f} | {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save({"model_state": model.state_dict(), "classes": CLASSES, "val_acc": val_acc}, ckpt_path)
            print(f"  ↳ New best model saved ({val_acc:.3f}) -> {ckpt_path}")

    print(f"\nTraining complete. Best val accuracy: {best_acc:.3f}")


if __name__ == "__main__":
    main()
