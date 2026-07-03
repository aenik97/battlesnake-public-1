"""Imitation Learning: train CNN to mimic the heuristic, export to ONNX."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class SnakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.net(x)


def train():
    data = np.load("data.npz")
    X = torch.tensor(data["X"])
    y = torch.tensor(data["y"])

    n = len(X)
    split = int(0.9 * n)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=512)

    model = SnakeCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.3)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(20):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(yb)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total += len(yb)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += len(yb)

        tr_acc = tr_correct / tr_total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1:2d}: loss={tr_loss/tr_total:.4f}  "
              f"train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "model_il.pt")

        scheduler.step()

    print(f"\nBest val_acc={best_val_acc:.4f} — saved model_il.pt")

    # Reload best weights and export to ONNX (dynamo exporter handles AdaptiveAvgPool2d)
    model.load_state_dict(torch.load("model_il.pt"))
    model.eval()
    dummy = torch.zeros(1, 6, 11, 11)
    torch.onnx.export(model, (dummy,), "model.onnx", dynamo=True)
    print("Saved model.onnx")


if __name__ == "__main__":
    train()
