
import torch
import torch.nn as nn

from eval import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(model, loader, optimizer, criterion, args):
    model.train()
    total_loss = 0

    for batch in loader:
        eeg = batch["eeg_input"].to(DEVICE)
        fnirs = batch["fnirs_input"].to(DEVICE)
        label = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        output = model(eeg, fnirs)
        target = torch.argmax(label, dim=1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def train(model, train_loader, val_loader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args)
        val_acc = evaluate(model, val_loader)
        print(f"epoch {epoch} loss {train_loss:.4f} val {val_acc:.4f}")

    return model