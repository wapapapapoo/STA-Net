import torch
import torch.nn as nn

from eval import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_loss(model, loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg_input"].to(DEVICE)
            fnirs = batch["fnirs_input"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            output = model(eeg, fnirs)
            target = torch.argmax(label, dim=1)

            loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(loader)


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

        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)

        val_loss = compute_loss(model, val_loader, criterion)

        print(
            f"epoch: {epoch}, "
            f"train_loss: {train_loss:.4f}, "
            f"val_loss: {val_loss:.4f} , "
            f"train_acc: {train_acc:.4f} , "
            f"val_acc: {val_acc:.4f}"
        )

    return model