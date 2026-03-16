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
        trial = batch["trial_label"].to(DEVICE)

        optimizer.zero_grad()
        output = model(eeg, fnirs)
        target = torch.argmax(label, dim=1)
        cls_loss = criterion(output, target)

        # trial consistency
        feat = output.detach()

        trial_mean = {}
        cons_loss = 0

        for i in range(len(trial)):
            t = trial[i].item()
            if t not in trial_mean:
                trial_mean[t] = []
            trial_mean[t].append(feat[i])

        for t in trial_mean:
            f = torch.stack(trial_mean[t])
            m = f.mean(0)
            cons_loss += ((f - m)**2).mean()

        cons_loss = cons_loss / len(trial_mean)
        loss = cls_loss + 0.1 * cons_loss
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
            f"epk:{epoch},"
            f"tl:{train_loss:.4f},"
            f"vl:{val_loss:.4f},"
            f"tacc:{train_acc:.4f},"
            f"vacc:{val_acc:.4f}"
        )

    return model
