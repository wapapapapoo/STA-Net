import torch
import torch.nn as nn
import torch.nn.functional as F

from eval import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def contrastive_loss(z, trial):
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T)
    mask = trial.unsqueeze(1) == trial.unsqueeze(0)
    pos = sim[mask].mean()
    neg = sim[~mask].mean()
    return neg - pos

def compute_loss(model, loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg_input"].to(DEVICE)
            fnirs = batch["fnirs_input"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            logits, _ = model(eeg, fnirs)
            target = torch.argmax(label, dim=1)
            loss = criterion(logits, target)
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
        logits, feat = model(eeg, fnirs)
        target = torch.argmax(label, dim=1)
        cls_loss = criterion(logits, target)

        feat_norm = F.normalize(feat, dim=1)
        sim = torch.matmul(feat_norm, feat_norm.T)
        mask = trial.unsqueeze(1) == trial.unsqueeze(0)
        pos = sim[mask].mean()
        neg = sim[~mask].mean()
        contrast = neg - pos

        unique_trial, inverse = torch.unique(trial, return_inverse=True)
        trial_mean = []
        for i in range(len(unique_trial)):
            trial_mean.append(feat[inverse == i].mean(0))
        trial_mean = torch.stack(trial_mean)
        cons_loss = ((feat - trial_mean[inverse])**2).mean()

        loss = cls_loss + 0.1 * cons_loss + 0.05 * contrast
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
