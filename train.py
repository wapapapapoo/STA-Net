import torch
import torch.nn as nn
import torch.nn.functional as F

from eval import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------
# trial contrastive
# ------------------------------------------------

def trial_contrastive(feat, trial):

    feat = F.normalize(feat, dim=1)

    sim = torch.matmul(feat, feat.T)

    mask = trial.unsqueeze(1) == trial.unsqueeze(0)

    eye = torch.eye(len(trial), device=trial.device).bool()

    pos_mask = mask & ~eye
    neg_mask = ~mask

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0.0, device=feat.device)

    pos = sim[pos_mask].mean()
    neg = sim[neg_mask].mean()

    return neg - pos


# ------------------------------------------------
# trial consistency
# ------------------------------------------------

def trial_consistency(feat, trial):

    unique_trial, inverse = torch.unique(trial, return_inverse=True)

    trial_mean = torch.zeros(
        len(unique_trial),
        feat.shape[1],
        device=feat.device
    )

    trial_mean.index_add_(0, inverse, feat)

    counts = torch.bincount(inverse).float().unsqueeze(1)

    trial_mean = trial_mean / counts

    mean_feat = trial_mean[inverse]

    loss = ((feat - mean_feat) ** 2).mean()

    return loss


# ------------------------------------------------
# compute val loss
# ------------------------------------------------

def compute_loss(model, loader, criterion):

    model.eval()

    total = 0

    with torch.no_grad():

        for batch in loader:

            eeg = batch["eeg_input"].to(DEVICE)
            fnirs = batch["fnirs_input"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            logits, _ = model(eeg, fnirs)

            target = torch.argmax(label, dim=1)

            loss = criterion(logits, target)

            total += loss.item()

    return total / len(loader)


# ------------------------------------------------
# train epoch
# ------------------------------------------------

def train_epoch(model, loader, optimizer, criterion):

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

        cons_loss = trial_consistency(feat, trial)

        ctr_loss = trial_contrastive(feat, trial)

        loss = cls_loss + 0.1 * cons_loss + 0.05 * ctr_loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ------------------------------------------------
# train
# ------------------------------------------------

def train(model, train_loader, val_loader, args):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-3
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(20):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

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