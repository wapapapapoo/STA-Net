import torch
import torch.nn as nn
from eval import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================
# Loss
# ==============================

class LossModule(nn.Module):

    def __init__(self, align_w=0.05, trial_w=0.05):
        super().__init__()

        self.ce = nn.CrossEntropyLoss()

        self.align_w = align_w
        self.trial_w = trial_w

    def classification(self, logits, label):

        target = torch.argmax(label, dim=1)

        return self.ce(logits, target)

    def alignment(self, eeg_feat, fnirs_feat):

        return torch.mean((eeg_feat - fnirs_feat) ** 2)

    def trial_consistency(self, feat, trial):

        unique_trials = torch.unique(trial)

        loss = 0.0
        count = 0

        for t in unique_trials:

            idx = trial == t
            f = feat[idx]

            if f.shape[0] < 2:
                continue

            mean = f.mean(dim=0, keepdim=True)

            loss += torch.mean((f - mean) ** 2)

            count += 1

        if count == 0:
            return torch.tensor(0.0, device=feat.device)

        return loss / count

    def forward(self, logits, label, eeg_feat, fnirs_feat, trial):

        cls_loss = self.classification(logits, label)

        align_loss = self.alignment(eeg_feat, fnirs_feat)

        trial_loss = self.trial_consistency(eeg_feat, trial)

        total = (
            cls_loss
            + self.align_w * align_loss
            + self.trial_w * trial_loss
        )

        return total


# ==============================
# Train Epoch
# ==============================

def train_epoch(model, loader, optimizer, loss_fn):

    model.train()

    total_loss = 0.0

    for batch in loader:

        eeg = batch["eeg_input"].to(DEVICE)
        fnirs = batch["fnirs_input"].to(DEVICE)
        label = batch["label"].to(DEVICE)
        trial = batch["trial_label"].to(DEVICE)

        optimizer.zero_grad()

        logits, eeg_feat, fnirs_feat = model(eeg, fnirs)

        loss = loss_fn(
            logits,
            label,
            eeg_feat,
            fnirs_feat,
            trial
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ==============================
# Validation Loss
# ==============================

def compute_val_loss(model, loader):

    criterion = nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            eeg = batch["eeg_input"].to(DEVICE)
            fnirs = batch["fnirs_input"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            logits, _, _ = model(eeg, fnirs)

            target = torch.argmax(label, dim=1)

            loss = criterion(logits, target)

            total_loss += loss.item()

    return total_loss / len(loader)


# ==============================
# Train Loop
# ==============================

def train(model, train_loader, val_loader, args):

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4
    )

    loss_fn = LossModule()

    for epoch in range(20):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn
        )

        train_acc = evaluate(model, train_loader)

        val_acc = evaluate(model, val_loader)

        val_loss = compute_val_loss(
            model,
            val_loader
        )

        print(
            f"ep:{epoch:02d} | "
            f"tl:{train_loss:.4f} | "
            f"vl:{val_loss:.4f} | "
            f"tacc:{train_acc:.4f} | "
            f"vacc:{val_acc:.4f}"
        )

    return model