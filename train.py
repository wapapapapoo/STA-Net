import torch
import torch.nn as nn

from eval import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# LOSS
# =========================================================

class LossModule:

    def __init__(self, cls_weight=1.0, align_weight=0.05, trial_weight=0.05):
        self.cls_weight = cls_weight
        self.align_weight = align_weight
        self.trial_weight = trial_weight

        self.ce = nn.CrossEntropyLoss()

    def classification(self, logits, label):

        target = torch.argmax(label, dim=1)

        return self.ce(logits, target)

    def alignment(self, eeg_feat, fnirs_feat):

        return ((eeg_feat - fnirs_feat) ** 2).mean()

    def trial_consistency(self, feat, trial):

        unique_trials = torch.unique(trial)

        loss = 0

        for t in unique_trials:

            idx = trial == t
            f = feat[idx]

            if f.shape[0] < 2:
                continue

            m = f.mean(dim=0)

            loss += ((f - m) ** 2).mean()

        return loss / len(unique_trials)

    def total(self, logits, label, eeg_feat, fnirs_feat, trial):

        cls_loss = self.classification(logits, label)

        align_loss = self.alignment(eeg_feat, fnirs_feat)

        trial_loss = self.trial_consistency(eeg_feat, trial)

        loss = (
            self.cls_weight * cls_loss
            + self.align_weight * align_loss
            + self.trial_weight * trial_loss
        )

        return loss
        

# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_epoch(model, loader, optimizer, loss_module):

    model.train()

    total_loss = 0

    for batch in loader:

        eeg = batch["eeg_input"].to(DEVICE)
        fnirs = batch["fnirs_input"].to(DEVICE)
        label = batch["label"].to(DEVICE)
        trial = batch["trial_label"].to(DEVICE)

        optimizer.zero_grad()

        logits, eeg_feat, fnirs_feat = model(eeg, fnirs)

        loss = loss_module.total(
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


# =========================================================
# VALIDATION LOSS
# =========================================================

def compute_val_loss(model, loader):

    ce = nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for batch in loader:

            eeg = batch["eeg_input"].to(DEVICE)
            fnirs = batch["fnirs_input"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            logits, _, _ = model(eeg, fnirs)

            target = torch.argmax(label, dim=1)

            loss = ce(logits, target)

            total_loss += loss.item()

    return total_loss / len(loader)


# =========================================================
# TRAIN LOOP
# =========================================================

def train(model, train_loader, val_loader, args):

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    loss_module = LossModule()

    for epoch in range(20):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_module
        )

        train_acc = evaluate(model, train_loader)

        val_acc = evaluate(model, val_loader)

        val_loss = compute_val_loss(model, val_loader)

        print(
            f"ep:{epoch} | "
            f"tl:{train_loss:.4f} | "
            f"vl:{val_loss:.4f} | "
            f"tacc:{train_acc:.4f} | "
            f"vacc:{val_acc:.4f}"
        )

    return model