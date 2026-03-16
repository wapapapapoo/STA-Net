import torch
import torch.nn as nn
from eval import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossModule(nn.Module):

    def __init__(self,
                 contrast_w=0.2,
                 pred_w=0.1,
                 temperature=0.07):

        super().__init__()

        self.ce = nn.CrossEntropyLoss()

        self.contrast_w = contrast_w
        self.pred_w = pred_w

        self.temp = temperature

    # --------------------------
    # classification
    # --------------------------

    def classification(self, logits, label):

        target = torch.argmax(label, dim=1)

        return self.ce(logits, target)

    # --------------------------
    # contrastive (InfoNCE)
    # --------------------------

    def contrastive(self, eeg_proj, fnirs_proj):

        eeg_proj = F.normalize(eeg_proj, dim=1)
        fnirs_proj = F.normalize(fnirs_proj, dim=1)

        logits = torch.matmul(eeg_proj, fnirs_proj.T) / self.temp

        labels = torch.arange(logits.shape[0]).to(logits.device)

        loss_e = F.cross_entropy(logits, labels)
        loss_f = F.cross_entropy(logits.T, labels)

        return (loss_e + loss_f) / 2

    # --------------------------
    # cross-modal prediction
    # --------------------------

    def prediction(self, fnirs_pred, fnirs_feat):

        return F.mse_loss(fnirs_pred, fnirs_feat)

    # --------------------------
    # total loss
    # --------------------------

    def forward(self, output, label):

        logits = output["logits"]
        eeg_proj = output["eeg_proj"]
        fnirs_proj = output["fnirs_proj"]
        fnirs_pred = output["fnirs_pred"]
        fnirs_feat = output["fnirs_feat"]

        cls_loss = self.classification(logits, label)

        contrast_loss = self.contrastive(
            eeg_proj,
            fnirs_proj
        )

        pred_loss = self.prediction(
            fnirs_pred,
            fnirs_feat
        )

        total = (
            cls_loss
            + self.contrast_w * contrast_loss
            + self.pred_w * pred_loss
        )

        return total





def train_epoch(model, loader, optimizer, loss_fn):

    model.train()

    total_loss = 0

    for batch in loader:

        eeg = batch["eeg_input"].to(DEVICE)
        fnirs = batch["fnirs_input"].to(DEVICE)
        label = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        output = model(eeg, fnirs)

        loss = loss_fn(
            output,
            label
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)





def compute_val_loss(model, loader):

    ce = nn.CrossEntropyLoss()

    model.eval()

    total = 0

    with torch.no_grad():

        for batch in loader:

            eeg = batch["eeg_input"].to(DEVICE)
            fnirs = batch["fnirs_input"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            output = model(eeg, fnirs)

            logits = output["logits"]

            target = torch.argmax(label, dim=1)

            loss = ce(logits, target)

            total += loss.item()

    return total / len(loader)











def train(model, train_loader, val_loader, args):

    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
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
