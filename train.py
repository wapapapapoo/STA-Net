import torch
import torch.nn as nn
from eval import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossModule(nn.Module):

    def __init__(self, align_w=0.2):

        super().__init__()

        self.ce = nn.CrossEntropyLoss()

        self.align_w = align_w

    def forward(self, output, label):

        logits = output["logits"]

        eeg_feat = output["eeg_feat"]

        fnirs_feat = output["fnirs_feat"]

        target = torch.argmax(label, dim=1)

        cls = self.ce(logits, target)

        align = F.mse_loss(eeg_feat, fnirs_feat)

        return cls + self.align_w * align




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

    for epoch in range(50):

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
