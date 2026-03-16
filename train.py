import torch
import torch.nn as nn
import torch.nn.functional as F

from eval import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LossModule(nn.Module):

    def __init__(self):

        super().__init__()

        self.ce = nn.CrossEntropyLoss()

    def forward(self, output, label):

        target = torch.argmax(label, dim=1)

        loss_main = (
            self.ce(output["eeg_logits"], target) +
            self.ce(output["fnirs_logits"], target) +
            self.ce(output["fusion_logits"], target)
        )

        trial_group = output["trial_group"]

        loss_session = (
            self.ce(output["session_eeg"], trial_group) +
            self.ce(output["session_fnirs"], trial_group) +
            self.ce(output["session_fusion"], trial_group)
        )

        loss = loss_main #+ 0.3 * loss_session

        return loss



def train_epoch(model, loader, optimizer, loss_fn, args):

    model.train()

    total_loss = 0

    for batch in loader:

        eeg = batch["eeg_input"].to(DEVICE)
        fnirs = batch["fnirs_input"].to(DEVICE)
        label = batch["label"].to(DEVICE)
        trial_label = batch["trial_label"].to(DEVICE)

        optimizer.zero_grad()

        # forward
        output = model(eeg, fnirs)

        # 计算 trial group
        trial_group = trial_label // args["TRAIL_GROUP"]

        # 放进 output
        output["trial_group"] = trial_group

        # loss
        loss = loss_fn(
            output,
            label
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.
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
            loss_fn,
            args
        )

        train_acc = evaluate(model, train_loader)

        val_acc = evaluate(model, val_loader)

        val_loss = compute_val_loss(
            model,
            val_loader
        )

        print(
            f"ep {epoch:02d} "
            f"tl {train_loss:.4f} "
            f"vl {val_loss:.4f} "
            f"tacc {train_acc:.4f} "
            f"vacc {val_acc:.4f} "
            f"tacc {evaluate(model, args['test_loader'])} ", flush=True
        )

    return model
