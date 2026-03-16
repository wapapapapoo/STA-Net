import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

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

        loss = loss_main + loss_session

        return loss


def update_ema(model, ema_model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)



def train_epoch(model, ema_model, loader, optimizer, loss_fn, args, ema_decay=0.995):

    model.train()

    total_loss = 0

    for batch in loader:

        eeg = batch["eeg_input"].to(DEVICE)
        fnirs = batch["fnirs_input"].to(DEVICE)
        label = batch["label"].to(DEVICE)
        trial_label = batch["trial_label"].to(DEVICE)

        optimizer.zero_grad()

        trial_group = trial_label // args["TRAIL_GROUP"]

        # forward 1
        output1 = model(eeg, fnirs)
        output1["trial_group"] = trial_group

        # forward 2
        output2 = model(eeg, fnirs)
        output2["trial_group"] = trial_group

        # CE loss
        loss1 = loss_fn(output1, label)
        loss2 = loss_fn(output2, label)

        ce_loss = 0.5 * (loss1 + loss2)

        # KL consistency
        logits1 = output1["fusion_logits"]
        logits2 = output2["fusion_logits"]

        kl = (
            F.kl_div(
                F.log_softmax(logits1, dim=1),
                F.softmax(logits2, dim=1),
                reduction="batchmean"
            ) +
            F.kl_div(
                F.log_softmax(logits2, dim=1),
                F.softmax(logits1, dim=1),
                reduction="batchmean"
            )
        ) / 2

        loss = ce_loss + kl

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.
        )

        for p in model.parameters():
            if p.grad is not None:
                p.grad += 0.001 * torch.randn_like(p.grad)

        optimizer.step()
        update_ema(model, ema_model, ema_decay)
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

    ema_model = copy.deepcopy(model)
    ema_model.eval()

    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4
    )

    loss_fn = LossModule()

    for epoch in range(50):
        decay = 0.995 if epoch > 3 else 0
        train_loss = train_epoch(
            model,
            ema_model,
            train_loader,
            optimizer,
            loss_fn,
            args,
            decay
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
