import torch
import torch.nn as nn
import torch.nn.functional as F

from eval import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LossModule(nn.Module):

    def __init__(self):

        super().__init__()

        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, output, label, epoch):

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

        loss = loss_main + max(0, (epoch - 20) / 30 * 0.3) * loss_session

        return loss




def train_epoch(epoch, model, loader, optimizer, loss_fn, args):
    model.train()
    total_loss = 0
    for batch in loader:
        eeg = batch["eeg_input"].to(DEVICE)
        fnirs = batch["fnirs_input"].to(DEVICE)
        label = batch["label"].to(DEVICE)
        trial_label = batch["trial_label"].to(DEVICE)

        optimizer.zero_grad()

        trial_group = trial_label // args["TRAIL_GROUP"]

        if epoch > 20:
            # forward 1
            output1 = model(eeg, fnirs)
            output1["trial_group"] = trial_group

            # forward 2
            output2 = model(eeg, fnirs)
            output2["trial_group"] = trial_group

            # CE loss
            loss1 = loss_fn(output1, label, epoch)
            loss2 = loss_fn(output2, label, epoch)

            ce_loss = 0.5 * (loss1 + loss2)

            # KL consistency
            logit1 = output1["fusion_logits"]
            logit2 = output2["fusion_logits"]

            def safe_softmax(x, dim=1, eps=1e-6):
                p = F.softmax(x, dim=dim)
                return p.clamp(min=eps, max=1.0)

            kl = (
                F.kl_div(
                    F.log_softmax(logit1, dim=1),
                    safe_softmax(logit2, dim=1),
                    reduction="batchmean"
                ) +
                F.kl_div(
                    F.log_softmax(logit2, dim=1),
                    safe_softmax(logit1, dim=1),
                    reduction="batchmean"
                )
            ) / 2

            loss = ce_loss + max(0, (epoch - 20) / 30 * 0.3) * kl
        
        else:
            output = model(eeg, fnirs)
            output["trial_group"] = trial_group
            loss = loss_fn(output, label, epoch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.
        )

        for p in model.parameters():
            if p.grad is not None:
                p.grad += 0.0001 * torch.randn_like(p.grad)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)



def compute_val_loss(model, loader):

    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

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
            epoch,
            model,
            train_loader,
            optimizer,
            loss_fn,
            args
        )

        train_acc, train_eeg_acc, train_fnirs_acc = evaluate(model, train_loader)

        val_acc, val_eeg_acc, val_fnirs_acc = evaluate(model, val_loader)

        test_acc, test_eeg_acc, test_fnirs_acc = evaluate(model, args['test_loader'])

        val_loss = compute_val_loss(
            model,
            val_loader
        )

        print(
            f"ep {epoch:02d} "
            f"tl {train_loss:.4f} "
            f"vl {val_loss:.4f} "
            f"ta {train_acc:.4f} "
            f"tea {train_eeg_acc:.4f} "
            f"tfa {train_fnirs_acc:.4f} "
            f"va {val_acc:.4f} "
            f"vea {val_eeg_acc:.4f} "
            f"vfa {val_fnirs_acc:.4f} "
            f"Ta {test_acc:.4f} "
            f"Tea {test_eeg_acc:.4f} "
            f"Tfa {test_fnirs_acc:.4f} ", flush=True
        )

    return model
