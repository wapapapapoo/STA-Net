import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in loader:

            eeg = batch["eeg_input"].to(DEVICE)
            fnirs = batch["fnirs_input"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            output = model(eeg, fnirs)

            logits = output["logits"]

            pred = torch.argmax(logits, dim=1)

            gt = torch.argmax(label, dim=1)

            correct += (pred == gt).sum().item()

            total += gt.size(0)

    if total == 0:
        return 0.0

    return correct / total