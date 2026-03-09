import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch_sta import STA_NET


############################################
# Dataset
############################################

class BrainDataset(Dataset):

    def __init__(self, eeg, fnirs, label):

        self.eeg = torch.tensor(eeg, dtype=torch.float32)
        self.fnirs = torch.tensor(fnirs, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):

        return (
            self.eeg[idx],
            self.fnirs[idx],
            self.label[idx]
        )


############################################
# evaluation
############################################

def evaluate(model, loader, device):

    model.eval()

    total_loss = 0
    total_correct = 0
    total = 0

    with torch.no_grad():

        for eeg, fnirs, label in loader:

            eeg = eeg.to(device)
            fnirs = fnirs.to(device)
            label = label.to(device)

            pred, eeg_pred = model(eeg, fnirs)

            loss1 = F.cross_entropy(pred, label.argmax(dim=1))
            loss2 = F.cross_entropy(eeg_pred, label.argmax(dim=1))

            loss = loss1 + loss2

            total_loss += loss.item() * eeg.size(0)

            pred_cls = pred.argmax(dim=1)
            true_cls = label.argmax(dim=1)

            total_correct += (pred_cls == true_cls).sum().item()
            total += eeg.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total
    }


############################################
# training
############################################

def train_epoch(model, loader, optimizer, device):

    model.train()

    total_loss = 0
    total = 0

    for eeg, fnirs, label in loader:

        eeg = eeg.to(device)
        fnirs = fnirs.to(device)
        label = label.to(device)

        pred, eeg_pred = model(eeg, fnirs)

        loss1 = F.cross_entropy(pred, label.argmax(dim=1))
        loss2 = F.cross_entropy(eeg_pred, label.argmax(dim=1))

        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * eeg.size(0)
        total += eeg.size(0)

    return total_loss / total


############################################
# main
############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

subject_path = "data/model_input"
subject_list = sorted(os.listdir(subject_path))

for subject in subject_list:

    data = np.load(os.path.join(subject_path, subject))

    eeg = data["eeg"]
    fnirs = data["fnirs"]
    label = data["label"].astype(float)

    fnirs *= 1e3

    for session in range(3):

        print(subject, session)

        test_slice = slice(session*200, (session+1)*200)

        eeg_test = eeg[test_slice]
        fnirs_test = fnirs[test_slice]
        label_test = label[test_slice]

        all_eeg = np.delete(eeg, test_slice, axis=0)
        all_fnirs = np.delete(fnirs, test_slice, axis=0)
        all_label = np.delete(label, test_slice, axis=0)

        np.random.seed(42)
        indices = np.random.choice(all_eeg.shape[0], size=80, replace=False)

        eeg_val = all_eeg[indices]
        fnirs_val = all_fnirs[indices]
        label_val = all_label[indices]

        eeg_train = np.delete(all_eeg, indices, axis=0)
        fnirs_train = np.delete(all_fnirs, indices, axis=0)
        label_train = np.delete(all_label, indices, axis=0)

        train_dataset = BrainDataset(eeg_train, fnirs_train, label_train)
        val_dataset = BrainDataset(eeg_val, fnirs_val, label_val)
        test_dataset = BrainDataset(eeg_test, fnirs_test, label_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)

        model = STA_NET().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        ####################################
        # first stage training
        ####################################

        best_val_loss = float("inf")
        patience = 50
        wait = 0
        target_loss = None

        print("begin first train")

        for epoch in range(300):

            train_loss = train_epoch(model, train_loader, optimizer, device)

            val_metrics = evaluate(model, val_loader, device)

            val_loss = val_metrics["loss"]

            print(
                f"Epoch {epoch} | train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

            if val_loss < best_val_loss:

                best_val_loss = val_loss
                wait = 0

                target_loss = train_loss

                best_state = model.state_dict()

            else:

                wait += 1

                if wait >= patience:
                    print("Early stopping")
                    break

        model.load_state_dict(best_state)

        ####################################
        # second stage training
        ####################################

        print("begin second train")

        full_dataset = BrainDataset(all_eeg, all_fnirs, all_label)
        full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)

        for epoch in range(200):

            train_loss = train_epoch(model, full_loader, optimizer, device)

            print(f"epoch {epoch} loss {train_loss}")

            if train_loss <= target_loss:

                print("Reached target loss value, stopping training")
                break

        ####################################
        # test
        ####################################

        print("begin test")

        test_metrics = evaluate(model, test_loader, device)

        with open("results.txt", "a") as f:

            f.write(
                f'{{"subject": "{subject}", "fold": {session}, '
                f'"result": {test_metrics}}}\n'
            )

print("all done")
