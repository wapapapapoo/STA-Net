import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 25
FOLD = 3
DATA_PATH = "data/model_input_2d"
RESULT_FILE = "results.txt"


# =========================================================
# Dataset
# =========================================================
class EEGFNIRSDataset(Dataset):
    def __init__(self, eeg, fnirs, label):
        # EEG
        eeg = np.squeeze(eeg, axis=-1)     # (N,28,600)

        # fNIRS
        N = fnirs.shape[0]

        # (N,11,36,30,2) → (N,11,72,30)
        fnirs = fnirs.reshape(N, 11, 36 * 2, 30)

        # (N,11,72,30) → (N,72,330)
        fnirs = fnirs.transpose(0,2,1,3).reshape(N, 72, 11 * 30)

        self.eeg = torch.tensor(eeg, dtype=torch.float32)
        self.fnirs = torch.tensor(fnirs, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return self.eeg.shape[0]

    def __getitem__(self, idx):
        return {
            "eeg_input": self.eeg[idx],
            "fnirs_input": self.fnirs[idx],
            "label": self.label[idx]
        }


# =========================================================
# Split
# =========================================================

def build_split(eeg, fnirs, label, session):

    fold_size = 600 // FOLD

    test_start = session * fold_size
    test_end = (session + 1) * fold_size

    eeg_test = eeg[test_start:test_end]
    fnirs_test = fnirs[test_start:test_end]
    label_test = label[test_start:test_end]

    all_eeg = np.delete(eeg, slice(test_start, test_end), axis=0)
    all_fnirs = np.delete(fnirs, slice(test_start, test_end), axis=0)
    all_label = np.delete(label, slice(test_start, test_end), axis=0)

    if session == 0:
        indices = np.arange(0, 100)

    elif session == FOLD - 1:
        indices = np.arange(300, 400)

    else:
        boundary = test_start
        indices = np.concatenate([
            np.arange(boundary - 50, boundary),
            np.arange(boundary, boundary + 50)
        ])

    eeg_val = all_eeg[indices]
    fnirs_val = all_fnirs[indices]
    label_val = all_label[indices]

    eeg_train = np.delete(all_eeg, indices, axis=0)
    fnirs_train = np.delete(all_fnirs, indices, axis=0)
    label_train = np.delete(all_label, indices, axis=0)

    return (
        eeg_train, fnirs_train, label_train,
        eeg_val, fnirs_val, label_val,
        eeg_test, fnirs_test, label_test
    )


# =========================================================
# Dummy model (你自己替换)
# =========================================================

class DummyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.eeg_net = nn.Sequential(
            nn.Conv1d(28, 64, 7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fnirs_net = nn.Sequential(
            nn.Conv1d(72, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(128, 2)

    def forward(self, eeg, fnirs):

        eeg = self.eeg_net(eeg).squeeze(-1)
        fnirs = self.fnirs_net(fnirs).squeeze(-1)

        x = torch.cat([eeg, fnirs], dim=1)

        return self.fc(x)


# =========================================================
# train
# =========================================================

def train(model, loader, optimizer, criterion):

    model.train()

    total_loss = 0

    for batch in loader:

        eeg = batch["eeg_input"].to(DEVICE)
        fnirs = batch["fnirs_input"].to(DEVICE)
        label = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        output = model(eeg, fnirs)

        target = torch.argmax(label, dim=1)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================================================
# evaluate
# =========================================================

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

            pred = torch.argmax(output, dim=1)
            gt = torch.argmax(label, dim=1)

            correct += (pred == gt).sum().item()
            total += gt.shape[0]

    return correct / total


# =========================================================
# main
# =========================================================

def main():
    subject_list = sorted(os.listdir(DATA_PATH))
    all_results = []

    for subject in subject_list:

        print("Subject:", subject)

        data = np.load(os.path.join(DATA_PATH, subject))

        eeg = data["eeg"]
        fnirs = data["fnirs"]
        label = data["label"].astype(np.float32)

        fnirs *= 1e3

        for session in range(FOLD):

            print("Fold:", session)

            (
                eeg_train, fnirs_train, label_train,
                eeg_val, fnirs_val, label_val,
                eeg_test, fnirs_test, label_test
            ) = build_split(eeg, fnirs, label, session)

            train_dataset = EEGFNIRSDataset(eeg_train, fnirs_train, label_train)
            val_dataset = EEGFNIRSDataset(eeg_val, fnirs_val, label_val)
            test_dataset = EEGFNIRSDataset(eeg_test, fnirs_test, label_test)

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

            model = DummyModel().to(DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(20):

                train_loss = train(model, train_loader, optimizer, criterion)

                val_acc = evaluate(model, val_loader)

                print(f"epoch {epoch} loss {train_loss:.4f} val {val_acc:.4f}")

            test_acc = evaluate(model, test_loader)

            print("Test:", test_acc)

            all_results.append((subject, session, test_acc))

            with open(RESULT_FILE, "a") as f:

                f.write(
                    str({
                        "subject": subject,
                        "fold": session,
                        "test_acc": float(test_acc)
                    }) + "\n"
                )

    # ==============================
    # Summary
    # ==============================

    print("\n===== FINAL RESULTS =====")

    subject_scores = {}

    for subject, fold, acc in all_results:
        if subject not in subject_scores:
            subject_scores[subject] = []
        subject_scores[subject].append(acc)

    all_acc = []

    for subject in subject_scores:
        scores = subject_scores[subject]
        mean_acc = np.mean(scores)
        all_acc.extend(scores)
        print(f"{subject} mean_acc = {mean_acc:.4f} folds = {scores}")
    print("\nOverall mean accuracy:", np.mean(all_acc))