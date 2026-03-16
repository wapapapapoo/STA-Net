import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from eval import evaluate
from train import train
from model import Model

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

            model = Model().to(DEVICE)

            train(model, train_loader, val_loader)

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

if __name__ == "__main__":
    main()
