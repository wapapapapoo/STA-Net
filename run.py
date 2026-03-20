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
    def __init__(self, eeg, fnirs, label, windows_per_trial=10):

        eeg = np.squeeze(eeg, axis=-1)     # (N,28,600)

        N = fnirs.shape[0]

        # fnirs: (N,11,36,30,2)

        # 1. remove overlap windows
        fnirs = fnirs[:, ::3]              # (N,4,36,30,2)

        # 2. merge HbO HbR
        fnirs = fnirs.reshape(N, 4, 36*2, 30)   # (N,4,72,30)

        # 3. move channel forward
        fnirs = fnirs.transpose(0,2,1,3)        # (N,72,4,30)

        # 4. flatten time
        fnirs = fnirs.reshape(N, 72, 4*30)      # (N,72,120)

        # trial id
        trial_id = np.arange(N) // windows_per_trial

        self.eeg = torch.tensor(eeg, dtype=torch.float32)
        self.fnirs = torch.tensor(fnirs, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)

        self.trial_label = torch.tensor(trial_id, dtype=torch.long)

    def __len__(self):
        return self.eeg.shape[0]

    def __getitem__(self, idx):
        return {
            "eeg_input": self.eeg[idx],
            "fnirs_input": self.fnirs[idx],
            "label": self.label[idx],
            "trial_label": self.trial_label[idx]
        }
    



# =========================================================
# Split
# =========================================================

def build_split(eeg, fnirs, label, session, n_trials=60, windows_per_trial=10):

    trials_per_fold = n_trials // FOLD

    test_start = session * trials_per_fold
    test_end = (session + 1) * trials_per_fold

    trial_ids = np.arange(n_trials)

    test_trials = trial_ids[test_start:test_end]
    train_trials = np.setdiff1d(trial_ids, test_trials)

    # -------- validation 固定抽取策略 --------
    n = len(train_trials)

    front = train_trials[:3]
    middle_start = n // 2 - 2
    middle = train_trials[middle_start:middle_start + 4]
    back = train_trials[-3:]

    val_trials = np.concatenate([front, middle, back])

    train_trials = np.setdiff1d(train_trials, val_trials)
    # ---------------------------------------

    # trial → window index
    def trials_to_indices(trials, selected_offsets=None):
        idx = []
        for t in trials:
            start = t * windows_per_trial
            if selected_offsets is None:
                idx.extend(range(start, start + windows_per_trial))
            else:
                for off in selected_offsets:
                    if off < windows_per_trial:
                        idx.append(start + off)
        return np.array(idx)

    # train 只取 0s / 3s / 6s / 9s
    train_idx = trials_to_indices(train_trials, selected_offsets=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    val_idx = trials_to_indices(val_trials)
    test_idx = trials_to_indices(test_trials)

    eeg_train = eeg[train_idx]
    fnirs_train = fnirs[train_idx]
    label_train = label[train_idx]

    eeg_val = eeg[val_idx]
    fnirs_val = fnirs[val_idx]
    label_val = label[val_idx]

    eeg_test = eeg[test_idx]
    fnirs_test = fnirs[test_idx]
    label_test = label[test_idx]

    return (
        eeg_train, fnirs_train, label_train,
        eeg_val, fnirs_val, label_val,
        eeg_test, fnirs_test, label_test
    )


# =========================================================
# main
# =========================================================

def main():
    args = {
        'TRAIL_GROUP': 20,
        'TRAIL_GROUP_AMOUNT': 20,
    }
    print("; model sparams")
    for key in args:
        print(f"; {key}: {args[key]}")

    model_example = Model(args).to(DEVICE)
    total_params = 0
    trainable_params = 0
    print("; model size summary")
    for name, param in model_example.named_parameters():
        num = param.numel()
        total_params += num
        if param.requires_grad:
            trainable_params += num
        print(f"; {name:40s} {num:10d}")
    print(f"; total: {total_params}, trainable: {trainable_params}")
    del model_example

    subject_list = sorted(os.listdir(DATA_PATH))
    all_results = []

    for subject in subject_list:

        data = np.load(os.path.join(DATA_PATH, subject))

        eeg = data["eeg"]
        fnirs = data["fnirs"]
        label = data["label"].astype(np.float32)

        fnirs *= 1e3

        for session in range(FOLD):

            print(f"# subject: {subject}, session: {session}")

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
            nargs = {
                **args, 'test_loader': test_loader
            }

            model = Model(args).to(DEVICE)

            train(model, train_loader, val_loader, nargs)

            test_acc, test_eeg_acc, test_fnirs_acc = evaluate(model, test_loader)
            all_results.append((subject, session, test_acc, test_eeg_acc, test_fnirs_acc))

    # ==============================
    # Summary
    # ==============================

    print("; ===== FINAL RESULTS =====")

    subject_scores = {}
    subject_scores_eeg = {}
    subject_scores_fnirs = {}

    for subject, fold, acc, acc_eeg, acc_fnirs in all_results:
        if subject not in subject_scores:
            subject_scores[subject] = []
        subject_scores[subject].append(acc)

        if subject not in subject_scores_eeg:
            subject_scores_eeg[subject] = []
        subject_scores_eeg[subject].append(acc_eeg)

        if subject not in subject_scores_fnirs:
            subject_scores_fnirs[subject] = []
        subject_scores_fnirs[subject].append(acc_fnirs)

    all_acc = []
    all_acc_eeg = []
    all_acc_fnirs = []
    for subject in subject_scores:
        scores = subject_scores[subject]
        scores_eeg = subject_scores_eeg[subject]
        scores_fnirs = subject_scores_fnirs[subject]
        mean_acc = np.mean(scores)
        mean_acc_eeg = np.mean(scores_eeg)
        mean_acc_fnirs = np.mean(scores_fnirs)
        all_acc.extend(scores)
        all_acc_eeg.extend(scores_eeg)
        all_acc_fnirs.extend(scores_fnirs)
        print(f"; sbj {subject} a {mean_acc:.4f} ea {mean_acc_eeg:.4f} fa {mean_acc_fnirs:.4f} "
              f"| {scores} {scores_eeg} {scores_fnirs}")
    print(f"; fusion: {np.mean(all_acc):.4f} ({np.std(all_acc):.4f}),"
          f" eeg: {np.mean(all_acc_eeg):.4f} ({np.std(all_acc_eeg):.4f}),"
          f" fnirs: {np.mean(all_acc_fnirs):.4f} ({np.std(all_acc_fnirs):.4f})")

if __name__ == "__main__":
    main()
