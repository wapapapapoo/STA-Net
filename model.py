import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# EEG ENCODER
# =========================================================

class EEGEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(28, 16, 15, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, 9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(32, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = self.conv(x)

        x = self.pool(x)

        return x.squeeze(-1)


# =========================================================
# FNIRS ENCODER
# =========================================================

class FNIRSEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(72, 16, 9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = self.conv(x)

        x = self.pool(x)

        return x.squeeze(-1)


# =========================================================
# EEG → fNIRS INTERACTION
# =========================================================

class EEGGuidedFNIRS(nn.Module):

    def __init__(self):
        super().__init__()

        self.proj = nn.Linear(32, 32)

        self.gate = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.Sigmoid()
        )

    def forward(self, eeg, fnirs):

        concat = torch.cat([eeg, fnirs], dim=1)

        g = self.gate(concat)

        fnirs = fnirs + g * self.proj(eeg)

        return eeg, fnirs


# =========================================================
# MODEL
# =========================================================

class Model(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.eeg_encoder = EEGEncoder()

        self.fnirs_encoder = FNIRSEncoder()

        self.interaction = EEGGuidedFNIRS()

        self.classifier = nn.Sequential(

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(32, 2)
        )

    def forward(self, eeg, fnirs):

        eeg_feat = self.eeg_encoder(eeg)      # (B,32)

        fnirs_feat = self.fnirs_encoder(fnirs) # (B,32)

        eeg_feat, fnirs_feat = self.interaction(eeg_feat, fnirs_feat)

        fused = torch.cat([eeg_feat, fnirs_feat], dim=1)   # (B,64)

        out = self.classifier(fused)

        return out, fused