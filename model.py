import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================
# Temporal Attention Pooling
# ======================================================

class TemporalAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Conv1d(channels, channels // 2, 1),
            nn.Tanh(),
            nn.Conv1d(channels // 2, 1, 1)
        )

    def forward(self, x):

        score = self.attn(x)          # (B,1,T)
        weight = torch.softmax(score, dim=2)

        feat = (x * weight).sum(dim=2)

        return feat


# ======================================================
# EEG Encoder
# ======================================================

class EEGEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(28, 32, 25, padding=12),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.MaxPool1d(2),  # 600 → 300

            nn.Conv1d(32, 64, 15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(2),  # 300 → 150

            nn.Conv1d(64, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.pool = TemporalAttention(64)

    def forward(self, x):

        x = self.net(x)

        feat = self.pool(x)

        return feat


# ======================================================
# fNIRS Encoder
# ======================================================

class FNIRSEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(72, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.pool = TemporalAttention(64)

    def forward(self, x):

        x = self.net(x)

        feat = self.pool(x)

        return feat


# ======================================================
# Model
# ======================================================

class Model(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.eeg_encoder = EEGEncoder()
        self.fnirs_encoder = FNIRSEncoder()

        # EEG -> fNIRS gating
        self.cross_gate = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(self, eeg, fnirs):

        eeg_feat = self.eeg_encoder(eeg)
        fnirs_feat = self.fnirs_encoder(fnirs)

        # EEG guide fNIRS
        gate = self.cross_gate(eeg_feat)

        fnirs_feat = fnirs_feat * gate

        fused = torch.cat([eeg_feat, fnirs_feat], dim=1)

        fused = self.fusion(fused)

        out = self.classifier(fused)

        return out, eeg_feat, fnirs_feat