import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# EEG ENCODER
# =====================================================

class EEGEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(28, 64, 15, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, 64, 9, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = self.net(x)

        x = self.pool(x)

        return x.squeeze(-1)
    

























class FNIRSEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(72, 64, 9, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = self.net(x)

        x = self.pool(x)

        return x.squeeze(-1)
















class CrossModalFusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 128)
        )

    def forward(self, eeg, fnirs):

        x = torch.cat([eeg, fnirs], dim=1)

        x = self.fc(x)

        return x




















class Model(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.eeg_encoder = EEGEncoder()

        self.fnirs_encoder = FNIRSEncoder()

        self.fusion = CrossModalFusion()

        self.classifier = nn.Sequential(

            nn.Linear(128, 64),
            nn.GELU(),

            nn.Dropout(0.5),

            nn.Linear(64, 2)
        )

    def forward(self, eeg, fnirs):

        eeg_feat = self.eeg_encoder(eeg)

        fnirs_feat = self.fnirs_encoder(fnirs)

        fused = self.fusion(eeg_feat, fnirs_feat)

        out = self.classifier(fused)

        return out, eeg_feat, fnirs_feat
