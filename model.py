import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # temporal feature extraction
        self.temporal = nn.Sequential(
            nn.Conv1d(28, 64, kernel_size=25, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # channel mixing
        self.channel_mix = nn.Conv1d(64, 64, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = self.temporal(x)

        x = self.channel_mix(x)

        x = self.pool(x)

        return x.squeeze(-1)


class FNIRSEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Conv1d(72, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.channel_mix = nn.Conv1d(64, 64, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = self.temporal(x)

        x = self.channel_mix(x)

        x = self.pool(x)

        return x.squeeze(-1)


class Model(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.eeg_encoder = EEGEncoder()
        self.fnirs_encoder = FNIRSEncoder()

        # modality gating
        self.gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, eeg, fnirs):

        eeg_feat = self.eeg_encoder(eeg)
        fnirs_feat = self.fnirs_encoder(fnirs)

        feat = torch.cat([eeg_feat, fnirs_feat], dim=1)

        gate = self.gate(feat)

        eeg_w = gate[:, 0].unsqueeze(1)
        fnirs_w = gate[:, 1].unsqueeze(1)

        eeg_feat = eeg_feat * eeg_w
        fnirs_feat = fnirs_feat * fnirs_w

        fused = torch.cat([eeg_feat, fnirs_feat], dim=1)

        out = self.classifier(fused)

        return out
