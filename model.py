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

            # block1
            nn.Conv1d(28, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.MaxPool1d(2),   # 600 → 300

            # block2
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(2),   # 300 → 150

            # block3
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.channel_mix = nn.Conv1d(64, 64, 1)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = self.net(x)

        x = self.channel_mix(x)

        x = self.pool(x)

        return x.squeeze(-1)


# =====================================================
# FNIRS ENCODER
# =====================================================

class FNIRSEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.temporal = nn.Sequential(

            nn.Conv1d(72, 32, 9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.channel_mix = nn.Conv1d(64, 64, 1)

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

        # EEG -> fNIRS spatial coupling
        self.spatial_proj = nn.Linear(64, 64)

        # modality reliability gate
        self.gate = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )

    def forward(self, eeg, fnirs):

        eeg_feat = self.eeg_encoder(eeg)
        fnirs_feat = self.fnirs_encoder(fnirs)

        # spatial coupling
        eeg_to_fnirs = self.spatial_proj(eeg_feat)
        fnirs_feat = fnirs_feat + eeg_to_fnirs

        feat = torch.cat([eeg_feat, fnirs_feat], dim=1)

        gate = self.gate(feat)

        eeg_w = gate[:, 0].unsqueeze(1)
        fnirs_w = gate[:, 1].unsqueeze(1)

        eeg_feat = eeg_feat * eeg_w
        fnirs_feat = fnirs_feat * fnirs_w

        fused = torch.cat([eeg_feat, fnirs_feat], dim=1)

        out = self.classifier(fused)

        return out