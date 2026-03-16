import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# EEG Encoder
# =====================================================

class EEGEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.temporal = nn.Sequential(

            nn.Conv1d(28, 32, 15, padding=7),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.MaxPool1d(2),   # 600 → 300

            nn.Conv1d(32, 64, 9, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.MaxPool1d(2),   # 300 → 150
        )

        self.spatial = nn.Conv1d(64, 64, 1)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = self.temporal(x)

        x = self.spatial(x)

        x = self.pool(x)

        return x.squeeze(-1)
    

# =====================================================
# fNIRS Encoder
# =====================================================

class FNIRSEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.temporal = nn.Sequential(

            nn.Conv1d(72, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = self.temporal(x)

        x = self.pool(x)

        return x.squeeze(-1)


# =====================================================
# Projection Heads (for contrastive learning)
# =====================================================

class ProjectionHead(nn.Module):

    def __init__(self, in_dim=64, proj_dim=32):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(in_dim, 64),
            nn.GELU(),

            nn.Linear(64, proj_dim)
        )

    def forward(self, x):

        return self.net(x)


# =====================================================
# Cross-modal predictor
# EEG → fNIRS
# =====================================================

class CrossModalPredictor(nn.Module):

    def __init__(self, dim=64):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(dim, 64),
            nn.GELU(),

            nn.Linear(64, dim)
        )

    def forward(self, eeg_feat):

        return self.net(eeg_feat)


# =====================================================
# Fusion module
# =====================================================

class Fusion(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(128, 64),
            nn.GELU(),

            nn.Dropout(0.4)
        )

    def forward(self, eeg_feat, fnirs_feat):

        x = torch.cat([eeg_feat, fnirs_feat], dim=1)

        return self.net(x)


# =====================================================
# Full Model
# =====================================================

class Model(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.eeg_encoder = EEGEncoder()

        self.fnirs_encoder = FNIRSEncoder()

        self.eeg_proj = ProjectionHead()

        self.fnirs_proj = ProjectionHead()

        self.cross_predict = CrossModalPredictor()

        self.fusion = Fusion()

        self.classifier = nn.Linear(64, 2)

    def forward(self, eeg, fnirs):

        # encoders
        eeg_feat = self.eeg_encoder(eeg)
        fnirs_feat = self.fnirs_encoder(fnirs)

        # contrastive projection
        eeg_proj = self.eeg_proj(eeg_feat)
        fnirs_proj = self.fnirs_proj(fnirs_feat)

        # cross-modal prediction
        fnirs_pred = self.cross_predict(eeg_feat)

        # fusion classifier
        fused = self.fusion(eeg_feat, fnirs_feat)

        logits = self.classifier(fused)

        return {
            "logits": logits,
            "eeg_feat": eeg_feat,
            "fnirs_feat": fnirs_feat,
            "eeg_proj": eeg_proj,
            "fnirs_proj": fnirs_proj,
            "fnirs_pred": fnirs_pred
        }