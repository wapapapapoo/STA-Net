import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# EEG Encoder
# input : (B,28,600)
# output: (B,64)
# =====================================================

class EEGEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.temporal = nn.Sequential(

            nn.Conv1d(28, 32, 15, padding=7),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 9, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.MaxPool1d(2),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = self.temporal(x)

        x = self.pool(x)

        return x.squeeze(-1)


# =====================================================
# Channel Attention
# =====================================================

class ChannelAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.fc1 = nn.Linear(channels, channels // 4)
        self.fc2 = nn.Linear(channels // 4, channels)

    def forward(self, x):

        # x: (B,C,T)

        w = x.mean(dim=2)

        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))

        return x * w.unsqueeze(-1)


# =====================================================
# fNIRS Encoder
# input : (B,72,120)
# output: (B,64)
# =====================================================

class FNIRSEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(72, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.att = ChannelAttention(64)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = self.conv(x)

        x = self.att(x)

        x = self.pool(x)

        return x.squeeze(-1)


# =====================================================
# Shared Representation
# =====================================================

class SharedBlock(nn.Module):

    def __init__(self, dim=64):

        super().__init__()

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):

        h = F.gelu(self.fc1(x))

        h = self.fc2(h)

        return x + h


# =====================================================
# Projection Head
# =====================================================

class ProjectionHead(nn.Module):

    def __init__(self, dim=64, proj_dim=32):

        super().__init__()

        self.bn = nn.BatchNorm1d(dim)

        self.fc = nn.Linear(dim, proj_dim, bias=False)

    def forward(self, x):

        x = self.bn(x)

        return self.fc(x)


# =====================================================
# Cross Modal Predictor
# =====================================================

class CrossModalPredictor(nn.Module):

    def __init__(self, dim=64):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(dim, dim),
            nn.GELU(),

            nn.Linear(dim, dim)
        )

    def forward(self, x):

        return self.net(x)


# =====================================================
# Fusion
# =====================================================

class Fusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(128, 64),
            nn.GELU(),

            nn.Dropout(0.3)
        )

    def forward(self, eeg, fnirs):

        x = torch.cat([eeg, fnirs], dim=1)

        return self.net(x)


# =====================================================
# Full Model
# =====================================================

class Model(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.eeg_encoder = EEGEncoder()

        self.fnirs_encoder = FNIRSEncoder()

        self.shared = SharedBlock()

        self.eeg_proj = ProjectionHead()

        self.fnirs_proj = ProjectionHead()

        self.cross_predict = CrossModalPredictor()

        self.fusion = Fusion()

        self.classifier = nn.Linear(64, 2)

    def forward(self, eeg, fnirs):

        eeg_feat = self.eeg_encoder(eeg)

        fnirs_feat = self.fnirs_encoder(fnirs)

        eeg_feat = self.shared(eeg_feat)

        fnirs_feat = self.shared(fnirs_feat)

        eeg_proj = self.eeg_proj(eeg_feat)

        fnirs_proj = self.fnirs_proj(fnirs_feat)

        fnirs_pred = self.cross_predict(eeg_feat)

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