import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# EEG Encoder (EEGNet style)
# =====================================================

class EEGEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.temporal = nn.Conv2d(
            1,
            16,
            (1,15),
            padding=(0,7),
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(16)

        # depthwise spatial conv
        self.spatial = nn.Conv2d(
            16,
            32,
            (28,1),
            groups=16,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(32)

        self.pool = nn.AvgPool2d((1,4))

        self.act = nn.ELU()

    def forward(self, x):

        # x: (B,28,600)

        x = x.unsqueeze(1)  # (B,1,28,600)

        x = self.temporal(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.spatial(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.pool(x)

        x = x.mean(dim=-1)
        x = x.squeeze(-1)

        return x


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
# =====================================================

class FNIRSEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(
            72,
            32,
            7,
            padding=3
        )

        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(
            32,
            64,
            5,
            padding=2
        )

        self.bn2 = nn.BatchNorm1d(64)

        self.att = ChannelAttention(64)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.gelu(x)

        x = self.att(x)

        x = self.pool(x)

        return x.squeeze(-1)


# =====================================================
# Shared Representation
# =====================================================

class SharedRepresentation(nn.Module):

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

        self.fc = nn.Linear(
            dim,
            proj_dim,
            bias=False
        )

    def forward(self, x):

        x = self.bn(x)

        return self.fc(x)


# =====================================================
# Cross Modal Predictor
# =====================================================

class CrossModalPredictor(nn.Module):

    def __init__(self, dim=64):

        super().__init__()

        self.fc1 = nn.Linear(dim, dim)

        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):

        x = F.gelu(self.fc1(x))

        return self.fc2(x)


# =====================================================
# Cross Attention Fusion
# =====================================================

class CrossAttentionFusion(nn.Module):

    def __init__(self, dim=64):

        super().__init__()

        self.q = nn.Linear(dim, dim)

        self.k = nn.Linear(dim, dim)

        self.v = nn.Linear(dim, dim)

        self.scale = dim ** -0.5

    def forward(self, eeg, fnirs):

        q = self.q(eeg)

        k = self.k(fnirs)

        v = self.v(fnirs)

        score = (q * k).sum(-1, keepdim=True)

        attn = torch.softmax(
            score * self.scale,
            dim=0
        )

        fused = eeg + attn * v

        return fused


# =====================================================
# Full Model
# =====================================================

class Model(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.eeg_encoder = EEGEncoder()

        self.fnirs_encoder = FNIRSEncoder()

        self.shared = SharedRepresentation()

        self.eeg_proj = ProjectionHead()

        self.fnirs_proj = ProjectionHead()

        self.cross_predict = CrossModalPredictor()

        self.fusion = CrossAttentionFusion()

        self.classifier = nn.Linear(
            64,
            2
        )

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