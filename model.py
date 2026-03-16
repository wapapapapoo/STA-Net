import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# Temporal Filter Block
# (depthwise convolution)
# =====================================================

class TemporalFilter(nn.Module):

    def __init__(self, channels, kernel):

        super().__init__()

        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel,
            padding=kernel//2,
            groups=channels
        )

        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        return F.gelu(x)


# =====================================================
# Channel Mixing
# =====================================================

class ChannelMix(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.conv = nn.Conv1d(in_ch, out_ch, 1)

        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        return F.gelu(x)


# =====================================================
# EEG Encoder
# input: (B,28,600)
# =====================================================

class EEGEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.temp1 = TemporalFilter(28, 25)

        self.mix = ChannelMix(28, 32)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = self.temp1(x)

        x = self.mix(x)

        x = self.pool(x)

        return x.squeeze(-1)


# =====================================================
# fNIRS Encoder
# input: (B,72,120)
# =====================================================

class FNIRSEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.temp = TemporalFilter(72, 9)

        self.mix = ChannelMix(72, 32)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        x = self.temp(x)

        x = self.mix(x)

        x = self.pool(x)

        return x.squeeze(-1)


# =====================================================
# Cross Modal Gating
# EEG → fNIRS
# =====================================================

class CrossModalGate(nn.Module):

    def __init__(self, dim=32):

        super().__init__()

        self.fc = nn.Linear(dim, dim)

    def forward(self, eeg, fnirs):

        g = torch.sigmoid(self.fc(eeg))

        fnirs = fnirs * g

        return fnirs


# =====================================================
# Fusion
# =====================================================

class Fusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc = nn.Sequential(

            nn.Linear(64, 32),

            nn.GELU(),

            nn.Dropout(0.2)
        )

    def forward(self, eeg, fnirs):

        x = torch.cat([eeg, fnirs], dim=1)

        return self.fc(x)


# =====================================================
# Model
# =====================================================

class Model(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.eeg_encoder = EEGEncoder()

        self.fnirs_encoder = FNIRSEncoder()

        self.cross = CrossModalGate()

        self.fusion = Fusion()

        self.classifier = nn.Linear(32, 2)

    def forward(self, eeg, fnirs):

        eeg_feat = self.eeg_encoder(eeg)

        fnirs_feat = self.fnirs_encoder(fnirs)

        fnirs_feat = self.cross(eeg_feat, fnirs_feat)

        fused = self.fusion(eeg_feat, fnirs_feat)

        logits = self.classifier(fused)

        return {

            "logits": logits,

            "eeg_feat": eeg_feat,

            "fnirs_feat": fnirs_feat
        }