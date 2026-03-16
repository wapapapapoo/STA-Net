import torch
import torch.nn as nn
import torch.nn.functional as F




class ChannelAttention(nn.Module):

    def __init__(self, c):
        super().__init__()

        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(c, c // 4, 1),
            nn.ReLU(),
            nn.Conv1d(c // 4, c, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        w = self.net(x)

        return x * w + x













class EEGEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(28, 32, 25, padding=12),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64,64,7,padding=6,dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.channel_attn = ChannelAttention(64)

        self.temporal_attn = nn.Conv1d(64, 1, 1)

    def forward(self, x):

        x = self.net(x)

        x = self.channel_attn(x)

        score = self.temporal_attn(x)

        weight = torch.softmax(score / 0.5, dim=2)

        feat = (x * weight).sum(dim=2)

        return feat













class FNIRSEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(72, 32, 9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.channel_attn = ChannelAttention(64)

        self.temporal_attn = nn.Conv1d(64, 1, 1)

    def forward(self, x):

        x = self.net(x)

        x = self.channel_attn(x)

        score = self.temporal_attn(x)

        feat = x.mean(dim=2)

        return feat









class CrossInteraction(nn.Module):

    def __init__(self):
        super().__init__()

        self.eeg_from_fnirs = nn.Linear(64,64)
        self.fnirs_from_eeg = nn.Linear(64,64)

        self.gate_eeg = nn.Sequential(
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.Sigmoid()
        )

        self.gate_fnirs = nn.Sequential(
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.Sigmoid()
        )

    def forward(self,eeg,fnirs):

        concat = torch.cat([eeg,fnirs],dim=1)

        g1 = self.gate_eeg(concat)
        g2 = self.gate_fnirs(concat)

        eeg_new = eeg + g1 * self.eeg_from_fnirs(fnirs)
        fnirs_new = fnirs + g2 * self.fnirs_from_eeg(eeg)

        return eeg_new,fnirs_new









class ReliabilityFusion(nn.Module):

    def __init__(self):
        super().__init__()

        self.eeg_rel = nn.Sequential(
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid()
        )

        self.fnirs_rel = nn.Sequential(
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid()
        )

        self.res = nn.Linear(128,128)

    def forward(self,eeg,fnirs):

        r_eeg = self.eeg_rel(eeg)
        r_fnirs = self.fnirs_rel(fnirs)

        eeg = eeg * r_eeg
        fnirs = fnirs * r_fnirs

        fused = torch.cat([eeg,fnirs],dim=1)

        fused = fused + self.res(fused)

        return fused











class Model(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.eeg_encoder = EEGEncoder()
        self.fnirs_encoder = FNIRSEncoder()

        self.cross = CrossInteraction()

        self.fusion = ReliabilityFusion()

        self.classifier = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64,32),
            nn.ReLU(),

            nn.Linear(32,2)
        )

    def forward(self, eeg, fnirs):

        eeg_feat = self.eeg_encoder(eeg)
        fnirs_feat = self.fnirs_encoder(fnirs)

        eeg_feat, fnirs_feat = self.cross(eeg_feat, fnirs_feat)

        fused = self.fusion(eeg_feat, fnirs_feat)

        out = self.classifier(fused)

        return out, fused
