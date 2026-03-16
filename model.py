import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------
# temporal compression
# ---------------------------------------

class TemporalReduce(nn.Module):

    def __init__(self, cin, cout):

        super().__init__()

        self.conv = nn.Conv1d(
            cin,
            cout,
            kernel_size=25,
            stride=5,
            padding=12
        )

        self.norm = nn.GroupNorm(4, cout)

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)

        return F.gelu(x)


# ---------------------------------------
# depthwise block
# ---------------------------------------

class DWBlock(nn.Module):

    def __init__(self, c):

        super().__init__()

        self.dw = nn.Conv1d(
            c,
            c,
            kernel_size=7,
            padding=3,
            groups=c
        )

        self.pw = nn.Conv1d(c, c, 1)

        self.norm = nn.GroupNorm(4, c)

    def forward(self, x):

        res = x

        x = self.dw(x)
        x = self.pw(x)

        x = self.norm(x)

        return F.gelu(x + res)


# ---------------------------------------
# encoder
# ---------------------------------------

class Encoder(nn.Module):

    def __init__(self, cin):

        super().__init__()

        hidden = 24

        self.reduce = TemporalReduce(cin, hidden)

        self.block1 = DWBlock(hidden)
        self.block2 = DWBlock(hidden)

        self.drop = nn.Dropout(0.4)

    def forward(self, x):

        x = self.reduce(x)

        x = self.block1(x)
        x = self.block2(x)

        x = self.drop(x)

        return x


# ---------------------------------------
# statistical pooling
# ---------------------------------------

class StatPool(nn.Module):

    def forward(self, x):

        mean = x.mean(-1)
        std = x.std(-1)

        return torch.cat([mean, std], dim=1)


# ---------------------------------------
# fusion
# ---------------------------------------

class Fusion(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, a, b):

        g = self.gate(torch.cat([a, b], dim=1))

        return g * a + (1 - g) * b


# ---------------------------------------
# classifier
# ---------------------------------------

class Classifier(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.fc = nn.Linear(dim, 2)

    def forward(self, x):

        return self.fc(x)


# ---------------------------------------
# model
# ---------------------------------------

class Model(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.eeg_encoder = Encoder(28)
        self.fnirs_encoder = Encoder(72)

        self.pool = StatPool()

        emb = 48  # hidden*2

        self.fusion = Fusion(emb)

        self.cls_eeg = Classifier(emb)
        self.cls_fnirs = Classifier(emb)
        self.cls_fusion = Classifier(emb)


    def forward(self, eeg, fnirs):

        eeg = self.eeg_encoder(eeg)
        fnirs = self.fnirs_encoder(fnirs)

        eeg_emb = self.pool(eeg)
        fnirs_emb = self.pool(fnirs)

        fusion_emb = self.fusion(eeg_emb, fnirs_emb)

        eeg_logits = self.cls_eeg(eeg_emb)
        fnirs_logits = self.cls_fnirs(fnirs_emb)
        fusion_logits = self.cls_fusion(fusion_emb)

        return {

            "logits": fusion_logits,

            "fusion_logits": fusion_logits,
            "eeg_logits": eeg_logits,
            "fnirs_logits": fnirs_logits,

            "eeg_embed": eeg_emb,
            "fnirs_embed": fnirs_emb,
            "fusion_embed": fusion_emb
        }