import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# -------------------------------------------------
# Signal augmentation
# -------------------------------------------------

def augment_signal(x):

    if random.random() < 0.5:
        x = x + torch.randn_like(x) * 0.05

    if random.random() < 0.5:
        shift = torch.randint(-15, 15, (1,)).item()
        x = torch.roll(x, int(shift), dims=-1)

    if random.random() < 0.3:
        ch = torch.rand(x.shape[1], device=x.device) < 0.2
        x[:, ch, :] = 0

    return x


# -------------------------------------------------
# Very small temporal encoder
# -------------------------------------------------

class TinyEncoder(nn.Module):

    def __init__(self, cin, hidden=32):

        super().__init__()

        self.conv1 = nn.Conv1d(cin, hidden, 9, padding=4)
        self.conv2 = nn.Conv1d(hidden, hidden, 9, padding=4)

        self.norm = nn.GroupNorm(4, hidden)

    def forward(self, x):

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        x = self.norm(x)

        return x


# -------------------------------------------------
# Global temporal pooling
# -------------------------------------------------

class GlobalPool(nn.Module):

    def forward(self, x):

        # (B,C,T) → (B,C)

        mean = x.mean(-1)
        std = x.std(-1)

        return torch.cat([mean, std], dim=1)


# -------------------------------------------------
# Fusion module
# -------------------------------------------------

class Fusion(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, eeg, fnirs):

        g = self.gate(torch.cat([eeg, fnirs], dim=1))

        fused = g * eeg + (1 - g) * fnirs

        return fused


# -------------------------------------------------
# Classifier
# -------------------------------------------------

class Classifier(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.fc = nn.Linear(dim, 2)

    def forward(self, x):

        return self.fc(x)


# -------------------------------------------------
# Main model
# -------------------------------------------------

class Model(nn.Module):

    def __init__(self, args):

        super().__init__()

        hidden = 32

        self.eeg_encoder = TinyEncoder(28, hidden)
        self.fnirs_encoder = TinyEncoder(72, hidden)

        self.pool = GlobalPool()

        emb_dim = hidden * 2

        self.fusion = Fusion(emb_dim)

        self.cls_eeg = Classifier(emb_dim)
        self.cls_fnirs = Classifier(emb_dim)
        self.cls_fusion = Classifier(emb_dim)


    def forward(self, eeg, fnirs):

        if self.training:

            eeg = augment_signal(eeg)
            fnirs = augment_signal(fnirs)

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

            "eeg_logits": eeg_logits,
            "fnirs_logits": fnirs_logits,
            "fusion_logits": fusion_logits,

            "eeg_embed": eeg_emb,
            "fnirs_embed": fnirs_emb,
            "fusion_embed": fusion_emb
        }