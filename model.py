import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# -------------------------------------------------
# Patch Embedding (break temporal memorization)
# -------------------------------------------------

class PatchEmbed(nn.Module):

    def __init__(self, cin, dim=16, patch=20):

        super().__init__()

        self.proj = nn.Conv1d(
            cin,
            dim,
            kernel_size=patch,
            stride=patch
        )

        self.norm = nn.GroupNorm(4, dim)

    def forward(self,x):

        x = self.proj(x)
        x = self.norm(x)

        return F.gelu(x)


# -------------------------------------------------
# Shared temporal encoder
# -------------------------------------------------

class TemporalEncoder(nn.Module):

    def __init__(self, dim=16):

        super().__init__()

        self.dw = nn.Conv1d(
            dim,
            dim,
            kernel_size=5,
            padding=2,
            groups=dim
        )

        self.pw = nn.Conv1d(dim,dim,1)

        self.norm = nn.GroupNorm(4,dim)

        self.drop = nn.Dropout(0.5)

    def forward(self,x):

        res = x

        x = self.dw(x)
        x = self.pw(x)

        x = self.norm(x)

        x = F.gelu(x + res)

        return self.drop(x)


# -------------------------------------------------
# Random temporal crop
# -------------------------------------------------

class RandomCrop(nn.Module):

    def __init__(self, size):

        super().__init__()
        self.size = size

    def forward(self,x):

        if not self.training:
            return x

        T = x.shape[-1]

        if T <= self.size:
            return x

        start = random.randint(0, T-self.size)

        return x[...,start:start+self.size]


# -------------------------------------------------
# Pooling
# -------------------------------------------------

class TokenPool(nn.Module):

    def forward(self,x):

        mean = x.mean(-1)
        std = x.std(-1)

        return torch.cat([mean,std],dim=1)


# -------------------------------------------------
# Fusion
# -------------------------------------------------

class Fusion(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim*2,dim),
            nn.Sigmoid()
        )

    def forward(self,a,b):

        g = self.gate(torch.cat([a,b],dim=1))

        return g*a + (1-g)*b


# -------------------------------------------------
# Classifier
# -------------------------------------------------

class Classifier(nn.Module):

    def __init__(self,dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(dim,8),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Linear(8,2)
        )

    def forward(self,x):

        return self.net(x)


# -------------------------------------------------
# Model
# -------------------------------------------------

class Model(nn.Module):

    def __init__(self, args):

        super().__init__()

        dim = 16

        # random crop
        self.crop_eeg = RandomCrop(500)
        self.crop_fnirs = RandomCrop(100)

        # patch embedding
        self.eeg_patch = PatchEmbed(28,dim,25)
        self.fnirs_patch = PatchEmbed(72,dim,10)

        # shared encoder
        self.encoder = TemporalEncoder(dim)

        # pool
        self.pool = TokenPool()

        emb = dim*2

        # fusion
        self.fusion = Fusion(emb)

        # classifiers
        self.cls_eeg = Classifier(emb)
        self.cls_fnirs = Classifier(emb)
        self.cls_fusion = Classifier(emb)


    def forward(self,eeg,fnirs):

        eeg = self.crop_eeg(eeg)
        fnirs = self.crop_fnirs(fnirs)

        eeg = self.eeg_patch(eeg)
        fnirs = self.fnirs_patch(fnirs)

        eeg = self.encoder(eeg)
        fnirs = self.encoder(fnirs)

        eeg_emb = self.pool(eeg)
        fnirs_emb = self.pool(fnirs)

        fusion_emb = self.fusion(eeg_emb,fnirs_emb)

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