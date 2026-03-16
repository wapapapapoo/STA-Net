import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# -------------------------------------------------
# Depthwise Separable Block (lighter)
# -------------------------------------------------

class DSBlock(nn.Module):

    def __init__(self, cin, cout, k, stride=1, drop=0.2):
        super().__init__()

        self.dw = nn.Conv1d(
            cin, cin,
            kernel_size=k,
            stride=stride,
            padding=k//2,
            groups=cin
        )

        self.pw = nn.Conv1d(cin, cout, 1)

        self.norm = nn.GroupNorm(8, cout)
        self.act = nn.GELU()

        self.drop = nn.Dropout(drop)

    def forward(self,x):

        x = self.dw(x)
        x = self.pw(x)

        x = self.norm(x)
        x = self.act(x)

        return self.drop(x)



# -------------------------------------------------
# EEG Encoder
# -------------------------------------------------

class EEGEncoder(nn.Module):

    def __init__(self,c=64):

        super().__init__()

        self.net = nn.Sequential(

            DSBlock(28,c,31),         # (B,64,600)
            DSBlock(c,c,31,stride=2), # (B,64,300)
            DSBlock(c,c,15,stride=2), # (B,64,150)
            DSBlock(c,c,15)           # (B,64,150)
        )

    def forward(self,x):

        return self.net(x)



# -------------------------------------------------
# FNIRS Encoder
# -------------------------------------------------

class FNIRSEncoder(nn.Module):

    def __init__(self,c=64):

        super().__init__()

        self.net = nn.Sequential(

            DSBlock(72,c,9),          # (B,64,120)
            DSBlock(c,c,9,stride=2),  # (B,64,60)
            DSBlock(c,c,7),
            DSBlock(c,c,7)
        )

    def forward(self,x):

        return self.net(x)



# -------------------------------------------------
# Temporal Align
# -------------------------------------------------

class TemporalAlign(nn.Module):

    def __init__(self,T=120):
        super().__init__()
        self.T = T

    def forward(self,x):

        return F.interpolate(
            x,
            size=self.T,
            mode="linear",
            align_corners=False
        )



# -------------------------------------------------
# Gated Fusion (stable)
# -------------------------------------------------

class GatedFusion(nn.Module):

    def __init__(self,c=64):

        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv1d(c*2,c,1),
            nn.Sigmoid()
        )

        self.norm = nn.GroupNorm(8,c)

    def forward(self,eeg,fnirs):

        g = self.gate(torch.cat([eeg,fnirs],dim=1))

        x = g*eeg + (1-g)*fnirs

        return self.norm(x)



# -------------------------------------------------
# Global Pool
# -------------------------------------------------

class TemporalPool(nn.Module):

    def forward(self,x):

        return x.mean(-1)



# -------------------------------------------------
# Classifier
# -------------------------------------------------

class Classifier(nn.Module):

    def __init__(self,dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(dim,64),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Linear(64,32),
            nn.GELU(),

            nn.Linear(32,2)
        )

    def forward(self,x):

        return self.net(x)



# -------------------------------------------------
# GRL
# -------------------------------------------------

class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx,x,alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx,grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x,alpha=1):
    return GradReverse.apply(x,alpha)



# -------------------------------------------------
# Model
# -------------------------------------------------

class Model(nn.Module):

    def __init__(self,args):

        super().__init__()

        c = 64

        self.eeg = EEGEncoder(c)
        self.fnirs = FNIRSEncoder(c)

        self.align = TemporalAlign(120)

        self.fusion = GatedFusion(c)

        self.pool = TemporalPool()

        self.classifier = Classifier(c)
        self.classifier_eeg = Classifier(c)
        self.classifier_fnirs = Classifier(c)

        self.session_classifier = nn.Sequential(
            nn.Linear(c,64),
            nn.GELU(),
            nn.Linear(64,args['TRAIL_GROUP_AMOUNT'])
        )

    def modality_dropout(self,eeg,fnirs):

        if not self.training:
            return eeg,fnirs

        if random.random() < 0.3:
            eeg = torch.zeros_like(eeg)

        if random.random() < 0.3:
            fnirs = torch.zeros_like(fnirs)

        return eeg,fnirs


    def forward(self,eeg,fnirs):

        # -----------------------
        # encode
        # -----------------------

        eeg = self.eeg(eeg)      # (B,64,150)
        fnirs = self.fnirs(fnirs) # (B,64,60)

        # -----------------------
        # temporal align
        # -----------------------

        eeg = self.align(eeg)
        fnirs = self.align(fnirs)

        # -----------------------
        # modality dropout
        # -----------------------

        eeg,fnirs = self.modality_dropout(eeg,fnirs)

        # -----------------------
        # fusion
        # -----------------------

        fusion = self.fusion(eeg,fnirs)

        # -----------------------
        # embedding
        # -----------------------

        eeg_emb = self.pool(eeg)
        fnirs_emb = self.pool(fnirs)
        fusion_emb = self.pool(fusion)

        # -----------------------
        # classification
        # -----------------------

        eeg_logits = self.classifier_eeg(eeg_emb)
        fnirs_logits = self.classifier_fnirs(fnirs_emb)
        fusion_logits = self.classifier(fusion_emb)

        # -----------------------
        # GRL
        # -----------------------

        rev = grad_reverse(fusion_emb)

        session_logits = self.session_classifier(rev)

        return {

            "logits": fusion_logits,

            "fusion_logits": fusion_logits,
            "eeg_logits": eeg_logits,
            "fnirs_logits": fnirs_logits,

            "session_logits": session_logits,

            "eeg_embed": eeg_emb,
            "fnirs_embed": fnirs_emb,
            "fusion_embed": fusion_emb
        }