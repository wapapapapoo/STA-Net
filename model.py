import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------
# Gradient Reversal Layer
# ------------------------------------------------

class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


# ------------------------------------------------
# EEG Encoder
# input: (B,28,600)
# ------------------------------------------------

class EEGEncoder(nn.Module):

    def __init__(self, embed_dim=128):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(28, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.Dropout(0.3)
        )

    def forward(self, x):

        x = self.net(x).squeeze(-1)

        return self.fc(x)


# ------------------------------------------------
# fNIRS Encoder
# input: (B,72,120)
# ------------------------------------------------

class FNIRSEncoder(nn.Module):

    def __init__(self, embed_dim=128):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(72, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Conv1d(128, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.Dropout(0.3)
        )

    def forward(self, x):

        x = self.net(x).squeeze(-1)

        return self.fc(x)


# ------------------------------------------------
# EEG guided fNIRS attention
# ------------------------------------------------

class EEGFNIRSAttention(nn.Module):

    def __init__(self, embed_dim=128, fnirs_time=120):

        super().__init__()

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Conv1d(72, embed_dim, 1)

        self.scale = embed_dim ** -0.5

    def forward(self, eeg_embed, fnirs_raw):

        q = self.query(eeg_embed).unsqueeze(1)

        k = self.key(fnirs_raw).transpose(1,2)

        attn = torch.matmul(q, k.transpose(1,2)) * self.scale

        attn = torch.softmax(attn, dim=-1)

        v = fnirs_raw.transpose(1,2)

        out = torch.matmul(attn, v)

        out = out.squeeze(1)

        return out


# ------------------------------------------------
# Fusion
# ------------------------------------------------

class FusionModule(nn.Module):

    def __init__(self, embed_dim=128):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(embed_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(256, embed_dim),
            nn.Dropout(0.4)
        )

    def forward(self, eeg, fnirs):

        x = torch.cat([eeg, fnirs], dim=1)

        return self.net(x)


# ------------------------------------------------
# Model
# ------------------------------------------------

class Model(nn.Module):

    def __init__(self, args):

        super().__init__()

        embed_dim = 128

        self.eeg_encoder = EEGEncoder(embed_dim)
        self.fnirs_encoder = FNIRSEncoder(embed_dim)

        self.attention = EEGFNIRSAttention(embed_dim)

        self.fusion = FusionModule(embed_dim)

        # shared classifier

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

        # session classifier

        self.session_classifier = nn.Sequential(

            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(128, args['TRAIL_GROUP_AMOUNT'])
        )

    def forward(self, eeg, fnirs):

        eeg_embed = self.eeg_encoder(eeg)

        fnirs_embed = self.fnirs_encoder(fnirs)

        # attention refined fnirs

        attn_fnirs = self.attention(eeg_embed, fnirs)

        attn_fnirs = self.fnirs_encoder(
            attn_fnirs.unsqueeze(-1).repeat(1,1,120)
        )

        fusion_embed = self.fusion(eeg_embed, attn_fnirs)

        # classification

        eeg_logits = self.classifier(eeg_embed)

        fnirs_logits = self.classifier(attn_fnirs)

        fusion_logits = self.classifier(fusion_embed)

        # GRL

        rev_eeg = grad_reverse(eeg_embed)

        rev_fnirs = grad_reverse(attn_fnirs)

        rev_fusion = grad_reverse(fusion_embed)

        session_eeg = self.session_classifier(rev_eeg)

        session_fnirs = self.session_classifier(rev_fnirs)

        session_fusion = self.session_classifier(rev_fusion)

        return {

            "logits": fusion_logits,

            "eeg_logits": eeg_logits,
            "fnirs_logits": fnirs_logits,
            "fusion_logits": fusion_logits,

            "session_eeg": session_eeg,
            "session_fnirs": session_fnirs,
            "session_fusion": session_fusion,

            "eeg_embed": eeg_embed,
            "fnirs_embed": attn_fnirs,
            "fusion_embed": fusion_embed
        }
