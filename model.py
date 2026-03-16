import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------
# Gradient Reversal
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
            nn.Dropout(0.3),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):

        x = self.net(x).squeeze(-1)
        x = self.fc(x)

        return x


# ------------------------------------------------
# fNIRS Encoder
# input: (B,72,120)
# ------------------------------------------------

class FNIRSEncoder(nn.Module):

    def __init__(self, embed_dim=128):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(72, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Conv1d(128, embed_dim, 5, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        feat = self.conv(x)           # (B,embed,T)
        embed = self.pool(feat).squeeze(-1)

        return feat, embed


# ------------------------------------------------
# EEG guided fNIRS attention
# ------------------------------------------------

class EEGFNIRSAttention(nn.Module):

    def __init__(self, embed_dim=128):
        super().__init__()

        self.query = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, eeg_embed, fnirs_feat):

        # fnirs_feat: (B,embed,T)

        q = self.query(eeg_embed).unsqueeze(1)       # (B,1,E)

        k = fnirs_feat.transpose(1,2)                # (B,T,E)
        v = k                                        # value = key

        attn = torch.matmul(q, k.transpose(1,2)) * self.scale
        attn = torch.softmax(attn, dim=-1)           # (B,1,T)

        out = torch.matmul(attn, v)                  # (B,1,E)
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
            nn.Dropout(0.5),

            nn.Linear(256, embed_dim)
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
        self.embed_dropout = nn.Dropout(0.3)

        self.attention = EEGFNIRSAttention(embed_dim)

        self.fusion = FusionModule(embed_dim)

        self.embed_dropout = nn.Dropout(0.3)

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
            nn.Dropout(0.4),

            nn.Linear(128, args['TRAIL_GROUP_AMOUNT'])
        )

    def forward(self, eeg, fnirs):

        # ----------------
        # encoders
        # ----------------

        eeg_embed = self.eeg_encoder(eeg)
        eeg_embed = self.embed_dropout(eeg_embed)

        fnirs_feat, fnirs_embed = self.fnirs_encoder(fnirs)
        fnirs_feat = self.embed_dropout(fnirs_feat)

        # ----------------
        # cross modal attention
        # ----------------

        attn_fnirs = self.attention(eeg_embed, fnirs_feat)

        # ----------------
        # fusion
        # ----------------

        eeg_embed = self.embed_dropout(eeg_embed)
        attn_fnirs = self.embed_dropout(attn_fnirs)

        fusion_embed = self.fusion(eeg_embed, attn_fnirs)

        # ----------------
        # classification
        # ----------------

        eeg_logits = self.classifier(eeg_embed)
        fnirs_logits = self.classifier(attn_fnirs)
        fusion_logits = self.classifier(fusion_embed)

        # ----------------
        # GRL (only fusion)
        # ----------------

        rev = grad_reverse(fusion_embed)

        session_logits = self.session_classifier(rev)

        return {

            "logits": fusion_logits,

            "eeg_logits": eeg_logits,
            "fnirs_logits": fnirs_logits,
            "fusion_logits": fusion_logits,

            "session_logits": session_logits,

            "eeg_embed": eeg_embed,
            "fnirs_embed": attn_fnirs,
            "fusion_embed": fusion_embed
        }
