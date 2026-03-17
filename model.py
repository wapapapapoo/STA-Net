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
# EEG Encoder (small + stable)
# ------------------------------------------------

class EEGEncoder(nn.Module):
    def __init__(self, in_ch=28, hidden=64, out_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),

            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x):
        # x: [B, 28, 600]
        x = self.net(x).squeeze(-1)
        x = self.fc(x)
        return x  # [B, d]


# ------------------------------------------------
# fNIRS Encoder (large kernel → slow dynamics)
# ------------------------------------------------

class FNIRSEncoder(nn.Module):
    def __init__(self, in_ch=72, hidden=64, out_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),

            nn.Conv1d(hidden, hidden, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        # x: [B, 72, 120]
        x = self.conv(x)              # [B, hidden, T]
        x = x.transpose(1, 2)         # [B, T, hidden]
        x = self.proj(x)              # [B, T, d]
        return x


# ------------------------------------------------
# FiLM Modulation (EEG → fNIRS)
# ------------------------------------------------

class FiLM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)

    def forward(self, cond, x):
        # cond: [B, d]
        # x: [B, T, d]

        gamma = self.gamma(cond).unsqueeze(1)
        beta = self.beta(cond).unsqueeze(1)

        return gamma * x + beta


# ------------------------------------------------
# Temporal Attention Pooling (lightweight)
# ------------------------------------------------

class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        # x: [B, T, d]

        w = self.score(x)             # [B, T, 1]
        w = torch.softmax(w, dim=1)

        out = (w * x).sum(dim=1)      # [B, d]
        return out


# ------------------------------------------------
# Model
# ------------------------------------------------

class Model(nn.Module):

    def __init__(self, args):
        super().__init__()

        d = args.embed_dim

        # Encoders
        self.eeg_encoder = EEGEncoder(out_dim=d)
        self.fnirs_encoder = FNIRSEncoder(out_dim=d)

        # Cross-modal (EEG → fNIRS)
        self.film = FiLM(d)
        self.temporal_pool = TemporalAttention(d)

        # Classifiers
        self.eeg_cls = nn.Linear(d, args.num_classes)
        self.fnirs_cls = nn.Linear(d, args.num_classes)
        self.fusion_cls = nn.Linear(2 * d, args.num_classes)

        # Session discriminators (domain adversarial)
        self.session_eeg = nn.Linear(d, args.num_sessions)
        self.session_fnirs = nn.Linear(d, args.num_sessions)
        self.session_fusion = nn.Linear(2 * d, args.num_sessions)

    def forward(self, eeg, fnirs, alpha=0.0):

        # ------------------------------------------------
        # EEG branch
        # ------------------------------------------------
        eeg_embed = self.eeg_encoder(eeg)   # [B, d]
        eeg_logits = self.eeg_cls(eeg_embed)

        # ------------------------------------------------
        # fNIRS branch
        # ------------------------------------------------
        fnirs_feat = self.fnirs_encoder(fnirs)   # [B, T, d]

        # EEG → FiLM modulation
        fnirs_feat = self.film(eeg_embed, fnirs_feat)

        # Temporal pooling
        fnirs_embed = self.temporal_pool(fnirs_feat)  # [B, d]
        fnirs_logits = self.fnirs_cls(fnirs_embed)

        # ------------------------------------------------
        # Fusion
        # ------------------------------------------------
        fusion_embed = torch.cat([eeg_embed, fnirs_embed], dim=-1)
        fusion_logits = self.fusion_cls(fusion_embed)

        # ------------------------------------------------
        # Domain adversarial (cross-session)
        # ------------------------------------------------
        rev_eeg = grad_reverse(eeg_embed, alpha)
        rev_fnirs = grad_reverse(fnirs_embed, alpha)
        rev_fusion = grad_reverse(fusion_embed, alpha)

        session_eeg = self.session_eeg(rev_eeg)
        session_fnirs = self.session_fnirs(rev_fnirs)
        session_fusion = self.session_fusion(rev_fusion)

        return {
            "logits": fusion_logits,

            "eeg_logits": eeg_logits,
            "fnirs_logits": fnirs_logits,
            "fusion_logits": fusion_logits,

            "session_eeg": session_eeg,
            "session_fnirs": session_fnirs,
            "session_fusion": session_fusion,

            "eeg_embed": eeg_embed,
            "fnirs_embed": fnirs_embed,
            "fusion_embed": fusion_embed
        }