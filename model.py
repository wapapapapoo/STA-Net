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

def chunk_pool(x, chunks=6):
    B, T, D = x.shape
    x = x.view(B, chunks, T // chunks, D)   # [B, C, Tc, D]
    x = x.mean(dim=2)                       # [B, C, D] 每段一个向量
    x = x.max(dim=1).values                 # [B, D] 选最重要的段
    return x

# ------------------------------------------------
# EEG Encoder (small + stable)
# ------------------------------------------------

class EEGEncoder(nn.Module):
    def __init__(self, in_ch=28, hidden=64, out_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.conv(x)          # [B, hidden, T]
        x = x.transpose(1, 2)     # [B, T, hidden]
        x = self.proj(x)          # [B, T, d]
        return torch.tanh(x)


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
            nn.Dropout(0.3),

            nn.Conv1d(hidden, hidden, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

    def forward(self, x):
        # x: [B, 72, 120]
        x = self.conv(x)              # [B, hidden, T]
        x = x.transpose(1, 2)         # [B, T, hidden]
        x = self.proj(x)              # [B, T, d]
        x = torch.tanh(x)
        return x

# # ------------------------------------------------
# # CrossModalAttn
# # ------------------------------------------------

# class CrossModalAttention(nn.Module):
#     def __init__(self, dim, Te=600, Tf=120, k_min=2.0, k_max=8.0):
#         super().__init__()

#         self.q_proj = nn.Linear(dim, dim)
#         self.k_proj = nn.Linear(dim, dim)
#         self.v_proj = nn.Linear(dim, dim)

#         self.scale = dim ** -0.5

#         mask = torch.zeros(Te, Tf, dtype=torch.bool)
#         for t in range(Te):
#             j0 = t // 20  # EEG→fNIRS 对齐（200Hz → 10Hz）
#             j_min = int(j0 + 10 * k_min)
#             j_max = int(j0 + 10 * k_max)
#             j_min = max(0, j_min)
#             j_max = min(Tf - 1, j_max)
#             if j_min <= j_max:
#                 mask[t, j_min:j_max+1] = True

#         self.register_buffer("mask", mask)

#     def forward(self, eeg_seq, fnirs_seq):
#         B, Te, d = eeg_seq.shape
#         Tf = fnirs_seq.shape[1]

#         Q = self.q_proj(eeg_seq)
#         K = self.k_proj(fnirs_seq)
#         V = self.v_proj(fnirs_seq)

#         attn = torch.matmul(Q, K.transpose(1, 2)) * self.scale  # [B, Te, Tf]
#         attn = attn.masked_fill(~self.mask.unsqueeze(0), float('-inf'))
#         attn = torch.softmax(attn, dim=-1)
#         out = torch.matmul(attn, V)

#         return out, attn

# ------------------------------------------------
# Model
# ------------------------------------------------

class Model(nn.Module):

    def __init__(self, args):
        super().__init__()

        d = 128
        num_classes = 2
        num_sessions = 40

        # Encoders
        self.eeg_encoder = EEGEncoder(out_dim=d)
        self.fnirs_encoder = FNIRSEncoder(out_dim=d)

        # self.cross_attn = CrossModalAttention(dim=d, k_min=5, k_max=10)

        # Classifiers
        self.eeg_cls = nn.Linear(d, num_classes)
        self.fnirs_cls = nn.Linear(d, num_classes)
        self.fusion_cls = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d, num_classes)
        )
        # self.fusion_cls = nn.Linear(2 * d, num_classes)

        # Session discriminators (domain adversarial)
        self.session_eeg = nn.Linear(d, num_sessions)
        self.session_fnirs = nn.Linear(d, num_sessions)
        self.session_fusion = nn.Linear(2 * d, num_sessions)

    def forward(self, eeg, fnirs, alpha=0.0):

        # ------------------------------------------------
        # EEG branch
        # ------------------------------------------------
        eeg_seq = self.eeg_encoder(eeg)   # [B, T, d]

        # ------------------------------------------------
        # fNIRS branch
        # ------------------------------------------------
        fnirs_seq = self.fnirs_encoder(fnirs)   # [B, T, d]

        # aligned_fnirs, attn = self.cross_attn(eeg_seq, fnirs_feat)  # [B, Te, d]
        eeg_embed = chunk_pool(eeg_seq)
        fnirs_embed = chunk_pool(fnirs_seq)

        eeg_logits = self.eeg_cls(eeg_embed)
        fnirs_logits = self.fnirs_cls(fnirs_embed)

        # ------------------------------------------------
        # Fusion
        # ------------------------------------------------
        fusion_embed = torch.cat([eeg_embed + fnirs_embed, eeg_embed * fnirs_embed], dim=-1)
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