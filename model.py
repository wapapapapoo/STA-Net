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
# EEG Encoder（不降采样）
# ------------------------------------------------

class EEGEncoder(nn.Module):
    def __init__(self, in_ch=28, hidden=64, out_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),

            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.conv(x)          # [B, H, T]
        x = x.transpose(1, 2)     # [B, T, H]
        x = self.proj(x)          # [B, T, d]
        return x


# ------------------------------------------------
# fNIRS Encoder（降capacity）
# ------------------------------------------------

class FNIRSEncoder(nn.Module):
    def __init__(self, in_ch=72, hidden=32, out_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x):
        # 去baseline（关键）
        x = x - x.mean(dim=-1, keepdim=True)

        x = self.conv(x)              # [B, H, T]
        x = x.transpose(1, 2)         # [B, T, H]
        x = self.proj(x)              # [B, T, d]
        return x


# ------------------------------------------------
# 双向 Cross Attention（带mask）
# ------------------------------------------------

class BiCrossAttention(nn.Module):
    def __init__(self, dim, Te=600, Tf=120, k_min=2.0, k_max=8.0):
        super().__init__()

        self.scale = dim ** -0.5

        self.qe = nn.Linear(dim, dim)
        self.ke = nn.Linear(dim, dim)
        self.ve = nn.Linear(dim, dim)

        self.qf = nn.Linear(dim, dim)
        self.kf = nn.Linear(dim, dim)
        self.vf = nn.Linear(dim, dim)

        # 构建mask（EEG→fNIRS）
        mask = torch.zeros(Te, Tf, dtype=torch.bool)

        for t in range(Te):
            j0 = t // 20
            j_min = int(j0 + 10 * k_min)
            j_max = int(j0 + 10 * k_max)

            j_min = max(0, j_min)
            j_max = min(Tf - 1, j_max)

            if j_min <= j_max:
                mask[t, j_min:j_max+1] = True

        self.register_buffer("mask_e2f", mask)
        self.register_buffer("mask_f2e", mask.t())

    def forward(self, eeg, fnirs):

        # EEG → fNIRS
        Qe = self.qe(eeg)
        Kf = self.kf(fnirs)
        Vf = self.vf(fnirs)

        attn_e2f = torch.matmul(Qe, Kf.transpose(1, 2)) * self.scale
        attn_e2f = attn_e2f.masked_fill(~self.mask_e2f.unsqueeze(0), -1e9)
        attn_e2f = torch.softmax(attn_e2f, dim=-1)

        aligned_fnirs = torch.matmul(attn_e2f, Vf)

        # fNIRS → EEG
        Qf = self.qf(fnirs)
        Ke = self.ke(eeg)
        Ve = self.ve(eeg)

        attn_f2e = torch.matmul(Qf, Ke.transpose(1, 2)) * self.scale
        attn_f2e = attn_f2e.masked_fill(~self.mask_f2e.unsqueeze(0), -1e9)
        attn_f2e = torch.softmax(attn_f2e, dim=-1)

        aligned_eeg = torch.matmul(attn_f2e, Ve)

        return aligned_eeg, aligned_fnirs


# ------------------------------------------------
# Segment Pool（可学习，不用max）
# ------------------------------------------------

class SegmentPool(nn.Module):
    def __init__(self, dim, chunks=6):
        super().__init__()
        self.chunks = chunks
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        B, T, D = x.shape
        x = x.view(B, self.chunks, T // self.chunks, D)
        x = x.mean(dim=2)                # [B,C,D]

        w = torch.softmax(self.score(x), dim=1)
        out = (w * x).sum(dim=1)

        return out


# ------------------------------------------------
# Fusion（抗单模态碾压）
# ------------------------------------------------

class Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, eeg, fnirs):
        z = torch.cat([eeg, fnirs], dim=-1)
        g = self.gate(z)

        return g * eeg + (1 - g) * fnirs


# ------------------------------------------------
# Model
# ------------------------------------------------

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        d = 128
        num_classes = 2
        num_sessions = 40

        self.eeg_enc = EEGEncoder(out_dim=d)
        self.fnirs_enc = FNIRSEncoder(out_dim=d)

        self.cross = BiCrossAttention(d)
        self.pool = SegmentPool(d)
        self.fusion = Fusion(d)

        self.cls = nn.Linear(d, num_classes)
        self.eeg_cls = nn.Linear(d, num_classes)
        self.fnirs_cls = nn.Linear(d, num_classes)

        self.session_eeg = nn.Linear(d, num_sessions)
        self.session_fnirs = nn.Linear(d, num_sessions)
        self.session_fusion = nn.Linear(d, num_sessions)

    def forward(self, eeg, fnirs, alpha=0.0, arch='fusion'):

        eeg_seq = self.eeg_enc(eeg)        # [B,600,d]
        fnirs_seq = self.fnirs_enc(fnirs)  # [B,120,d]

        # 双向对齐
        aligned_eeg, aligned_fnirs = self.cross(eeg_seq, fnirs_seq)

        # 残差
        eeg_seq = aligned_fnirs
        fnirs_seq = aligned_eeg

        # pooling
        eeg_embed = self.pool(eeg_seq)
        fnirs_embed = self.pool(fnirs_seq)

        eeg_logits = self.eeg_cls(eeg_embed)
        fnirs_logits = self.fnirs_cls(fnirs_embed)

        # -------------------------
        # fusion
        # -------------------------
        fused = self.fusion(eeg_embed, fnirs_embed)
        fusion_logits = self.cls(fused)

        # -------------------------
        # domain adversarial
        # -------------------------
        rev_eeg = grad_reverse(eeg_embed, alpha)
        rev_fnirs = grad_reverse(fnirs_embed, alpha)
        rev_fusion = grad_reverse(fused, alpha)

        session_eeg = self.session_eeg(rev_eeg)
        session_fnirs = self.session_fnirs(rev_fnirs)
        session_fusion = self.session_fusion(rev_fusion)

        return {
            # 主任务
            "logits": fusion_logits,
            "fusion_logits": fusion_logits,

            # 单模态监督
            "eeg_logits": eeg_logits,
            "fnirs_logits": fnirs_logits,

            # domain
            "session_eeg": session_eeg,
            "session_fnirs": session_fnirs,
            "session_fusion": session_fusion,

            # 表征
            "eeg_embed": eeg_embed,
            "fnirs_embed": fnirs_embed,
            "fusion_embed": fused,
        }