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
    return x.mean(dim=1)
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
    def __init__(self, in_ch=72, hidden=36, out_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),

            # nn.Conv1d(hidden, hidden, kernel_size=15, padding=7),
            # nn.BatchNorm1d(hidden),
            # nn.ReLU(),
            # nn.Dropout(0.5),
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


# ------------------------------------------------
# Model
# ------------------------------------------------

class Model(nn.Module):

    def __init__(self, args):
        super().__init__()

        deeg = 128
        dfnirs = 32
        d = 128
        num_classes = 2
        num_sessions = 40

        # Encoders
        self.eeg_encoder = EEGEncoder(out_dim=deeg)
        self.fnirs_encoder = FNIRSEncoder(out_dim=dfnirs)

        # Classifiers
        self.eeg_cls = nn.Linear(deeg, num_classes)
        self.fnirs_cls = nn.Linear(dfnirs, num_classes)
        self.fusion_cls = nn.Sequential(
            nn.Linear(deeg + dfnirs, d),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d, num_classes)
        )
        # self.fusion_cls = nn.Linear(deeg + dfnirs, num_classes)

        # Session discriminators (domain adversarial)
        self.session_eeg = nn.Linear(deeg, num_sessions)
        self.session_fnirs = nn.Linear(dfnirs, num_sessions)
        self.session_fusion = nn.Linear(deeg + dfnirs, num_sessions)

    def forward(self, eeg, fnirs, alpha=0.0, arch='fusion'):
        eeg_seq = self.eeg_encoder(eeg)   # [B, T, d]
        # fnirs_seq = self.fnirs_encoder(fnirs)   # [B, T, d]

        eeg_embed = chunk_pool(eeg_seq)
        # fnirs_embed = chunk_pool(fnirs_seq)

        eeg_logits = self.eeg_cls(eeg_embed)
        # fnirs_logits = self.fnirs_cls(fnirs_embed)

        # if arch == 'fusion':
        #     fusion_embed = torch.cat([eeg_embed, fnirs_embed], dim=-1)
        # if arch == 'rev-fusion':
        #     fusion_embed = torch.cat([fnirs_embed, eeg_embed], dim=-1)
        # if arch == 'eeg':
        #     fusion_embed = torch.cat([eeg_embed, torch.zeros_like(fnirs_embed)], dim=-1)
        # if arch == 'fnirs':
        #     fusion_embed = torch.cat([torch.zeros_like(eeg_embed), fnirs_embed], dim=-1)
        # fusion_logits = self.fusion_cls(fusion_embed)

        rev_eeg = grad_reverse(eeg_embed, alpha)
        # rev_fnirs = grad_reverse(fnirs_embed, alpha)
        # rev_fusion = grad_reverse(fusion_embed, alpha)

        session_eeg = self.session_eeg(rev_eeg)
        # session_fnirs = self.session_fnirs(rev_fnirs)
        # session_fusion = self.session_fusion(rev_fusion)

        # return {
        #     "logits": fusion_logits,

        #     "eeg_logits": eeg_logits,
        #     "fnirs_logits": fnirs_logits,
        #     "fusion_logits": fusion_logits,

        #     "session_eeg": session_eeg,
        #     "session_fnirs": session_fnirs,
        #     "session_fusion": session_fusion,

        #     "eeg_embed": eeg_embed,
        #     "fnirs_embed": fnirs_embed,
        #     "fusion_embed": fusion_embed
        # }

        
        return {
            "logits": eeg_logits,

            "eeg_logits": eeg_logits,
            "fnirs_logits": eeg_logits,
            "fusion_logits": eeg_logits,

            "session_eeg": session_eeg,
            "session_fnirs": session_eeg,
            "session_fusion": session_eeg,
        }