import torch
import torch.nn as nn
import torch.nn.functional as F

class DSBlock(nn.Module):

    def __init__(self, cin, cout, k, stride=1, drop=0.25):
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

        self.drop = nn.Dropout1d(drop)

    def forward(self,x):

        # x (B,C,T)

        x = self.dw(x)      # (B,C,T)
        x = self.pw(x)      # (B,Cout,T)

        x = self.norm(x)
        x = self.act(x)

        x = self.drop(x)

        return x



class TemporalPyramid(nn.Module):

    def __init__(self,c):

        super().__init__()

        self.b1 = nn.Conv1d(c,c,3,padding=1)
        self.b2 = nn.Conv1d(c,c,5,padding=2)
        self.b3 = nn.Conv1d(c,c,7,padding=3)

        self.mix = nn.Conv1d(c*3,c,1)

    def forward(self,x):

        # x (B,C,T)

        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)

        x = torch.cat([b1,b2,b3],dim=1)
        # (B,3C,T)

        x = self.mix(x)
        # (B,C,T)

        return x


class EEGFNIRSAlign(nn.Module):

    def __init__(self,c):

        super().__init__()

        self.q = nn.Conv1d(c,c,1)
        self.k = nn.Conv1d(c,c,1)
        self.v = nn.Conv1d(c,c,1)

        self.norm_eeg = nn.GroupNorm(8,c)
        self.norm_fnirs = nn.GroupNorm(8,c)

        self.scale = c ** -0.5

    def forward(self,eeg,fnirs):

        # eeg   (B,96,150)
        # fnirs (B,96,60)

        eeg = self.norm_eeg(eeg)
        fnirs = self.norm_fnirs(fnirs)

        q = self.q(eeg).transpose(1,2)      # (B,150,96)
        k = self.k(fnirs).transpose(1,2)    # (B,60,96)
        v = self.v(fnirs).transpose(1,2)    # (B,60,96)

        attn = torch.matmul(q,k.transpose(1,2)) * self.scale
        # (B,150,60)

        attn = torch.softmax(attn,dim=-1)

        out = torch.matmul(attn,v)
        # (B,150,96)

        out = out.transpose(1,2)
        # (B,96,150)

        return out



class EEGBranch(nn.Module):

    def __init__(self):

        super().__init__()

        self.stage1 = DSBlock(28,64,31)
        # (B,64,600)

        self.stage2 = DSBlock(64,96,31,stride=2)
        # (B,96,300)

        self.stage3 = DSBlock(96,96,15,stride=2)
        # (B,96,150)

        self.stage4 = DSBlock(96,96,15)
        # (B,96,150)

        self.pyramid = TemporalPyramid(96)
        # (B,96,150)

    def forward(self,x):

        # x (B,28,600)

        x = self.stage1(x)   # (B,64,600)
        x = self.stage2(x)   # (B,96,300)
        x = self.stage3(x)   # (B,96,150)

        feat = x             # (B,96,150)

        x = self.stage4(x)   # (B,96,150)

        x = self.pyramid(x)  # (B,96,150)

        return feat,x



class FNIRSBranch(nn.Module):

    def __init__(self):

        super().__init__()

        self.stage1 = DSBlock(72,64,9)
        # (B,64,120)

        self.stage2 = DSBlock(64,96,9,stride=2)
        # (B,96,60)

        self.align = EEGFNIRSAlign(96)

        self.post1 = DSBlock(96,96,7)
        self.post2 = DSBlock(96,96,7)

        self.pyramid = TemporalPyramid(96)

    def forward(self,fnirs,eeg_feat):

        # fnirs (B,72,120)

        x = self.stage1(fnirs)   # (B,64,120)
        x = self.stage2(x)       # (B,96,60)

        align = self.align(eeg_feat,x)
        # (B,96,150)

        align = F.interpolate(
            align,
            size=x.shape[-1],
            mode="linear",
            align_corners=False
        )
        # (B,96,60)

        x = x + align
        # (B,96,60)

        x = self.post1(x)
        x = self.post2(x)

        x = self.pyramid(x)
        # (B,96,60)

        return x




class TemporalAlign(nn.Module):

    def __init__(self, T=120):
        super().__init__()
        self.T = T

    def forward(self, x):

        # x (B,C,T)

        x = F.interpolate(
            x,
            size=self.T,
            mode="linear",
            align_corners=False
        )

        return x
    




class CrossModalFusion(nn.Module):

    def __init__(self,c=96):

        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv1d(c*2,c,1),
            nn.Sigmoid()
        )

        self.norm = nn.GroupNorm(8,c)

    def forward(self,eeg,fnirs):
        if self.training:
            mask = torch.rand(eeg.shape[0],1,1,device=eeg.device)

            eeg = eeg * (mask > 0.2)
            fnirs = fnirs * (mask < 0.8)

        g = self.gate(torch.cat([eeg,fnirs],dim=1))
        # (B,96,T)

        x = g*eeg + (1-g)*fnirs
        # (B,96,T)

        x = self.norm(x)

        return x




class TemporalProjector(nn.Module):

    def __init__(self,c=96):
        super().__init__()

        self.conv = nn.Conv1d(c,32,3,padding=1)
        self.drop = nn.Dropout(0.3)

        self.attn = nn.Conv1d(32,1,1)

    def forward(self,x):

        x = self.conv(x)
        x = self.drop(x)

        w = torch.softmax(self.attn(x),dim=-1)

        x = (x*w).sum(-1)

        return x








class Classifier(nn.Module):

    def __init__(self,in_dim=48):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(in_dim,48),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(48,24),
            nn.GELU(),

            nn.Linear(24,2)
        )

    def forward(self,x):

        return self.net(x)












class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1):
    return GradReverse.apply(x, alpha)




class Model(nn.Module):

    def __init__(self,args):

        super().__init__()

        self.eeg = EEGBranch()
        self.fnirs = FNIRSBranch()

        self.align = TemporalAlign(120)

        self.fusion = CrossModalFusion(96)

        self.proj_eeg = TemporalProjector(96)
        self.proj_fnirs = TemporalProjector(96)
        self.proj_fusion = TemporalProjector(96)

        emb = 48

        self.classifier_eeg = Classifier(emb)
        self.classifier_fnirs = Classifier(emb)
        self.classifier = Classifier(emb)

        self.session_classifier = nn.Sequential(
            nn.Linear(emb,64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64,args['TRAIL_GROUP_AMOUNT'])
        )

    def forward(self,eeg,fnirs):

        # --------------------------------
        # encoder
        # --------------------------------

        eeg_feat, eeg_final = self.eeg(eeg)
        # (B,192,150)

        fnirs_final = self.fnirs(fnirs,eeg_feat)
        # (B,192,60)

        # --------------------------------
        # temporal align
        # --------------------------------

        eeg_final = self.align(eeg_final)
        fnirs_final = self.align(fnirs_final)

        if self.training:
            eeg_final = eeg_final + torch.randn_like(eeg_final)*0.02
            fnirs_final = fnirs_final + torch.randn_like(fnirs_final)*0.02

        # (B,192,120)

        # --------------------------------
        # fusion
        # --------------------------------

        fusion = self.fusion(eeg_final,fnirs_final)
        # (B,192,120)

        # --------------------------------
        # embedding
        # --------------------------------

        eeg_embed = self.proj_eeg(eeg_final)
        fnirs_embed = self.proj_fnirs(fnirs_final)
        fusion_embed = self.proj_fusion(fusion)

        # (B,192*120)

        # --------------------------------
        # classification
        # --------------------------------

        eeg_logits = self.classifier_eeg(eeg_embed)
        fnirs_logits = self.classifier_fnirs(fnirs_embed)
        fusion_logits = self.classifier(fusion_embed)

        # --------------------------------
        # GRL
        # --------------------------------

        rev_eeg = grad_reverse(eeg_embed)
        rev_fnirs = grad_reverse(fnirs_embed)
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
            "fnirs_embed": fnirs_embed,
            "fusion_embed": fusion_embed
        }