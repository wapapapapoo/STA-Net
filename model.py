import torch
import torch.nn as nn
import torch.nn.functional as F

class DSBlock(nn.Module):

    def __init__(self, cin, cout, k, stride=1, drop=0.2):

        super().__init__()

        self.depth = nn.Conv1d(
            cin, cin,
            kernel_size=k,
            padding=k//2,
            stride=stride,
            groups=cin
        )

        self.point = nn.Conv1d(
            cin, cout,
            kernel_size=1
        )

        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.GELU()

        self.drop = nn.Dropout1d(drop)

    def forward(self,x):

        # x (B,C,T)

        x = self.depth(x)      # (B,C,T)
        x = self.point(x)      # (B,Cout,T)

        x = self.bn(x)
        x = self.act(x)

        x = self.drop(x)       # (B,Cout,T)

        return x






class TemporalASPP(nn.Module):

    def __init__(self, cin, cout, drop=0.3):

        super().__init__()

        self.b1 = nn.Conv1d(cin, cout, 1)

        self.b2 = nn.Conv1d(
            cin, cout, 3,
            padding=3,
            dilation=3
        )

        self.b3 = nn.Conv1d(
            cin, cout, 3,
            padding=6,
            dilation=6
        )

        self.b4 = nn.Conv1d(
            cin, cout, 3,
            padding=12,
            dilation=12
        )

        self.proj = nn.Conv1d(cout*4, cout, 1)

        self.drop = nn.Dropout1d(drop)

    def forward(self,x):

        # x (B,C,T)

        b1 = self.b1(x)  
        b2 = self.b2(x)  
        b3 = self.b3(x)  
        b4 = self.b4(x)  

        x = torch.cat([b1,b2,b3,b4],dim=1)  # (B,4C,T)

        x = self.proj(x)  # (B,C,T)

        x = self.drop(x)

        return x





class EEGFNIRSAlign(nn.Module):

    def __init__(self, c, drop=0.2):

        super().__init__()

        self.q = nn.Conv1d(c,c,1)
        self.k = nn.Conv1d(c,c,1)
        self.v = nn.Conv1d(c,c,1)

        self.scale = c ** -0.5

        self.drop = nn.Dropout(drop)

    def forward(self,eeg,fnirs):

        # eeg   (B,C,Te)
        # fnirs (B,C,Tf)

        q = self.q(eeg)
        k = self.k(fnirs)
        v = self.v(fnirs)

        q = q.transpose(1,2)   # (B,Te,C)
        k = k.transpose(1,2)   # (B,Tf,C)
        v = v.transpose(1,2)   # (B,Tf,C)

        attn = torch.matmul(q,k.transpose(1,2)) * self.scale
        # (B,Te,Tf)

        attn = torch.softmax(attn,dim=-1)

        attn = self.drop(attn)

        out = torch.matmul(attn,v)
        # (B,Te,C)

        out = out.transpose(1,2)

        return out
    





class EEGBranch(nn.Module):

    def __init__(self):

        super().__init__()

        self.stage1 = DSBlock(28,64,31,drop=0.15)
        # (B,64,600)

        self.stage2 = DSBlock(64,128,31,stride=2,drop=0.2)
        # (B,128,300)

        self.stage3 = DSBlock(128,192,15,stride=2,drop=0.25)
        # (B,192,150)

        self.post = DSBlock(192,192,15,drop=0.3)

        self.aspp = TemporalASPP(192,192,drop=0.35)

    def forward(self,x):

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        feat = x

        x = self.post(x)
        x = self.aspp(x)

        return feat, x







class FNIRSBranch(nn.Module):

    def __init__(self):

        super().__init__()

        self.stage1 = DSBlock(72,128,15,drop=0.2)
        # (B,128,120)

        self.stage2 = DSBlock(128,192,15,stride=2,drop=0.25)
        # (B,192,60)

        self.align = EEGFNIRSAlign(192,drop=0.2)

        self.post1 = DSBlock(192,192,9,drop=0.3)
        self.post2 = DSBlock(192,192,9,drop=0.3)

        self.aspp = TemporalASPP(192,192,drop=0.35)

    def forward(self,fnirs,eeg_feat):

        x = self.stage1(fnirs)
        x = self.stage2(x)

        align = self.align(eeg_feat,x)

        align = F.interpolate(
            align,
            size=x.shape[-1],
            mode="linear",
            align_corners=False
        )

        x = x + align

        x = self.post1(x)
        x = self.post2(x)

        x = self.aspp(x)

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

    def __init__(self, c=192):

        super().__init__()

        self.eeg_gate = nn.Conv1d(c,c,1)
        self.fnirs_gate = nn.Conv1d(c,c,1)

        self.mix = nn.Conv1d(c*2,c,1)

        self.drop = nn.Dropout1d(0.35)

    def forward(self,eeg,fnirs):

        g1 = torch.sigmoid(self.eeg_gate(eeg))
        g2 = torch.sigmoid(self.fnirs_gate(fnirs))

        eeg = eeg * g2
        fnirs = fnirs * g1

        x = torch.cat([eeg,fnirs],dim=1)

        x = self.mix(x)

        x = self.drop(x)

        return x





class TemporalProjector(nn.Module):

    def __init__(self,c=192):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(c,c,5,padding=2),
            nn.GELU(),
            nn.Dropout1d(0.4),

            nn.Conv1d(c,c,5,padding=2),
            nn.GELU(),
            nn.Dropout1d(0.4),

            nn.Conv1d(c,c,1)
        )

    def forward(self,x):

        x = self.net(x)

        x = torch.flatten(x,1)

        return x







class Classifier(nn.Module):

    def __init__(self,in_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(in_dim,512),
            nn.GELU(),
            nn.Dropout(0.6),

            nn.Linear(512,128),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Linear(128,2)
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


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)




class Model(nn.Module):

    def __init__(self,args):

        super().__init__()

        self.eeg = EEGBranch()
        self.fnirs = FNIRSBranch()

        self.align = TemporalAlign(120)

        self.fusion = CrossModalFusion(192)

        self.proj_eeg = TemporalProjector(192)
        self.proj_fnirs = TemporalProjector(192)
        self.proj_fusion = TemporalProjector(192)

        emb = 192*120

        self.classifier = Classifier(emb)

        self.session_classifier = nn.Sequential(

            nn.Linear(emb,256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256,args['TRAIL_GROUP_AMOUNT'])
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

        eeg_logits = self.classifier(eeg_embed)
        fnirs_logits = self.classifier(fnirs_embed)
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