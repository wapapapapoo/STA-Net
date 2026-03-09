import torch
import torch.nn as nn
import torch.nn.functional as F


############################################
# Pearson correlation
############################################

def pearson_r(eeg, fnirs):

    mx = eeg.mean(dim=1, keepdim=True)
    my = fnirs.mean(dim=1, keepdim=True)

    xm = eeg - mx
    ym = fnirs - my

    r_num = (xm * ym).mean(dim=1)
    r_den = xm.std(dim=1) * ym.std(dim=1) + 1e-6

    plcc = torch.abs(r_num / r_den)
    return plcc.mean()


############################################
# Positional embedding
############################################

class pos_embedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.pos_embedding = None

    def forward(self, x):

        if self.pos_embedding is None:
            B, L, D = x.shape
            self.pos_embedding = nn.Parameter(
                torch.empty(1, L, D)
            )
            nn.init.kaiming_uniform_(self.pos_embedding)

        return x + self.pos_embedding


############################################
# GAP
############################################

class gap(nn.Module):

    def forward(self, x):
        return x.mean(dim=-2, keepdim=True)


############################################
# expand_dims_layer
############################################

class expand_dims_layer(nn.Module):

    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return torch.unsqueeze(x, dim=self.axis)


############################################
# reduce_sum_layer
############################################

class reduce_sum_layer(nn.Module):

    def __init__(self, axis, keepaxis):
        super().__init__()
        self.axis = axis
        self.keepaxis = keepaxis

    def forward(self, x):
        return torch.sum(x, dim=self.axis, keepdim=self.keepaxis)


############################################
# TimeDistributed Conv3D
############################################

class TimeDistributedConv3D(nn.Module):

    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x):

        B, T = x.shape[:2]

        x = x.reshape(B*T, *x.shape[2:])

        x = x.permute(0,4,3,1,2)

        x = self.conv(x)

        x = x.permute(0,3,4,2,1)

        x = x.reshape(B, T, *x.shape[1:])

        return x


############################################
# FGA module
############################################

class fga(nn.Module):

    def __init__(self, tem_kernel_size):

        super().__init__()

        self.channel_pooling = TimeDistributedConv3D(
            nn.Conv3d(
                in_channels=2,
                out_channels=1,
                kernel_size=(tem_kernel_size,3,3),
                padding="same"
            )
        )

        self.tap_fnirs = gap()

        self.residual_para = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):

        eeg_fusion, eeg, fnirs = inputs

        fnirs_attention = self.channel_pooling(fnirs)

        fnirs_attention_map = self.tap_fnirs(fnirs_attention)

        fnirs_attention_map = fnirs_attention_map.mean(dim=1)

        fnirs_attention_map_norm = torch.sigmoid(fnirs_attention_map)

        eeg_fusion_guided = eeg_fusion * fnirs_attention_map_norm

        residual_para_norm = torch.sigmoid(self.residual_para)

        eeg_add = residual_para_norm * eeg + (1-residual_para_norm) * eeg_fusion

        fga_feature = eeg_fusion_guided + eeg_add

        eeg_plcc = eeg.mean(dim=(-1,-2))
        eeg_plcc = eeg_plcc.flatten(1)

        fnirs_plcc = fnirs_attention_map_norm.flatten(1)

        fga_loss = pearson_r(eeg_plcc, fnirs_plcc)

        return fga_feature


############################################
# conv_block
############################################

class conv_block(nn.Module):

    def __init__(self,
                 eeg_filter, eeg_size, eeg_stride,
                 fnirs_filter, fnirs_size, fnirs_stride,
                 eegfusion_filter, eegfusion_size, eegfusion_stride,
                 tem_kernel_size):

        super().__init__()

        self.eeg_conv = nn.Conv3d(
            1, eeg_filter,
            kernel_size=(eeg_size[2],eeg_size[0],eeg_size[1]),
            stride=(eeg_stride[2],eeg_stride[0],eeg_stride[1]),
            padding="same"
        )

        self.eeg_bn = nn.BatchNorm3d(eeg_filter)

        self.fnirs_conv = TimeDistributedConv3D(
            nn.Conv3d(
                2, fnirs_filter,
                kernel_size=(fnirs_size[2],fnirs_size[0],fnirs_size[1]),
                stride=(fnirs_stride[2],fnirs_stride[0],fnirs_stride[1]),
                padding="same"
            )
        )

        self.fnirs_bn = nn.BatchNorm3d(fnirs_filter)

        self.eegfusion_conv = nn.Conv3d(
            1, eegfusion_filter,
            kernel_size=(eegfusion_size[2],eegfusion_size[0],eegfusion_size[1]),
            stride=(eegfusion_stride[2],eegfusion_stride[0],eegfusion_stride[1]),
            padding="same"
        )

        self.eegfusion_bn = nn.BatchNorm3d(eegfusion_filter)

        self.fga = fga(tem_kernel_size)

    def forward(self, inputs):

        eegfusion, eeg, fnirs = inputs

        eeg = eeg.permute(0,4,3,1,2)

        eeg_feature = self.eeg_conv(eeg)
        eeg_feature = self.eeg_bn(eeg_feature)
        eeg_feature = F.elu(eeg_feature)

        eeg_feature = eeg_feature.permute(0,3,4,2,1)

        fnirs_feature = self.fnirs_conv(fnirs)
        fnirs_feature = F.elu(fnirs_feature)

        eegfusion = eegfusion.permute(0,4,3,1,2)

        eegfusion_feature = self.eegfusion_conv(eegfusion)
        eegfusion_feature = self.eegfusion_bn(eegfusion_feature)
        eegfusion_feature = F.elu(eegfusion_feature)

        eegfusion_feature = eegfusion_feature.permute(0,3,4,2,1)

        eegfusion_fga = self.fga((eegfusion_feature, eeg_feature, fnirs_feature))

        return eegfusion_fga, eeg_feature, fnirs_feature


############################################
# e_f_attention
############################################

class e_f_attention(nn.Module):

    def __init__(self, emb_size, heads, drop):

        super().__init__()

        self.q_proj = nn.Linear(512, emb_size)
        self.fusion_proj = nn.Linear(512, emb_size)

        self.k_proj = nn.Linear(512, emb_size)

        self.pos = pos_embedding()

        self.mha = nn.MultiheadAttention(
            emb_size,
            heads,
            dropout=drop,
            batch_first=True
        )

    def forward(self, inputs):

        eeg, fnirs = inputs

        q_eeg = eeg.flatten(1)

        fusion_output = self.fusion_proj(q_eeg)

        q_eeg = self.q_proj(q_eeg).unsqueeze(1)

        k_fnirs = fnirs.reshape(fnirs.shape[0],11,-1)

        k_fnirs = self.pos(k_fnirs)

        k_fnirs = self.k_proj(k_fnirs)

        fnirs_weighted, attn = self.mha(q_eeg, k_fnirs, k_fnirs)

        attention_weights = attn.mean(dim=1)

        q_eeg = q_eeg.mean(dim=1)

        fnirs_weighted = fnirs_weighted.mean(dim=1)

        ef_loss = pearson_r(q_eeg, fnirs_weighted)

        return fusion_output, fnirs_weighted, attention_weights


############################################
# STA NET
############################################

class STA_NET(nn.Module):

    def __init__(self):

        super().__init__()

        self.block1 = conv_block(
            16,(2,2,13),(2,2,6),
            16,(2,2,5),(2,2,2),
            16,(2,2,13),(2,2,6),
            5
        )

        self.block2 = conv_block(
            32,(2,2,5),(2,2,2),
            32,(2,2,3),(2,2,2),
            32,(2,2,5),(2,2,2),
            3
        )

        self.gap = gap()

        self.attn = e_f_attention(256,10,0.5)

        self.fc_fusion = nn.Linear(256,256)
        self.fc_fnirs = nn.Linear(256,256)

        self.fc_eeg = nn.Linear(512,256)

        self.pred_fusion = nn.Linear(256,2)
        self.pred_fnirs = nn.Linear(256,2)
        self.pred_eeg = nn.Linear(256,2)

        self.fnirs_p = nn.Linear(256,1)
        self.fusion_p = nn.Linear(256,1)

    def forward(self, eeg, fnirs):

        eegfusion1, eeg1, fnirs1 = self.block1((eeg,eeg,fnirs))

        eegfusion2, eeg2, fnirs2 = self.block2((eegfusion1,eeg1,fnirs1))

        eegfusion2 = self.gap(eegfusion2)
        eeg2 = self.gap(eeg2)
        fnirs2 = self.gap(fnirs2)

        fusion_feature, fnirs_feature, _ = self.attn((eegfusion2,fnirs2))

        fusion_feature = F.elu(self.fc_fusion(fusion_feature))
        fnirs_feature = F.elu(self.fc_fnirs(fnirs_feature))

        eeg_feature = eeg2.flatten(1)
        eeg_feature = F.elu(self.fc_eeg(eeg_feature))

        eegfusion_pred = F.softmax(self.pred_fusion(fusion_feature),dim=1)
        fnirs_pred = F.softmax(self.pred_fnirs(fnirs_feature),dim=1)
        eeg_pred = F.softmax(self.pred_eeg(eeg_feature),dim=1)

        eegfusion_pred = eegfusion_pred.unsqueeze(1)
        fnirs_pred = fnirs_pred.unsqueeze(1)

        the_pred = torch.cat([eegfusion_pred,fnirs_pred],dim=1)

        fnirs_w = self.fnirs_p(fnirs_feature)
        fusion_w = self.fusion_p(fusion_feature)

        p_weight = torch.cat([fusion_w,fnirs_w],dim=1)
        p_weight = F.softmax(p_weight,dim=1).unsqueeze(-1)

        the_pred = the_pred * p_weight

        the_pred = the_pred.sum(dim=1)

        return the_pred, eeg_pred
