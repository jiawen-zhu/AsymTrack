import pdb
import torch, torch.nn as nn, torch.nn.functional as F
from timm.models.layers import trunc_normal_, to_2tuple


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4, patch_stride=4, patch_pad=0, norm_layer=None):
        """
        In-shape [b,h,w,c], out-shape[b, h',w',c']
        Args:
            patch_size:
            in_chans:
            embed_dim:
            patch_pad:
            norm_layer:
        """
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=patch_pad)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x


class DySepConvAtten(nn.Module):
    def __init__(self, hidden_dim, n_proposals):
        super(DySepConvAtten, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_proposals = n_proposals
        self.kernel_size = 3  # conv_kernel_size_1D

        self.weight_linear = nn.Linear(self.hidden_dim, self.num_proposals + self.kernel_size)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, query, value):
        B, N, C = query.shape

        dy_conv_weight = self.weight_linear(query)
        dy_depth_conv_weight = dy_conv_weight[:, :, :self.kernel_size].view(B, self.num_proposals, 1, self.kernel_size)
        dy_point_conv_weight = dy_conv_weight[:, :, self.kernel_size:].view(B, self.num_proposals, self.num_proposals,
                                                                            1)

        res = []
        value = value.unsqueeze(1)
        for i in range(B):
            out = F.relu(F.conv1d(input=value[i], weight=dy_depth_conv_weight[i], groups=N, padding="same"))
            out = F.conv1d(input=out, weight=dy_point_conv_weight[i], padding='same')

            res.append(out)
        point_out = torch.cat(res, dim=0)
        point_out = self.norm(point_out)
        return point_out


class ETM(nn.Module):
    def __init__(self, in_channels, hidden_dim, deploy=False,
                 residual=False, version=None, **kwargs):
        super(ETM, self).__init__()
        self.deploy = deploy
        self.residual = residual
        self.version = version
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.kernels = None

        self.pre_kernels = nn.Linear(self.in_channels, self.hidden_dim)

        self.pre_kernels_x = nn.Linear(self.in_channels, self.hidden_dim)

        self.post_kernels = nn.Linear(self.hidden_dim, self.in_channels)

        self.downsample = PatchEmbed(
            in_chans=self.in_channels,
            embed_dim=self.in_channels,
            patch_size=3,
            patch_stride=2,
            patch_pad=1,
            norm_layer=nn.LayerNorm
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.proj_Cto1 = nn.Linear(self.in_channels, 1)
        self.ln = nn.LayerNorm(self.in_channels)
        self.act = nn.ReLU()
        self.conv_att = DySepConvAtten(self.in_channels, self.hidden_dim)

    def forward(self, x_feat, z_feat, Train=True):

        if ((self.kernels is None) and not Train) or Train:

            z_feat_core = self.pre_kernels(z_feat)
            kernels = torch.einsum('bhwn,bhwc->bnc', z_feat_core, z_feat)
            self.kernels = kernels.detach().clone()
        else:
            kernels = self.kernels

        if self.version == 'v5':
            x_feat_n = self.pre_kernels_x(x_feat)  # bhwn
            kernels_x = torch.einsum('bhwn,bhwc->bnc', x_feat_n, x_feat)  # [bnc]
            kernels_x = self.conv_att(self.kernels, kernels_x)  # bnc
            x_feat_correlated = torch.einsum("bnc,bhwn->bhwc", kernels_x, x_feat_n)  # bhwc
            tem_prototype = self.proj_Cto1(kernels_x)  # bn1, d=1
            tem_prototype = torch.sigmoid(torch.einsum('bnd,bnc->bdc', tem_prototype, kernels_x)).unsqueeze(-2)  # b1c
            x_feat_correlated = x_feat_correlated * tem_prototype

        x_feat = x_feat + self.act(self.ln(x_feat_correlated))

        return x_feat


class Interact(nn.Module):
    def __init__(self, in_channels, hidden_dim, bias=True, version=None, **kwargs):
        super(Interact, self).__init__()

        self.version = version
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fc = nn.Linear(in_channels, hidden_dim, bias=bias)

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_feat, z_feat):
        z_feat = self.upsample(z_feat.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        xz_feat = self.fc(torch.cat([x_feat, z_feat], dim=-1))

        x_feat += torch.sigmoid(self.alpha) * F.relu(xz_feat)

        return x_feat
