import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch.jit import Final
from lib.models.AsymTrack.ibe import OPE, DySepConvAtten
from lib.models.AsymTrack.mobileone import MBoneBlock
from lib.models.AsymTrack.tem_kernel import ETM, Interact


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


class AttMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bias=True):
        # channel last
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    fast_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        # assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = max(dim // num_heads, 32)
        self.scale = self.head_dim ** -0.5
        self.fast_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.qkv = nn.Linear(dim, self.num_heads * self.head_dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fast_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.head_dim * self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class AttentionBlock(nn.Module):

    def __init__(
            self,
            dim, mlp_ratio=4., num_heads=8, qkv_bias=False, qk_norm=False, drop=0., attn_drop=0.,
            init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = AttMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, *image, fuselist=None, Fuse=0, Train=True, index=0, total_att_num=None):
        if Fuse == 0:  # z pass att
            z = image[0]
            B, H2, W2, C = z.size()
            z = z.reshape(B, H2 * W2, C).contiguous()
            z = z + self.drop_path1(self.ls1(self.attn(self.norm1(z))))
            z = z + self.drop_path2(self.ls2(self.mlp(self.norm2(z))))
            z = z.reshape(B, H2, W2, C).contiguous()
            return z
        elif Fuse == 1 or Fuse == 2:  # x pass with z | x pass with z feat
            x = image[0]
            B, H1, W1, C = x.size()
            x = x.reshape(B, H1 * W1, C).contiguous()
            if Fuse == 1:
                z = image[1]
                B, H2, W2, C = z.size()
                z = z.reshape(B, H2 * W2, C).contiguous()
                if not Train:
                    fuselist.append(z)  # save z for interaction
                else:
                    if len(fuselist) < total_att_num:
                        fuselist.append(z)
                    else:
                        fuselist[index] = z

            x1 = torch.cat([x, fuselist[index]], dim=1)
            x1 = x1 + self.drop_path1(self.ls1(self.attn(self.norm1(x1))))
            x1 = x1 + self.drop_path2(self.ls2(self.mlp(self.norm2(x1))))
            x = x1[:, :H1 * W1, :]
            x = x.reshape(B, H1, W1, C).contiguous()
            return x


class ContextLayer(nn.Module):
    def __init__(self, in_dim, conv_dim, context_size=[3], context_act=nn.GELU,
                 context_f=True, context_g=True):
        # channel last
        super().__init__()
        self.f = nn.Linear(in_dim, conv_dim) if context_f else nn.Identity()
        self.g = nn.Linear(conv_dim, in_dim) if context_g else nn.Identity()
        self.context_size = context_size
        self.act = context_act() if context_act else nn.Identity()
        if not isinstance(context_size, (list, tuple)):
            context_size = [context_size]
        self.context_list = nn.ModuleList()
        for c_size in context_size:
            self.context_list.append(
                nn.Conv2d(conv_dim, conv_dim, c_size, stride=1, padding=c_size // 2, groups=conv_dim)
            )

    def forward(self, x):
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        out = 0
        for i in range(len(self.context_list)):
            ctx = self.act(self.context_list[i](x))
            out = out + ctx
        out = self.g(out.permute(0, 2, 3, 1).contiguous())
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., bias=True, conv_in_mlp=True,
                 conv_group_dim=4, context_size=3, context_act=nn.GELU,
                 context_f=True, context_g=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.conv_in_mlp = conv_in_mlp
        if self.conv_in_mlp:
            self.conv_group_dim = conv_group_dim
            self.conv_dim = hidden_features // conv_group_dim
            self.context_layer = ContextLayer(in_features, self.conv_dim, context_size,
                                              context_act, context_f, context_g)

        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

        if hidden_features == in_features and conv_group_dim == 1:
            self.expand_dim = False
        else:
            self.expand_dim = True
            self.act = act_layer()
            self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.conv_in_mlp:
            conv_x = self.context_layer(x)  # CTX layer
        x = self.fc1(x)  # FC
        if self.expand_dim:
            x = self.act(x)
            x = self.drop(x)
        if self.conv_in_mlp:
            if self.expand_dim:
                x = x * conv_x.repeat(1, 1, 1, self.conv_group_dim)
            else:
                x = x * conv_x

        x = self.fc2(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., conv_in_mlp=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 bias=True, use_layerscale=False, layerscale_value=1e-4,
                 conv_group_dim=4, context_size=3, context_act=nn.GELU,
                 context_f=True, context_g=True
                 ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop, bias=bias, conv_in_mlp=conv_in_mlp,
                       conv_group_dim=conv_group_dim, context_size=context_size, context_act=context_act,
                       context_f=context_f, context_g=context_g)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_1 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        shortcut = x
        x = shortcut + self.drop_path(self.gamma_1 * self.mlp(self.norm(x)))
        return x


class BasicLayer_dsca(nn.Module):
    def __init__(self, dim, out_dim, depth,
                 mlp_ratio=4., att_ratio=4., conv_in_mlp=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 bias=True, use_layerscale=False, layerscale_value=1e-4,
                 conv_group_dim=4, context_size=3, context_act=nn.GELU,
                 context_f=True, context_g=True,
                 downsample=None, patch_size=3, patch_stride=2, patch_pad=1, patch_norm=True,
                 attention_depth=0, net_type=None, output_layers=None, extra_module=None,
                 train=False, i_layer=None):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.net_type = net_type
        self.output_layers = output_layers
        self.extra_module = extra_module

        if not isinstance(mlp_ratio, (list, tuple)):
            mlp_ratio = [mlp_ratio] * depth
        if not isinstance(conv_group_dim, (list, tuple)):
            conv_group_dim = [conv_group_dim] * depth
        if not isinstance(context_size, (list, tuple)):
            context_size = [context_size] * depth
        # build blocks
        self.blocks = nn.ModuleList([
            BasicBlock(
                dim=dim, mlp_ratio=mlp_ratio[i], conv_in_mlp=conv_in_mlp, drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_layer=act_layer, norm_layer=norm_layer,
                bias=bias, use_layerscale=use_layerscale, layerscale_value=layerscale_value,
                conv_group_dim=conv_group_dim[i], context_size=context_size[i], context_act=context_act,
                context_f=context_f, context_g=context_g
            )
            for i in range(depth)])

        if attention_depth > 0:
            for j in range(attention_depth):
                self.blocks.append(AttentionBlock(
                    dim=dim, mlp_ratio=att_ratio, drop=drop, drop_path=drop_path[depth + j],
                    act_layer=act_layer, norm_layer=norm_layer,
                ))

        if extra_module == 'dsca' and i_layer == 2:
            self.blocks.append(DySepConvAtten(dim))

        if downsample is not None:
            self.downsample = downsample(
                in_chans=dim,
                embed_dim=out_dim,
                patch_size=patch_size,
                patch_stride=patch_stride,
                patch_pad=patch_pad,
                norm_layer=norm_layer if patch_norm else None
            )
        else:
            self.downsample = None

    def forward(self, index, x, z, train, tempinit, fuselist):
        if self.net_type in ["efficientMod_xxs", "efficientMod_xs", "efficientMod_s"]:
            att_num = [0, 0, 0]  # accumulate att num, each block, exclude itself
            total_att_num = 1

        if train or not tempinit:
            att_layer_index = 0
            for i, blk in enumerate(self.blocks):
                if type(blk).__name__ in ["DySepConvAtten"]:
                    if not train:
                        mode = 'test_init'
                    else:
                        mode = 'train'
                    x = blk(z, x, mode=mode, fuselist=fuselist, index=att_num[index] + att_layer_index,
                            total_att_num=total_att_num)
                    att_layer_index += 1
                else:
                    x, z = blk(x), blk(z)
        elif not train and tempinit:  # after temp init
            att_layer_index = 0
            for i, blk in enumerate(self.blocks):
                if type(blk).__name__ in ["DySepConvAtten"]:
                    x = blk(z, x, mode='test', fuselist=fuselist, index=att_num[index] + att_layer_index,
                            total_att_num=total_att_num)
                    att_layer_index += 1
                else:
                    x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
            if train or not tempinit:
                z = self.downsample(z)

        return x, z


class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, depth, mlp_ratio=4., att_ratio=4., conv_in_mlp=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, bias=True, use_layerscale=False, layerscale_value=1e-4,
                 conv_group_dim=4, context_size=3, context_act=nn.GELU, context_f=True, context_g=True, downsample=None,
                 patch_size=3, patch_stride=2, patch_pad=1, patch_norm=True, attention_depth=0, net_type=None,
                 output_layers=None, extra_module=None, train=False, layer_index=None):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.net_type = net_type
        self.output_layers = output_layers
        self.extra_module = extra_module
        self.layer_index = layer_index

        self.z_feat = None

        if not isinstance(mlp_ratio, (list, tuple)):
            mlp_ratio = [mlp_ratio] * depth
        if not isinstance(conv_group_dim, (list, tuple)):
            conv_group_dim = [conv_group_dim] * depth
        if not isinstance(context_size, (list, tuple)):
            context_size = [context_size] * depth
        # build blocks
        self.blocks = nn.ModuleList([
            BasicBlock(
                dim=dim, mlp_ratio=mlp_ratio[i], conv_in_mlp=conv_in_mlp, drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_layer=act_layer, norm_layer=norm_layer,
                bias=bias, use_layerscale=use_layerscale, layerscale_value=layerscale_value,
                conv_group_dim=conv_group_dim[i], context_size=context_size[i], context_act=context_act,
                context_f=context_f, context_g=context_g
            )
            for i in range(depth)])

        if self.layer_index in [1, 2]:
            self.blocks.append(ETM(dim, dim // 32, deploy=not train, residual=True, version='v5'))
        self.blocks.append(OPE(dim, dim, deploy=not train, residual=True, version='v4'))

        if attention_depth > 0:
            for j in range(attention_depth):
                self.blocks.append(AttentionBlock(
                    dim=dim, mlp_ratio=att_ratio, drop=drop, drop_path=drop_path[depth + j],
                    act_layer=act_layer, norm_layer=norm_layer,
                ))

        if downsample is not None:
            self.downsample = downsample(
                in_chans=dim,
                embed_dim=out_dim,
                patch_size=patch_size,
                patch_stride=patch_stride,
                patch_pad=patch_pad,
                norm_layer=norm_layer if patch_norm else None
            )
        else:
            self.downsample = None

    def forward(self, index, x, z, train, tempinit, fuselist):
        if self.net_type in ["efficientMod_xxs", "efficientMod_xs", "efficientMod_s"]:
            att_num = [0, 0, 0]  # accumulate att num, each block, exclude itself
            total_att_num = 1
            if self.output_layers in ["stage4"]:
                att_num = [0, 0, 0, 1]
                total_att_num = 3
            elif self.output_layers in ["stage3"]:
                att_num = [0, 0, 0]
                total_att_num = 1
        elif self.net_type == 'efficientMod_xxs_matt':
            att_num = [0, 0, 1]  # accumulate att num
            total_att_num = 2

        if train or not tempinit:
            att_layer_index = 0
            for i, blk in enumerate(self.blocks):
                if type(blk).__name__ in ["AttentionBlock"]:
                    x = blk(x, z, Fuse=1, Train=train, fuselist=fuselist, index=att_num[index] + att_layer_index,
                            total_att_num=total_att_num)  # x pass att with z
                    z = blk(z, Fuse=0, Train=train)  # z pass att
                    att_layer_index += 1

                elif type(blk).__name__ in ["ETM"]:
                    x = blk(x, z, Train=True)
                elif type(blk).__name__ in ["Interact"]:
                    self.z_feat = z
                    x = blk(x, z)
                else:
                    x, z = blk(x), blk(z)

        elif not train and tempinit:  # after temp init
            att_layer_index = 0
            for i, blk in enumerate(self.blocks):
                if type(blk).__name__ in ["AttentionBlock"]:
                    x = blk(x, Fuse=2, Train=train, fuselist=fuselist, index=att_num[index] + att_layer_index,
                            total_att_num=total_att_num)  # x pass att with z feat
                    att_layer_index += 1

                elif type(blk).__name__ in ["ETM"]:
                    x = blk(x, None, Train=False)
                elif type(blk).__name__ in ["Interact"]:
                    x = blk(x, self.z_feat)
                else:
                    x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
            if train or not tempinit:
                z = self.downsample(z)

        return x, z


class BasicLayer_separate(nn.Module):
    def __init__(self, dim, out_dim, depth,
                 mlp_ratio=4., att_ratio=4., conv_in_mlp=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 bias=True, use_layerscale=False, layerscale_value=1e-4,
                 conv_group_dim=4, context_size=3, context_act=nn.GELU,
                 context_f=True, context_g=True,
                 downsample=None, patch_size=3, patch_stride=2, patch_pad=1, patch_norm=True,
                 attention_depth=0, net_type=None, output_layers=None, extra_module=None,
                 train=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.net_type = net_type
        self.output_layers = output_layers
        self.extra_module = extra_module

        if not isinstance(mlp_ratio, (list, tuple)):
            mlp_ratio = [mlp_ratio] * depth
        if not isinstance(conv_group_dim, (list, tuple)):
            conv_group_dim = [conv_group_dim] * depth
        if not isinstance(context_size, (list, tuple)):
            context_size = [context_size] * depth
        # build blocks
        self.blocks = nn.ModuleList([
            BasicBlock(
                dim=dim, mlp_ratio=mlp_ratio[i], conv_in_mlp=conv_in_mlp, drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_layer=act_layer, norm_layer=norm_layer,
                bias=bias, use_layerscale=use_layerscale, layerscale_value=layerscale_value,
                conv_group_dim=conv_group_dim[i], context_size=context_size[i], context_act=context_act,
                context_f=context_f, context_g=context_g
            )
            for i in range(depth)])

        self.blocks_z = nn.ModuleList([
            BasicBlock(
                dim=dim, mlp_ratio=mlp_ratio[i], conv_in_mlp=conv_in_mlp, drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_layer=act_layer, norm_layer=norm_layer,
                bias=bias, use_layerscale=use_layerscale, layerscale_value=layerscale_value,
                conv_group_dim=conv_group_dim[i], context_size=context_size[i], context_act=context_act,
                context_f=context_f, context_g=context_g
            )
            for i in range(depth)])

        if attention_depth > 0:
            for j in range(attention_depth):
                self.blocks.append(AttentionBlock(
                    dim=dim, mlp_ratio=att_ratio, drop=drop, drop_path=drop_path[depth + j],
                    act_layer=act_layer, norm_layer=norm_layer,
                ))

        if downsample is not None:
            self.downsample = downsample(
                in_chans=dim,
                embed_dim=out_dim,
                patch_size=patch_size,
                patch_stride=patch_stride,
                patch_pad=patch_pad,
                norm_layer=norm_layer if patch_norm else None
            )
            self.downsample_z = downsample(
                in_chans=dim,
                embed_dim=out_dim,
                patch_size=patch_size,
                patch_stride=patch_stride,
                patch_pad=patch_pad,
                norm_layer=norm_layer if patch_norm else None
            )
        else:
            self.downsample = None
            self.downsample_z = None

    def forward(self, index, x, z, train, tempinit, fuselist):
        if self.net_type in ["efficientMod_xxs", "efficientMod_xxs_separate"]:
            att_num = [0, 0, 0]  # accumulate att num, each block, exclude itself
            total_att_num = 1

        if train or not tempinit:
            att_layer_index = 0
            for i, blk in enumerate(self.blocks):
                if type(blk).__name__ in ["AttentionBlock"]:
                    x = blk(x, z, Fuse=1, Train=train, fuselist=fuselist, index=att_num[index] + att_layer_index,
                            total_att_num=total_att_num)  # x pass att with z
                    att_layer_index += 1
                else:
                    x, z = blk(x), self.blocks_z[i](z)
        elif not train and tempinit:  # after temp init
            att_layer_index = 0
            for i, blk in enumerate(self.blocks):
                if type(blk).__name__ in ["AttentionBlock"]:
                    x = blk(x, Fuse=2, Train=train, fuselist=fuselist, index=att_num[index] + att_layer_index,
                            total_att_num=total_att_num)  # x pass att with z feat
                    att_layer_index += 1
                else:
                    x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
            if train or not tempinit:
                z = self.downsample_z(z)

        return x, z

class EfficientMod_stage3(nn.Module):
    def __init__(self,
                 in_chans=3, num_classes=1000,
                 # downsize patch-related
                 patch_size=[4, 3, 3, 3], patch_stride=[4, 2, 2, 2], patch_pad=[0, 1, 1, 1], patch_norm=True,
                 # newwork configuration
                 embed_dim=[64, 128, 256, 512], depths=[2, 2, 6, 2], attention_depth=[0, 0, 0, 0],
                 mlp_ratio=[4.0, 4.0, 4.0, 4.0], att_ratio=[4, 4, 4, 4],
                 conv_in_mlp=[True, True, True, True],
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_layerscale=False, layerscale_value=1e-4,
                 bias=True, drop_rate=0., drop_path_rate=0.0,
                 conv_group_dim=[4, 4, 4, 4], context_size=[3, 3, 3, 3], context_act=nn.GELU,
                 context_f=True, context_g=True, interact=None, net_type=None, output_layers=None,
                 extra_module=None, train=True, stride=None,
                 **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.num_channels = embed_dim[3]
        self.depths = depths
        self.embed_dim_list = None
        self.attention_depth = attention_depth
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim[-1]
        resolution_x = int(256 / 32)
        resolution_z = int(128 / 32)
        self.num_patches_search = resolution_x ** 2
        self.num_patches_template = resolution_z ** 2
        self.tempinit = False
        self.fuselist = []
        self.corrlist = []
        self.interact = interact
        self.net_type = net_type
        self.output_layers = output_layers
        self.extra_module = extra_module
        self.stride = stride

        if self.interact in ["correlation"]:
            self.inter_layers = nn.ModuleList()
            embed_dim_corr = [256, 64]
            for i in range(self.num_layers - 1):
                layer = MobileCorrelation(embed_dim[i + 1], embed_dim_corr[i])
                self.inter_layers.append(layer)

        # split image into patches using either non-overlapped embedding or overlapped embedding
        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            patch_size=patch_size[0],
            patch_stride=patch_stride[0],
            patch_pad=patch_pad[0],
            norm_layer=norm_layer if self.patch_norm else None)

        if self.stride == 64:
            self.input_downsample = PatchEmbed(
                in_chans=embed_dim[0],
                embed_dim=embed_dim[0],
                patch_size=3,
                patch_stride=2,
                patch_pad=1,
                norm_layer=norm_layer if patch_norm else None
            )

        # stochastic depth
        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, (sum(depths) + sum(attention_depth)))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=embed_dim[i_layer],
                               out_dim=embed_dim[i_layer + 1],
                               depth=depths[i_layer],
                               mlp_ratio=mlp_ratio[i_layer],
                               att_ratio=att_ratio[i_layer],
                               conv_in_mlp=conv_in_mlp[i_layer],
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]) + sum(attention_depth[:i_layer]):sum(
                                   depths[:i_layer + 1]) + sum(attention_depth[:i_layer + 1])],
                               act_layer=act_layer, norm_layer=norm_layer,
                               bias=bias, use_layerscale=use_layerscale, layerscale_value=layerscale_value,
                               conv_group_dim=conv_group_dim[i_layer],
                               context_size=context_size[i_layer],
                               context_act=context_act,
                               context_f=context_f,
                               context_g=context_g,
                               downsample=PatchEmbed,
                               patch_size=patch_size[i_layer + 1],
                               patch_stride=patch_stride[i_layer + 1],
                               patch_pad=patch_pad[i_layer + 1],
                               patch_norm=patch_norm,
                               attention_depth=attention_depth[i_layer],
                               net_type=net_type, output_layers=output_layers,
                               extra_module=extra_module, train=train,
                               layer_index=i_layer,
                               )
            self.layers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, z, train, tempinit, fuselist):

        x = self.patch_embed(x.permute(0, 2, 3, 1))

        if self.stride == 64:
            x = self.input_downsample(x)

        if train == True or (train == False and tempinit == False):
            z = self.patch_embed(z.permute(0, 2, 3, 1))

        output_list = []
        for i, layer in enumerate(self.layers):
            x, z = layer(i, x, z, train, tempinit, fuselist)

            if i in [1, 2]:
                output_list.append(x)

        return output_list

    def forward(self, images_list, train=True):
        x = images_list[0]
        z = images_list[1]
        x_list = self.forward_features(x, z, train, self.tempinit, self.fuselist)
        if train == False and self.tempinit == False:
            self.tempinit = True
        return x_list

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, OPE):
                m.switch_to_deploy()
            if isinstance(m, MBoneBlock):
                m.switch_to_deploy()
        print("==============> Switch To Deploy")

class EfficientMod(nn.Module):
    def __init__(self,
                 in_chans=3, num_classes=1000,
                 # downsize patch-related
                 patch_size=[4, 3, 3, 3], patch_stride=[4, 2, 2, 2], patch_pad=[0, 1, 1, 1], patch_norm=True,
                 # newwork configuration
                 embed_dim=[64, 128, 256, 512], depths=[2, 2, 6, 2], attention_depth=[0, 0, 0, 0],
                 mlp_ratio=[4.0, 4.0, 4.0, 4.0], att_ratio=[4, 4, 4, 4],
                 conv_in_mlp=[True, True, True, True],
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_layerscale=False, layerscale_value=1e-4,
                 bias=True, drop_rate=0., drop_path_rate=0.0,
                 conv_group_dim=[4, 4, 4, 4], context_size=[3, 3, 3, 3], context_act=nn.GELU,
                 context_f=True, context_g=True,
                 **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.num_channels = 256
        self.depths = depths
        self.embed_dim_list = None
        self.attention_depth = attention_depth
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim[-1]
        resolution_x = int(256 / 32)
        resolution_z = int(128 / 32)
        self.num_patches_search = resolution_x ** 2
        self.num_patches_template = resolution_z ** 2
        self.tempinit = False
        self.fuselist = []

        # split image into patches using either non-overlapped embedding or overlapped embedding
        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            patch_size=patch_size[0],
            patch_stride=patch_stride[0],
            patch_pad=patch_pad[0],
            norm_layer=norm_layer if self.patch_norm else None)

        # stochastic depth
        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, (sum(depths) + sum(attention_depth)))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=embed_dim[i_layer],
                               out_dim=embed_dim[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                               depth=depths[i_layer],
                               mlp_ratio=mlp_ratio[i_layer],
                               att_ratio=att_ratio[i_layer],
                               conv_in_mlp=conv_in_mlp[i_layer],
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]) + sum(attention_depth[:i_layer]):sum(
                                   depths[:i_layer + 1]) + sum(attention_depth[:i_layer + 1])],
                               act_layer=act_layer, norm_layer=norm_layer,
                               bias=bias, use_layerscale=use_layerscale, layerscale_value=layerscale_value,
                               conv_group_dim=conv_group_dim[i_layer],
                               context_size=context_size[i_layer],
                               context_act=context_act,
                               context_f=context_f,
                               context_g=context_g,
                               downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                               patch_size=patch_size[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                               patch_stride=patch_stride[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                               patch_pad=patch_pad[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                               patch_norm=patch_norm,
                               attention_depth=attention_depth[i_layer]
                               )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, z, train, tempinit, fuselist):
        x = self.patch_embed(x.permute(0, 2, 3, 1))
        if train == True or (train == False and tempinit == False):
            z = self.patch_embed(z.permute(0, 2, 3, 1))
        for i, layer in enumerate(self.layers):
            x, z = layer(i, x, z, train, tempinit, fuselist)
        return x

    def forward(self, images_list, train=True):
        x = images_list[0]
        z = images_list[1]
        x = self.forward_features(x, z, train, self.tempinit, self.fuselist)
        if train == False and self.tempinit == False:
            self.tempinit = True
        B, H1, W1, C = x.shape
        return x.view(B, H1 * W1, C)


from typing import Any, Union, Tuple, List


class SepConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            bias: bool = True,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileCorrelation(nn.Module):
    """
    Mobile Correlation module
    """

    def __init__(self, num_channels: int, num_corr_channels: int = 64, conv_block: str = "regular"):
        super().__init__()
        self.enc = nn.Sequential(
            SepConv(num_channels + num_corr_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, z, x):
        x, z = x.permute(0, 3, 1, 2), z.permute(0, 3, 1, 2)
        b, c, w, h = x.size()
        z = z.reshape(z.size(0), z.size(1), -1)

        s = torch.matmul(z.permute(0, 2, 1), x.view(b, c, -1)).view(b, -1, w, h)
        s = torch.cat([x, s], dim=1)

        s = self.enc(s)

        return s.permute(0, 2, 3, 1)


@register_model
def efficientMod_xxs(pretrained=True, output_layers=None, interact=None, extra_module=None, stride=None, **kwargs):
    depths = [2, 2, 6, 2]
    attention_depth = [0, 0, 1, 2]
    att_ratio = [0, 0, 4, 4]
    mlp_ratio = [
        [1, 6, 1, 6],
        [1, 6, 1, 6],
        [1, 6] * 3,
        [1, 6, 1, 6],
    ]
    context_size = [
        [7] * 10,
        [7] * 10,
        [7] * 20,
        [7] * 10,
    ]
    patch_size = [7, 3, 3, 3]
    patch_stride = [4, 2, 2, 2]
    patch_pad = [3, 1, 1, 1]
    embed_dim = [32, 64, 128, 256]
    conv_in_mlp = [True, True, True, True]
    conv_group_dim = mlp_ratio


    if "3stage1" in output_layers:
        depths = [2, 2, 1]
        model = EfficientMod_stage3(in_chans=3, num_classes=1000,
                                    patch_size=patch_size, patch_stride=patch_stride, patch_pad=patch_pad,
                                    patch_norm=True,
                                    embed_dim=embed_dim, depths=depths, attention_depth=attention_depth,
                                    mlp_ratio=mlp_ratio, att_ratio=att_ratio,
                                    conv_in_mlp=conv_in_mlp,
                                    act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_layerscale=True,
                                    layerscale_value=1e-4,
                                    bias=True, drop_rate=0., drop_path_rate=0.0,
                                    conv_group_dim=conv_group_dim, context_size=context_size, context_act=nn.GELU,
                                    context_f=True, context_g=True, interact=interact, net_type='efficientMod_xxs',
                                    extra_module=extra_module, train=pretrained, stride=stride,
                                    )
        model.embed_dim = embed_dim[0:3]
        if pretrained:
            load_pretrained_partial(model, "pretrained_model/efficientMod/xxs/model_best.pth.tar",
                                    model_type="efficientMod_xxs", out_layer="3stage1", extra_module=extra_module)
    elif "3stage3" in output_layers:
        depths = [2, 2, 3]
        model = EfficientMod_stage3(in_chans=3, num_classes=1000,
                                    patch_size=patch_size, patch_stride=patch_stride, patch_pad=patch_pad,
                                    patch_norm=True,
                                    embed_dim=embed_dim, depths=depths, attention_depth=attention_depth,
                                    mlp_ratio=mlp_ratio, att_ratio=att_ratio,
                                    conv_in_mlp=conv_in_mlp,
                                    act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_layerscale=True,
                                    layerscale_value=1e-4,
                                    bias=True, drop_rate=0., drop_path_rate=0.0,
                                    conv_group_dim=conv_group_dim, context_size=context_size, context_act=nn.GELU,
                                    context_f=True, context_g=True, interact=interact, net_type='efficientMod_xxs',
                                    extra_module=extra_module, train=pretrained, stride=stride,
                                    )
        model.embed_dim = embed_dim[0:3]
        if pretrained:
            load_pretrained_partial(model, "pretrained_model/efficientMod/xxs/model_best.pth.tar",
                                    model_type="efficientMod_xxs", out_layer="3stage3", extra_module=extra_module)

        if pretrained:
            load_pretrained(model, "pretrained_model/efficientMod/xxs/model_best.pth.tar")
    return model


def load_pretrained(model, weights):
    checkpoint = torch.load(
        weights, map_location='cpu')
    state_dict = checkpoint['state_dict']
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print('Load pretrained model from: ' + weights)
    print(f'Missing_Keys: {missing_keys}')
    print(f'Unexpected_Keys: {unexpected_keys}')


def load_pretrained_partial(model, weights, model_type=None, out_layer=None, extra_module=None):

    checkpoint = torch.load(weights, map_location='cpu')
    state_dict = checkpoint['state_dict']
    state_dict_modify = {}

    if model_type == "efficientMod_xxs":
        if out_layer in ["3stage3"]:
            for k, v in state_dict.items():
                if "layers.2.blocks.6" in k:
                    k = k.replace(".blocks.6.", ".blocks.5.")
                    state_dict_modify[k] = v
                elif "layers.2.blocks.3" in k:
                    pass
                elif "layers.2.blocks.4" in k:
                    pass
                elif "layers.2.blocks.5" in k:
                    pass
                else:
                    state_dict_modify[k] = v
        elif out_layer in ["3stage1"]:
            for k, v in state_dict.items():
                if "layers.2.blocks.6" in k:
                    k = k.replace(".blocks.6.", ".blocks.3.")
                    state_dict_modify[k] = v
                elif "layers.2.blocks.1" in k:
                    pass
                elif "layers.2.blocks.2" in k:
                    pass
                elif "layers.2.blocks.3" in k:
                    pass
                else:
                    state_dict_modify[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(state_dict_modify, strict=False)
    print('Load pretrained model from: ' + weights)
    print(f'Missing_Keys: {missing_keys}')
    print(f'Unexpected_Keys: {unexpected_keys}')


def load_pretrained_partial_separate(model, weights, model_type=None, out_layer=None, extra_module=None):

    checkpoint = torch.load(weights, map_location='cpu')
    state_dict = checkpoint['state_dict']
    state_dict_modify = {}

    if model_type == "efficientMod_xxs" and out_layer in ["3stage3"]:
        for k, v in state_dict.items():

            if "layers.2.blocks.6" in k:
                k = k.replace(".blocks.6.", ".blocks.3.")
                state_dict_modify[k] = v
            else:
                state_dict_modify[k] = v
            if "blocks" in k:
                k_z = k.replace("blocks", "blocks_z")
                state_dict_modify[k_z] = v
            if "downsample" in k:
                k_z = k.replace("downsample", "downsample_z")
                state_dict_modify[k_z] = v
            if "patch_embed" in k:
                k_z = k.replace("patch_embed", "patch_embed_z")
                state_dict_modify[k_z] = v
    missing_keys, unexpected_keys = model.load_state_dict(state_dict_modify, strict=False)
    print('Load pretrained model from: ' + weights)
    print(f'Missing_Keys: {missing_keys}')
    print(f'Unexpected_Keys: {unexpected_keys}')
