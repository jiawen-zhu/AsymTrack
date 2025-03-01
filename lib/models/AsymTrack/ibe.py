import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn import Parameter


class DySepConvAtten(nn.Module):
    def __init__(self, hidden_dim):
        super(DySepConvAtten, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_proposals = 320
        self.kernel_size = 3
        self.weight_linear = nn.Linear(self.hidden_dim, self.num_proposals + self.kernel_size)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, query, value, mode='train', fuselist=None, index=0, total_att_num=None):
        # query, value: z,x

        if mode == 'test_init':
            fuselist.append(query)
        elif mode == 'test':
            query = fuselist[index]

        B, Hx, Wx, C = value.size()
        value = value.reshape(B, Hx * Wx, C).contiguous()
        B, Hz, Wz, C = query.size()
        query = query.reshape(B, Hz * Wz, C).contiguous()
        query = value = torch.cat([value, query], dim=1)

        B, N, C = query.shape

        # dynamic depth-wise conv
        dy_conv_weight = self.weight_linear(query)
        dy_depth_conv_weight = dy_conv_weight[:, :, :self.kernel_size].view(B, self.num_proposals, 1, self.kernel_size)
        dy_point_conv_weight = dy_conv_weight[:, :, self.kernel_size:].view(B, self.num_proposals, self.num_proposals,1)
        res = []
        value = value.unsqueeze(1)
        for i in range(B):
            out = F.relu(F.conv1d(input=value[i], weight=dy_depth_conv_weight[i], groups=N, padding="same"))
            out = F.conv1d(input=out, weight=dy_point_conv_weight[i], padding='same')
            res.append(out)
        point_out = torch.cat(res, dim=0)
        point_out = self.norm(point_out)
        point_out = point_out[:, :Hx * Wx, :].reshape(B, Hx, Wx, C).contiguous()

        return point_out


class OPE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, deploy=False,
                 residual=False, version=None, **kwargs):
        super(OPE, self).__init__()
        self.deploy = deploy
        self.residual = residual
        self.version = version
        self.bias = bias

        if False:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.num_conv_branches = 3
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias))
            self.rbr_conv = nn.ModuleList(rbr_conv)
            if self.version in ['v5']:
                self.rbr_scale = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                           dilation=dilation, groups=groups, bias=bias)

            self.theta = Parameter(torch.zeros([1]))

        self.ln = nn.LayerNorm(out_channels)
        self.act = nn.ReLU()

        # layer scale
        layerscale_value = 0.1
        self.ibe_scale = nn.Parameter(layerscale_value * torch.zeros((out_channels)), requires_grad=True)

    def merge_bn(self, conv, bn):
        conv_w = conv
        conv_b = torch.zeros_like(bn.running_mean)

        factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        weight = nn.Parameter(conv_w *
                              factor.reshape([conv_w.shape[0], 1, 1, 1]))
        bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
        return weight, bias

    def switch_to_deploy(self):
        if self.version in ['v4']:
            theta = F.sigmoid(self.theta)
            weight_final = 0
            for idx in range(self.num_conv_branches):
                weight_conv = self.rbr_conv[idx].weight
                kernel_diff = theta * self.rbr_conv[idx].weight.sum(2).sum(2)[:, :, None, None]
                weight_diff = nn.ZeroPad2d(1)(kernel_diff)
                weight_final += (weight_conv - weight_diff)
            self.bias = False

            self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_conv[0].in_channels,
                                         out_channels=self.rbr_conv[0].out_channels,
                                         kernel_size=self.rbr_conv[0].kernel_size, stride=self.rbr_conv[0].stride,
                                         padding=self.rbr_conv[0].padding, dilation=self.rbr_conv[0].dilation,
                                         groups=self.rbr_conv[0].groups,
                                         bias=self.bias)

            self.rbr_reparam.weight.data = weight_final
            for para in self.parameters():
                para.detach_()
            delattr(self, 'rbr_conv')
            delattr(self, 'theta')

    def forward(self, x_ori):
        x = x_ori.permute(0, 3, 1, 2).contiguous()

        if self.deploy:
            out_normal = self.rbr_reparam(x)
            outs = self.act(self.ln(out_normal.permute(0, 2, 3, 1).contiguous()))

            # add res
            if self.residual:
                outs = self.ibe_scale.unsqueeze(0).unsqueeze(0).unsqueeze(0) * outs + x_ori
        else:
            out_normal, out_diff = 0, 0
            for ix in range(self.num_conv_branches):
                out_normal += self.rbr_conv[ix](x)
                kernel_diff = self.rbr_conv[ix].weight.sum(2).sum(2)
                kernel_diff = kernel_diff[:, :, None, None]
                out_diff += F.conv2d(input=x, weight=kernel_diff, stride=self.rbr_conv[ix].stride, padding=0,
                                     groups=self.rbr_conv[ix].groups)

            theta = F.sigmoid(self.theta)

            outs = self.act(self.ln((out_normal - theta * out_diff).permute(0, 2, 3, 1).contiguous()))

            # add res
            if self.residual:
                outs = self.ibe_scale.unsqueeze(0).unsqueeze(0).unsqueeze(0) * outs + x_ori
        return outs
