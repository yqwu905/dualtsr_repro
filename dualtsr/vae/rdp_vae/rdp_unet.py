import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def checkpoint(func, inputs, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        return torch_checkpoint(func, *inputs, use_reentrant=False)
    else:
        return func(*inputs)


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        # N, C, H, W = x.size()
        # mu = x.mean(1, keepdim=True)
        # var = (x - mu).pow(2).mean(1, keepdim=True)
        # y = (x - mu) / (var + self.eps).sqrt()
        # y = self.weight.view(1, C, 1, 1) * y + self.bias.view(1, C, 1, 1)
        # return y
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def __init__(self, sg_mode='half'):
        super().__init__()
        self.sg_mode = sg_mode  # sg_mode = 'half' or 'odd_even'

    def forward(self, x):
        if self.sg_mode == 'half':
            x1, x2 = x.chunk(2, dim=1)
        else:
            x1, x2 = x[:, 0::2, :, :], x[:, 1::2, :, :]

        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, mode='base', DW_Expand=2, FFN_Expand=2, drop_out_rate=0., ccb_flag='before',
                 padding_mode='zeros'):
        super().__init__()
        self.mode = mode
        self.ccb_flag = ccb_flag
        if mode == 'new':
            DW_Expand = 1
            FFN_Expand = 1
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel, bias=True, padding_mode=padding_mode)
        if mode == 'new':
            self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                                   bias=True)
        else:
            self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                                   groups=1, bias=True)

        # SimpleGate
        if self.ccb_flag == 'before':
            self.sg = SimpleGate(sg_mode='half')
        else:
            self.lrelu = nn.LeakyReLU(inplace=True, negative_slope=0.0078125)
            self.sg = SimpleGate(sg_mode='odd_even')

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        if mode == 'new':
            self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1,
                                   groups=1, bias=True)
        else:
            self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                                   groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        if mode == 'new':
            self.square_bias0 = torch.nn.Parameter(torch.zeros(1, c, 1, 1), requires_grad=True)
            self.linear_bias0 = torch.nn.Parameter(torch.zeros(1, c, 1, 1), requires_grad=True)
            self.square_bias1 = torch.nn.Parameter(torch.zeros(1, c, 1, 1), requires_grad=True)
            self.linear_bias1 = torch.nn.Parameter(torch.zeros(1, c, 1, 1), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)

        if self.ccb_flag == 'before':
            if self.mode == 'new':
                x = x * (x + self.linear_bias0) + self.square_bias0
            else:
                x = self.sg(x)
        else:
            if self.mode == 'new':
                x = self.lrelu(x)
                x = x * (x + self.linear_bias0) + self.square_bias0
            else:
                x = self.lrelu(x)
                x = self.sg(x)
        # x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        if self.ccb_flag == 'before':
            if self.mode == 'new':
                x = x * (x + self.linear_bias0) + self.square_bias0
            else:
                x = self.sg(x)
        else:
            if self.mode == 'new':
                x = self.lrelu(x)
                x = x * (x + self.linear_bias0) + self.square_bias0
            else:
                x = self.lrelu(x)
                x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlock_enhance(nn.Module):
    def __init__(self, c, mode='base', DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.mode = mode
        if mode == 'new':
            DW_Expand = 1
            FFN_Expand = 1
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        if mode == 'new':
            self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                                   bias=True)
        else:
            self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                                   groups=1, bias=True)

        # SimpleGate
        self.lrelu = nn.LeakyReLU(inplace=True, negative_slope=0.0078125)
        self.sg = SimpleGate(sg_mode='odd_even')

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=ffn_channel, kernel_size=3, padding=1, stride=1,
                               groups=ffn_channel,
                               bias=True)
        if mode == 'new':
            self.conv6 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1,
                                   groups=1, bias=True)
        else:
            self.conv6 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                                   groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        if mode == 'new':
            self.square_bias0 = torch.nn.Parameter(torch.zeros(1, c, 1, 1), requires_grad=True)
            self.linear_bias0 = torch.nn.Parameter(torch.zeros(1, c, 1, 1), requires_grad=True)
            self.square_bias1 = torch.nn.Parameter(torch.zeros(1, c, 1, 1), requires_grad=True)
            self.linear_bias1 = torch.nn.Parameter(torch.zeros(1, c, 1, 1), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        if self.mode == 'new':
            x = self.lrelu(x)
            x = x * (x + self.linear_bias0) + self.square_bias0
        else:
            x = self.lrelu(x)
            x = self.sg(x)
        # x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.conv5(x)
        if self.mode == 'new':
            x = self.lrelu(x)
            x = x * (x + self.linear_bias0) + self.square_bias0
        else:
            x = self.lrelu(x)
            x = self.sg(x)
        x = self.conv6(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class RdpUnet(nn.Module):

    def __init__(
            self,
            width=16,
            middle_blk_num=2,
            enc_blk_nums=[1, 1, 1],
            dec_blk_nums=[1, 1, 1],
            use_checkpoint=False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.middle_blks = nn.ModuleList()

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan, ccb_flag='after') for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan, ccb_flag='after') for _ in range(middle_blk_num)])

        for num in dec_blk_nums[:-1]:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan, ccb_flag='after') for _ in range(num)]))

        self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
        chan = chan // 2
        self.decoders.append(nn.Sequential(*[NAFBlock_enhance(chan) for _ in range(dec_blk_nums[-1])]))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.use_checkpoint and self.training)

    def _forward(self, x):
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks[0](x)
        x = self.middle_blks[1](x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        return x


if __name__ == "__main__":
    model = RdpUnet()
    model.eval()
    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(model, (16, 1024, 1024), as_strings=True, print_per_layer_stat=True)
    print(f"模型 FLOPs: {macs}")
    print(f"模型参数量: {params}")
    # save onnx
    # out_path = "model_zoo/jdd_hdr_raw2bgr/"
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)
    # export_onnx_file = out_path + "jdd_hdr_raw2bgr.onnx"
    # torch.onnx.export(model, (input1, input2, input3), export_onnx_file, do_constant_folding=True, opset_version=10)
    # print(f"{export_onnx_file} is saved")
