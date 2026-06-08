# coding=gb2312
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import time
import warnings
warnings.filterwarnings("ignore")

class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1, act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

    def flops(self, HW):
        flops = 0
        flops += HW * self.in_channels * self.kernel_size ** 2 / self.stride ** 2
        flops += HW * self.in_channels * self.out_channels
        print("SeqConv2d:{%.2f}" % (flops / 1e9))
        return flops


class FastLeff(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv = nn.Sequential(SepConv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1), act_layer())
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x = self.linear1(x)
        x = rearrange(x, ' b (h w) (c) -> b c h w', h=hh, w=hh)
        x = self.dwconv(x)
        x = rearrange(x, ' b c h w -> b (h w) c ', h=hh, w=hh)
        x = self.linear2(x)
        return x

    def flops(self, H, W):
        flops = 0
        flops += H*W*self.dim*self.hidden_dim
        flops += H*W*self.hidden_dim*3*3
        flops += H*W*self.hidden_dim*self.dim
        print("Leff:{%.2f}" % (flops / 1e9))
        return flops

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride
    )



class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()

        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img




class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out

    def flops(self, H, W):
        flops = H * W * self.in_channel * self.out_channel * (
                    3 * 3 + 1) + H * W * self.out_channel * self.out_channel * 3 * 3
        return flops



class UNet(nn.Module):
    def __init__(self, block=ConvBlock, dim=32):
        super(UNet, self).__init__()
        self.dim = dim
        self.ConvBlock1 = ConvBlock(3, dim, strides=1)
        self.pool1 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim*2, strides=1)
        self.pool2 = nn.Conv2d(dim*2, dim*2, kernel_size=4, stride=2, padding=1)

        self.ConvBlock3 = block(dim*2, dim*4, strides=1)
        self.pool3 = nn.Conv2d(dim*4, dim*4, kernel_size=4, stride=2, padding=1)

        self.ConvBlock4 = block(dim*4, dim*8, strides=1)
        self.pool4 = nn.Conv2d(dim*8, dim*8, kernel_size=4, stride=2, padding=1)


        self.ConvBlock5 = block(dim*8, dim*16, strides=1)


        self.upv6 = nn.ConvTranspose2d(dim*16, dim*8, 2, stride=2)
        self.ConvBlock6 = block(dim*16, dim*8, strides=1)

        self.upv7 = nn.ConvTranspose2d(dim*8, dim*4, 2, stride=2)
        self.ConvBlock7 = block(dim*8, dim*4, strides=1)

        self.upv8 = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.ConvBlock8 = block(dim*4, dim*2, strides=1)

        self.upv9 = nn.ConvTranspose2d(dim*2, dim, 2, stride=2)
        self.ConvBlock9 = block(dim*2, dim, strides=1)

        self.conv10 = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        out = x + conv10

        return out

    def flops(self, H, W):
        flops = 0
        flops += self.ConvBlock1.flops(H, W)
        flops += H / 2 * W / 2 * self.dim * self.dim * 4 * 4
        flops += self.ConvBlock2.flops(H / 2, W / 2)
        flops += H / 4 * W / 4 * self.dim * 2 * self.dim * 2 * 4 * 4
        flops += self.ConvBlock3.flops(H / 4, W / 4)
        flops += H / 8 * W / 8 * self.dim * 4 * self.dim * 4 * 4 * 4
        flops += self.ConvBlock4.flops(H / 8, W / 8)
        flops += H / 16 * W / 16 * self.dim * 8 * self.dim * 8 * 4 * 4

        flops += self.ConvBlock5.flops(H / 16, W / 16)

        flops += H / 8 * W / 8 * self.dim * 16 * self.dim * 8 * 2 * 2
        flops += self.ConvBlock6.flops(H / 8, W / 8)
        flops += H / 4 * W / 4 * self.dim * 8 * self.dim * 4 * 2 * 2
        flops += self.ConvBlock7.flops(H / 4, W / 4)
        flops += H / 2 * W / 2 * self.dim * 4 * self.dim * 2 * 2 * 2
        flops += self.ConvBlock8.flops(H / 2, W / 2)
        flops += H * W * self.dim * 2 * self.dim * 2 * 2
        flops += self.ConvBlock9.flops(H, W)

        flops += H * W * self.dim * 3 * 3 * 3
        return flops





class LPU(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LPU,self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=True)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        result = (self.depthwise(x) + x).flatten(2).transpose(1, 2).contiguous()
        return result

    def flops(self, H, W):
        flops = 0
        flops += H * W * self.out_channels * 3 * 3
        return flops




class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, padding=1, bias=True, groups=embed_dim))
        self.s = s

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        H = H or int(math.sqrt(N))
        W = W or int(math.sqrt(N))
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]




class SELayer(nn.Module):
    def __init__(self, channel, reduction=2, bias=False):
        super(SELayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self):
        flops = 0
        flops += self.channel * self.channel * self.k_size

        return flops




class eca_layer_1d(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self):
        flops = 0
        flops += self.channel * self.channel * self.k_size

        return flops




class ConvProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout=0.,
                 last_stage=False, bias=True):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q, k, v

    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = 0
        flops += self.to_q.flops(q_L)
        flops += self.to_k.flops(kv_L)
        flops += self.to_v.flops(kv_L)
        return flops




class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v

    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = q_L * self.dim * self.inner_dim + kv_L * self.dim * self.inner_dim * 2
        return flops





class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if token_projection == 'conv':
            self.qkv = ConvProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        elif token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        q, k, v = self.qkv(x)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

    def flops(self, H, W):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        N = self.win_size[0] * self.win_size[1]
        nW = H * W / N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(H * W, H * W)

        # attn = (q @ k.transpose(-2, -1))

        flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)

        # x = self.proj(x)
        flops += nW * N * self.dim * self.dim
        print("W-MSA:{%.2f}" % (flops / 1e9))
        return flops




def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows



def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x





class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, img_size = (128, 128)):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H, W = img_size[0], img_size[1]
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H / 2 * W / 2 * self.in_channel * self.out_channel * 4 * 4
        print("Downsample:{%.2f}" % (flops / 1e9))
        return flops


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, img_size = (128, 128)):
        B, L, C = x.shape
        H, W = img_size[0], img_size[1]
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * 2 * W * 2 * self.in_channel * self.out_channel * 2 * 2
        print("Upsample:{%.2f}" % (flops / 1e9))
        return flops




class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * W * self.in_channel * self.out_channel * 3 * 3

        if self.norm is not None:
            flops += H * W * self.out_channel
        print("Input_proj:{%.2f}" % (flops / 1e9))
        return flops



class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, img_size = (128, 128)):
        B, L, C = x.shape
        H, W = img_size[0], img_size[1]
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * W * self.in_channel * self.out_channel * 3 * 3

        if self.norm is not None:
            flops += H * W * self.out_channel
        print("Output_proj:{%.2f}" % (flops / 1e9))
        return flops


class PModule(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1)
        #self.selayer = SELayer(hidden_dim//2)
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim//2, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x, img_size=(128, 128)):
        # bs x hw x c
        hh,ww = img_size[0],img_size[1]
        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=ww)

        x1,x2 = self.dwconv(x).chunk(2, dim=1)
        x3 = x1 * x2
        #x4=self.selayer(x3)
        # flaten
        x3 = rearrange(x3, ' b c h w -> b (h w) c', h=hh, w=ww)
        y = self.linear2(x3)

        return y

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.dim * self.hidden_dim
        # dwconv
        flops += H * W * self.hidden_dim * 3 * 3
        flops += H * W * self.hidden_dim//2
        # fc2
        flops += H * W * self.hidden_dim//2 * self.dim
        print("LeFF:{%.2f}" % (flops / 1e9))
        # eca
        return flops



class BasicUniEncoderBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear'):
        super(BasicUniEncoderBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.PModule = PModule(dim=dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm2 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def with_pos_embed(self,tensor,pos):
        return tensor if pos is None else tensor+pos

    def forward(self, x, img_size=(128, 128)):
        shortcut = x
        B, L, C = x.shape
        H, W = img_size
        x = self.norm2(x)
        x = x.view(B, H, W, C)

        x_windows = window_partition(x, self.win_size)
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)

        attn_windows = self.attn(x_windows)

        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        restored_x = window_reverse(attn_windows, self.win_size, H, W)
        restored_x = restored_x.view(B, H * W, C)
        x = shortcut + self.drop_path(restored_x)
        x = x + self.drop_path(self.PModule(self.norm1(x), img_size=img_size))
        return x


    def flops(self):
        flops = 0
        H, W = self.input_resolution

        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.PModule.flops(H, W)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops





class BasicUniDecoderBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear'):
        super(BasicUniDecoderBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm2 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.PModule = PModule(dim=dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, img_size=(128, 128)):

        shortcut = x
        B, L, C = x.shape
        H, W = img_size[0], img_size[1]

        x = self.norm2(x)
        x = x.view(B, H, W, C)

        x_windows = window_partition(x, self.win_size)
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)

        attn_windows = self.attn(x_windows)

        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        restored_x = window_reverse(attn_windows, self.win_size, H, W)
        restored_x = restored_x.view(B, H*W, C)
        x = shortcut + self.drop_path(restored_x)

        x = x + self.drop_path(self.PModule(self.norm1(x), img_size=img_size))
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        if self.cross_modulator is not None:
            flops += self.dim * H * W
            flops += self.cross_attn.flops(H * W, self.win_size * self.win_size)

        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H, W)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops





class BasicUniEncoderLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear'):

        super(BasicUniEncoderLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        block = []
        for i in range(depth):
            block.append(BasicUniEncoderBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                win_size=win_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                token_projection=token_projection,
            ))
        self.blocks = nn.ModuleList(block)


    def forward(self, x, img_size=(128,128)):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, img_size=img_size)
            else:
                x = blk(x, img_size=img_size)
        return x


    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops



class BasicUniDecoderLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear'):
        super(BasicUniDecoderLayer, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        block = []
        for i in range(depth):
            block.append(BasicUniDecoderBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                win_size=win_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                token_projection=token_projection,
            ))
        self.blocks = nn.ModuleList(block)

    def forward(self, x,img_size=(128,128)):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, img_size=img_size)
            else:
                x = blk(x,img_size=img_size)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops



class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SpatialAttention(nn.Module):

    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        att = self.conv(combined)
        return x*self.sigmoid(att)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        mid_channels = max(1, channels // reduction)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        return x * channel_att


class StructExtractor(nn.Module):

    def __init__(self, in_channels=3, out_channels=32, kernel_sizes=[3,5,7], sigma_range=(0.5, 2.0)):
        super(StructExtractor, self).__init__()
        assert isinstance(kernel_sizes, (list, tuple)) and len(kernel_sizes) >= 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.sigma_range = sigma_range

        self.gauss_convs = nn.ModuleList()
        if isinstance(sigma_range, (list, tuple)):
            sigmas = torch.linspace(sigma_range[0], sigma_range[1], steps=len(kernel_sizes)).tolist()
        else:
            sigmas = [float(sigma_range)] * len(kernel_sizes)

        for ks, sigma in zip(kernel_sizes, sigmas):
            # depthwise conv: out_channels = in_channels, groups = in_channels
            conv = nn.Conv2d(in_channels, in_channels, kernel_size=ks, padding=ks//2, groups=in_channels, bias=False)
            kernel = self._create_gaussian_kernel(ks, sigma)  # [ks, ks]
            weight = kernel.view(1, 1, ks, ks).repeat(in_channels, 1, 1, 1)
            conv.weight.data.copy_(weight)
            conv.weight.requires_grad = False
            self.gauss_convs.append(conv)

        total_feats = in_channels * len(kernel_sizes)
        self.adapter = nn.Sequential(
            nn.Conv2d(total_feats, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        # self.use_sobel = False

    def _create_gaussian_kernel(self, kernel_size, sigma):
        # coords centered at 0
        coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
        x_grid = coords.view(1, -1).repeat(kernel_size, 1)
        y_grid = coords.view(-1, 1).repeat(1, kernel_size)
        kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * (sigma**2)))
        kernel = kernel / kernel.sum()
        return kernel

    def forward(self, x):
        """
        x: [B, C=3, H, W]
        return: struct_feat [B, out_channels, H, W]
        """
        B, C, H, W = x.shape
        high_feats = []
        for conv in self.gauss_convs:
            blurred = conv(x)                     # [B, C, H, W]
            high = x - blurred
            high_feats.append(high)

        multi_high = torch.cat(high_feats, dim=1)  # [B, C * n_scales, H, W]

        struct_feat = self.adapter(multi_high)     # [B, out_channels, H, W]
        return struct_feat






class FeatureFusionModule(nn.Module):

    def __init__(self, in_channels, struct_channels):
        super().__init__()
        self.struct_adapter = nn.Sequential(
            nn.Conv2d(struct_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, struct_feat, img_size):
        B, L, C = x.shape
        H, W = img_size
        x = x.transpose(1, 2).view(B, C, H, W)

        struct_feat = F.interpolate(struct_feat, size=(H, W), mode='bilinear')
        struct_feat = self.struct_adapter(struct_feat)

        calibration_map = self.gate(x)
        enhanced = x * (1 + calibration_map) + struct_feat

        return enhanced.flatten(2).transpose(1, 2)



class PhasorFormer(nn.Module):
    def __init__(self, img_size=256, in_chans=3,
                 embed_dim=32, depths=[2,2,2,2,2,2,2,2,2], num_heads=[1,2,4,8,16,16,8,4,1],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, patch_norm=True, using_checkpoint=False, token_projection='linear',
                 token_mlp='leff', downsample=Downsample, upsample=Upsample,
                 **kwargs
                 ):
        super(PhasorFormer, self).__init__()
        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.win_size = win_size
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.patch_norm = patch_norm

        self.struct_fusion = StructExtractor(
            in_channels=3,
            out_channels=embed_dim,
            kernel_sizes=[3, 5, 7],
            sigma_range=(0.5, 2.0)
        )

        self.struct_fusions = nn.ModuleList([
            FeatureFusionModule(embed_dim, 32),
            FeatureFusionModule(embed_dim * 2, 32),
            FeatureFusionModule(embed_dim * 4, 32),
            FeatureFusionModule(embed_dim * 8, 32),
            FeatureFusionModule(embed_dim * 16, 32)
        ])
        self.decoder_fusions = nn.ModuleList([
            FeatureFusionModule(embed_dim * 16, 32),
            FeatureFusionModule(embed_dim * 8, 32),
            FeatureFusionModule(embed_dim * 4, 32),
            FeatureFusionModule(embed_dim * 2 + 1, 32)
        ])

        self.input_proj = InputProj(in_channel=4, out_channel=embed_dim, kernel_size=3, stride=1,
                                  act_layer=nn.LeakyReLU)

        self.output_proj = OutputProj(in_channel=2*embed_dim+2, out_channel=in_chans, kernel_size=3, stride=1)
        self.encodelayer_0=BasicUniEncoderLayer(dim=embed_dim,
                                                output_dim=embed_dim,
                                                input_resolution=(img_size,img_size),
                                                depth=depths[0],
                                                num_heads=num_heads[0],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                norm_layer=norm_layer,
                                                use_checkpoint=using_checkpoint,
                                                token_projection=token_projection,
                                                )
        self.downsample_0 = downsample(embed_dim, embed_dim*2)

        self.encodelayer_1=BasicUniEncoderLayer(dim=embed_dim*2,
                                                output_dim=embed_dim*2,
                                                input_resolution=(img_size//2, img_size//2),
                                                depth=depths[1],
                                                num_heads=num_heads[1],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                norm_layer=norm_layer,
                                                use_checkpoint=using_checkpoint,
                                                token_projection=token_projection,
                                                )
        self.downsample_1 = downsample(embed_dim*2, embed_dim*4)

        self.encodelayer_2=BasicUniEncoderLayer(dim=embed_dim*4,
                                                output_dim=embed_dim*4,
                                                input_resolution=(img_size//4,img_size//4),
                                                depth=depths[2],
                                                num_heads=num_heads[2],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                norm_layer=norm_layer,
                                                use_checkpoint=using_checkpoint,
                                                token_projection=token_projection,
                                                )
        self.downsample_2 = downsample(embed_dim*4, embed_dim*8)

        self.encodelayer_3=BasicUniEncoderLayer(dim=embed_dim*8,
                                                output_dim=embed_dim*8,
                                                input_resolution=(img_size//8,img_size//8),
                                                depth=depths[3],
                                                num_heads=num_heads[3],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                norm_layer=norm_layer,
                                                use_checkpoint=using_checkpoint,
                                                token_projection=token_projection,
                                                )
        self.downsample_3 = downsample(embed_dim*8,embed_dim*16)


        #bottle
        self.conv=BasicUniEncoderLayer(dim=embed_dim*16,
                                       output_dim=embed_dim*16,
                                       input_resolution=(img_size//16,img_size//16),
                                       depth=depths[4],
                                       num_heads=num_heads[4],
                                       win_size=win_size,
                                       mlp_ratio=self.mlp_ratio,
                                       qkv_bias=qkv_bias,
                                       qk_scale=qk_scale,
                                       drop=drop_rate,
                                       attn_drop=attn_drop_rate,
                                       norm_layer=norm_layer,
                                       use_checkpoint=using_checkpoint,
                                       token_projection=token_projection)

        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8)
        self.decodelayer_0=BasicUniDecoderLayer(dim=embed_dim*16,
                                                output_dim=embed_dim*16,
                                                input_resolution=(img_size//8,img_size//8),
                                                depth=depths[5],
                                                num_heads=num_heads[5],
                                                win_size=win_size,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                norm_layer=norm_layer,
                                                use_checkpoint=using_checkpoint,
                                                token_projection=token_projection)

        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4)
        self.decodelayer_1=BasicUniDecoderLayer(dim=embed_dim*8,
                                                output_dim=embed_dim*8,
                                                input_resolution=(img_size//4,img_size//4),
                                                depth=depths[6],
                                                num_heads=num_heads[6],
                                                win_size=win_size,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                norm_layer=norm_layer,
                                                use_checkpoint=using_checkpoint,
                                                token_projection=token_projection)

        self.upsample_2 = upsample(embed_dim * 8, embed_dim * 2)
        self.decodelayer_2=BasicUniDecoderLayer(dim=embed_dim*4,
                                                output_dim=embed_dim*4,
                                                input_resolution=(img_size//2,img_size//2),
                                                depth=depths[7],
                                                num_heads=num_heads[7],
                                                win_size=win_size,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                norm_layer=norm_layer,
                                                use_checkpoint=using_checkpoint,
                                                token_projection=token_projection)

        self.upsample_3 = upsample(embed_dim*4,embed_dim)
        self.decodelayer_3=BasicUniDecoderLayer(dim=embed_dim*2+1,
                                                output_dim=embed_dim*2+1,
                                                input_resolution=(img_size,img_size),
                                                depth=depths[8],
                                                num_heads=num_heads[8],
                                                win_size=win_size,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                norm_layer=norm_layer,
                                                use_checkpoint=using_checkpoint,
                                                token_projection=token_projection)

        self.apply(self._init_weights)


    def _init_weights(self,m):
        if isinstance(m,nn.Linear):
            trunc_normal_(m.weight,std=.02)
            if isinstance(m,nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias,0)
            nn.init.constant_(m.weight,1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x, xm):
        struct_feat = self.struct_fusion(x)  # [B, embed_dim, H, W]
        struct_seq = struct_feat.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]

        # Input Projection
        H, W = (x.shape[2], x.shape[3])
        y = self.input_proj(torch.cat([x, xm], 1))
        y = self.pos_drop(y)
        # Encoder encodelayer -> downsample
        conv0 = self.encodelayer_0(y, img_size=(H, W))
        conv0 = self.struct_fusions[0](conv0, struct_feat, img_size=(H, W))
        pool0 = self.downsample_0(conv0, img_size=(H, W))

        conv1 = self.encodelayer_1(pool0, img_size=(H // 2, W // 2))
        conv1 = self.struct_fusions[1](conv1, struct_feat, img_size=(H // 2, W // 2))
        pool1 = self.downsample_1(conv1, img_size=(H // 2, W // 2))

        conv2 = self.encodelayer_2(pool1, img_size=(H // 4, W // 4))
        conv2 = self.struct_fusions[2](conv2, struct_feat, img_size=(H // 4, W // 4))
        pool2 = self.downsample_2(conv2, img_size=(H // 4, W // 4))

        conv3 = self.encodelayer_3(pool2, img_size=(H // 8, W // 8))
        conv3 = self.struct_fusions[3](conv3, struct_feat, img_size=(H // 8, W // 8))
        pool3 = self.downsample_3(conv3, img_size=(H // 8, W // 8))

        # Bottleneck
        conv4 = self.conv(pool3, img_size=(H // 16, W // 16))
        conv4 = self.struct_fusions[4](conv4, struct_feat, img_size=(H // 16, W // 16))

        # Decoder upsample -> cat -> decodelayer
        up0 = self.upsample_0(conv4, img_size=(H // 16, W // 16))
        deconv0 = torch.cat([up0, conv3], -1)
        deconv0 = self.decodelayer_0(deconv0, img_size=(H // 8, W // 8))
        deconv0 = self.decoder_fusions[0](deconv0, struct_feat, img_size=(H // 8, W // 8))

        up1 = self.upsample_1(deconv0, img_size=(H // 8, W // 8))
        deconv1 = torch.cat([up1, conv2], -1)
        deconv1 = self.decodelayer_1(deconv1, img_size=(H // 4, W // 4))
        deconv1 = self.decoder_fusions[1](deconv1, struct_feat, img_size=(H // 4, W // 4))

        up2 = self.upsample_2(deconv1, img_size=(H // 4, W // 4))
        deconv2 = torch.cat([up2, conv1], -1)
        deconv2 = self.decodelayer_2(deconv2, img_size=(H // 2, W // 2))
        deconv2 = self.decoder_fusions[2](deconv2, struct_feat, img_size=(H // 2, W // 2))

        up3 = self.upsample_3(deconv2, img_size=(H // 2, W // 2))
        deconv3 = torch.cat([up3, conv0, xm.flatten(2).transpose(1, 2)], -1)
        deconv3 = self.decodelayer_3(deconv3, img_size=(H, W))
        deconv3 = self.decoder_fusions[3](deconv3, struct_feat, img_size=(H, W))

        # Output Projection
        y = self.output_proj(torch.cat([deconv3, xm.flatten(2).transpose(1, 2).contiguous()], -1), img_size=(H, W))
        return x + y

    def flops(self):
        flops = 0
        # Input Projection
        flops += self.input_proj.flops(self.reso, self.reso)
        # Encoder
        flops += self.encodelayer_0.flops() + self.downsample_0.flops(self.reso, self.reso)
        flops += self.encodelayer_1.flops() + self.downsample_1.flops(self.reso // 2, self.reso // 2)
        flops += self.encodelayer_2.flops() + self.downsample_2.flops(self.reso // 2 ** 2, self.reso // 2 ** 2)
        flops += self.encodelayer_3.flops() + self.downsample_3.flops(self.reso // 2 ** 3, self.reso // 2 ** 3)

        # Bottleneck
        flops += self.conv.flops()

        # Decoder
        flops += self.upsample_0.flops(self.reso // 2 ** 4, self.reso // 2 ** 4) + self.decodelayer_0.flops()
        flops += self.upsample_1.flops(self.reso // 2 ** 3, self.reso // 2 ** 3) + self.decodelayer_1.flops()
        flops += self.upsample_2.flops(self.reso // 2 ** 2, self.reso // 2 ** 2) + self.decodelayer_2.flops()
        flops += self.upsample_3.flops(self.reso // 2, self.reso // 2) + self.decodelayer_3.flops()

        # Output Projection
        flops += self.output_proj.flops(self.reso, self.reso)
        return flops




if __name__ == "__main__":

    input_size = 256
    arch = PhasorFormer
    depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    model_restoration = PhasorFormer(win_size=10)
    # print(model_restoration)

    NUM = 20
    model_restoration = model_restoration.cuda()

    from thop import profile
    input_single = torch.randn((1, 3, 320, 320)).cuda()
    xm_single = torch.ones((1, 1, 320, 320)).cuda()
    flops, params = profile(model_restoration, inputs=(input_single, xm_single))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"参数量: {params / 1e6:.2f} M")





