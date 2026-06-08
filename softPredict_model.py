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

'''-------------------------------------------------------------------------------------------------------------------'''



'''-------------------------------------------------------------------------------------------------------------------'''
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
        joint_input_dim = in_chans   # x (3) + xm (1) = 4
        self.initial_proj = nn.Linear(joint_input_dim, embed_dim)
        self.input_proj = InputProj(in_channel=3, out_channel=embed_dim, kernel_size=3, stride=1,
                                  act_layer=nn.LeakyReLU)

        self.output_proj = OutputProj(in_channel=2*embed_dim, out_channel=1, kernel_size=3, stride=1)
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
        self.decodelayer_3=BasicUniDecoderLayer(dim=embed_dim*2,
                                                output_dim=embed_dim*2,
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

    def forward(self, x):
        # Input Projection
        H, W = (x.shape[2], x.shape[3])
        y = self.input_proj(x)
        y = self.pos_drop(y)
        # Encoder encodelayer -> downsample
        conv0 = self.encodelayer_0(y, img_size=(H, W))
        pool0 = self.downsample_0(conv0, img_size=(H, W))

        conv1 = self.encodelayer_1(pool0, img_size=(H // 2, W // 2))
        pool1 = self.downsample_1(conv1, img_size=(H // 2, W // 2))

        conv2 = self.encodelayer_2(pool1, img_size=(H // 4, W // 4))
        pool2 = self.downsample_2(conv2, img_size=(H // 4, W // 4))

        conv3 = self.encodelayer_3(pool2, img_size=(H // 8, W // 8))
        pool3 = self.downsample_3(conv3, img_size=(H // 8, W // 8))

        # Bottleneck
        conv4 = self.conv(pool3, img_size=(H // 16, W // 16))

        # Decoder upsample -> cat -> decodelayer
        up0 = self.upsample_0(conv4, img_size=(H // 16, W // 16))
        deconv0 = torch.cat([up0, conv3], -1)
        deconv0 = self.decodelayer_0(deconv0, img_size=(H // 8, W // 8))

        up1 = self.upsample_1(deconv0, img_size=(H // 8, W // 8))
        deconv1 = torch.cat([up1, conv2], -1)
        deconv1 = self.decodelayer_1(deconv1, img_size=(H // 4, W // 4))

        up2 = self.upsample_2(deconv1, img_size=(H // 4, W // 4))
        deconv2 = torch.cat([up2, conv1], -1)
        deconv2 = self.decodelayer_2(deconv2, img_size=(H // 2, W // 2))

        up3 = self.upsample_3(deconv2, img_size=(H // 2, W // 2))
        deconv3 = torch.cat([up3, conv0], -1)
        deconv3 = self.decodelayer_3(deconv3, img_size=(H, W))

        # Output Projection
        y = self.output_proj(deconv3, img_size=(H, W))
        return y

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
    model_restoration = PhasorFormer(img_size=input_size,
                                     win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff')

    model_restoration = model_restoration.cuda()

    from thop import profile

    input_single = torch.randn((1, 3, 256, 256)).cuda()

    flops, params = profile(model_restoration.cuda(), inputs=(input_single,))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.2f} M")





