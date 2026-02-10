# Mamba Out - branch 별 channel 다르게
# 12블록 84채널 #param : 860004

# 여기에, 다운사이즈를 Mambaout단계에서 하고, 키만 다운사이즈(원본사이즈 z가 gate)
# coef가 바뀌어도 대응할 수 있게
# channel-mix를 2개 레이어로?
# 또는 branch의 idx끼리만 믹싱 후 결합해서 1개의 channel-mix?

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from basicsr.utils.registry import ARCH_REGISTRY
import math

from timm.models.layers import DropPath

from itertools import repeat
import collections.abc
from typing import Tuple

from pdb import set_trace as st
import numpy as np
from einops import rearrange, repeat

from functools import partial
from pdb import set_trace as st

##################################################
# Layer Norm (입력이 [B,C,H,W]인 것을 전제)
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
        
##############################################################
## ConvFFN
class ConvFFN_v1(nn.Module):
    def __init__(self,
                 dim,
                 ffn_scale=2,
                 p_ratio=0.5,
                 use_norm=False,
                 drop_path = 0.,
                 ):
        super().__init__()

        self.expand_dim = int(dim * ffn_scale)
        self.expand = nn.Conv2d(dim, self.expand_dim, 1, bias=True)    # conv(dim, dim*2, kernel=1)
        self.act = nn.GELU()

        self.p = int(p_ratio * self.expand_dim)  # half : dwconv
        self.dwconv = nn.Conv2d(self.p, self.p, 3, padding=1, groups=self.p, bias=True)
        self.norm = nn.GroupNorm(1, self.p) if use_norm else nn.Identity()

        self.project = nn.Conv2d(self.expand_dim, dim, 1, bias=True)  # conv(dim*2, dim, kernel=1)


    def forward(self, x):
        y = self.act(self.expand(x))

        y1, y2 = torch.split(y, [self.p, self.expand_dim - self.p], dim=1)
        y1 = self.norm(self.dwconv(y1))

        y = torch.cat([y1, y2], dim=1)
        y = self.project(y)

        return y

class ConvFFN_v2(nn.Module):
    f"""
    x - conv33 - query |
      |- conv11 - key --- cat - conv11 - output
    실패작
    """
    def __init__(self,
                 dim,
                 ffn_scale=2,
                 p_ratio=0.5,
                 use_norm=False,
                 drop_path = 0.,
                 ):
        super().__init__()

        self.expand_dim = int(dim * ffn_scale)
        self.act = nn.GELU()
        self.p = int(p_ratio * self.expand_dim)  # conv33/conv11 per half

        self.queryconv = nn.Conv2d(self.p, self.p, 3, 1, 1)
        self.keyconv = nn.Conv2d(self.expand_dim - self.p,
                                 self.expand_dim - self.p, 1, 1, 0)
        self.norm = nn.GroupNorm(1, self.p) if use_norm else nn.Identity()

        self.valueconv = nn.Conv2d(self.expand_dim, dim, 1, bias=True)  # conv(dim*2, dim, kernel=1)


    def forward(self, x):
        query = self.act(self.queryconv(x))
        key = self.act(self.keyconv(x))

        y = torch.cat([query, key], dim=1)
        y = self.valueconv(y)

        return y


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class PCFN(nn.Module):
    f"""
    성공작
    x - conv11(x2) -GeLu & split -- x1 (채널 : dim//2) -- conv33 -|
                                 |- x2 (채널 : dim*(3/4)) ------- cat -- conv11(/2) - output
    채널을 2배로 늘린 뒤, 그 1/4(원래 채널의 1/2)에만 conv@33 후 act
    나머지 3/4와 컨캣 후 conv로 원래 채널 수복
    """
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim,hidden_dim,1,1,0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim ,3,1,1)

        self.act =nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x,[self.p_dim,self.hidden_dim-self.p_dim],dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1,x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x[:,:self.p_dim,:,:] = self.act(self.conv_1(x[:,:self.p_dim,:,:]))
            x = self.conv_2(x)
        return x

class ConvFFN_v3(nn.Module):
    def __init__(self,
                 dim,
                 ffn_scale=2.0,
                 divide_ratio=0.25
                 ):
        super().__init__()
        f"""
        x -- conv2d(x2) & split(by ratio) -- x1 - conv33 --|
                                          |- x2 - dwconv55 - cat - conv11(/2) - output
        """
        self.dim = dim
        self.expand_dim = int(dim*ffn_scale)

        self.deep_conv_dim = int(self.expand_dim * divide_ratio)
        self.shallow_conv_dim = self.expand_dim - self.deep_conv_dim

        self.conv1 = nn.Conv2d(dim, self.expand_dim, 1, 1, 0)
        self.deep_conv = nn.Conv2d(self.deep_conv_dim,
                                   self.deep_conv_dim, 3, 1, 1)
        self.shallow_conv = nn.Conv2d(self.shallow_conv_dim,
                                      self.shallow_conv_dim,
                                      kernel_size=5,
                                      stride=1,
                                      padding=2, groups=self.shallow_conv_dim)
        # self.shallow_conv = nn.Identity()

        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(self.expand_dim, dim, 1, 1, 0)


    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = torch.split(x, [self.deep_conv_dim, self.shallow_conv_dim], dim=1)
        x1 = self.deep_conv(x1)
        x2 = self.shallow_conv(x2)
        x = self.conv2(torch.cat([x1, x2], dim=1))
        return x

class ConvFFN_v4(nn.Module):
    def __init__(self,
                 dim,
                 ffn_scale=2.0,
                 divide_ratio=0.5,
                 shallow_kernel=7
                 ):
        super().__init__()
        f"""
        ConvFFN_v4를 적용할 시, 채널의 수를 크게 줄여야 함.
        dim 84, block 9 - 969K (kernel=5)
        dim 80, block 9 - 899K (kernel=7)
        x -- conv2d(x2) & split(by ratio) -- x1 - conv33 -----------|
                                          |- x2 - dwconv55 & GeLU - * - conv11 - output
        """
        # channel
        self.dim = dim
        self.expand_dim = int(dim*ffn_scale)
        self.deep_conv_dim = int(self.expand_dim * divide_ratio)
        self.shallow_conv_dim = self.expand_dim - self.deep_conv_dim

        # convolution
        self.conv1 = nn.Conv2d(dim, self.expand_dim, 1, 1, 0)
        self.deep_conv = nn.Conv2d(self.deep_conv_dim,
                                   self.deep_conv_dim, 3, 1, 1)
        self.shallow_conv = nn.Conv2d(self.shallow_conv_dim,
                                      self.shallow_conv_dim,
                                      kernel_size=shallow_kernel,
                                      stride=1,
                                      padding=shallow_kernel//2,
                                      groups=self.shallow_conv_dim)
        # self.shallow_conv = nn.Identity()
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = torch.split(x, [self.deep_conv_dim, self.shallow_conv_dim], dim=1)
        x1 = self.deep_conv(x1)
        x2 = self.act(self.shallow_conv(x2))
        x = self.conv2(x1 * x2)
        return x


##############################################################
## Channel-Mixing
class CMX(nn.Module):
    def __init__(self,
                 dim,
                 n_levels,
                 kernel_horizon = 7):
        super().__init__()

        self.dim = dim
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # args: size
        self.n_levels = n_levels

        # convolution
        # 77 -> pad:2 stride:1
        self.conv = nn.Conv2d(self.n_levels,    # 4
                              self.n_levels,    # 4
                              kernel_size=(4, kernel_horizon),
                              padding=(0, kernel_horizon//2),
                              padding_mode='circular',
                              stride=(self.n_levels,1), # stride : (4, 1)
                              groups=self.n_levels, # groups : 4
                              )
        self.act = nn.SiLU()

    def forward(self, x):

        # 4x[B, C//4, H, W] -> stack [B, 4, C//4, H, W] -> pool * squeeze:[B, 4, C//4]
        xs = torch.stack(x, dim=1)
        batch, branch, chan, h, w = xs.shape
        pooled = self.avgpool(xs).squeeze(-1).squeeze(-1)   # [B, 4, C//4]

        # [B, 4, C//4] -> [B, 1, 4, C//4] -> [B, 4, 4, C//4]
        pooled = pooled.unsqueeze(1).repeat(1,4,1,1)

        # -> after conv: [B, 4, 1, C//4] -> squeeze [B, 4, C//4] -> view [B, C] -> unsqueeze(for h,w)
        out = self.conv(pooled).squeeze(2)
        out = out.view(batch, branch * chan).unsqueeze(-1).unsqueeze(-1)
        out = self.act(out)
        out = xs.view(batch, branch * chan, h, w) * out

        return out

##############################################################
## Mamba-Out IR
class BlqMamba(nn.Module):
    def __init__(self,
                 dim,
                 expansion_ratio=8/3,
                 kernel_size=7,
                 conv_ratio=1.0,
                 act_layer=nn.GELU,
                 drop_path=0.,
                 downsize_idx=0,
                 n_levels=4,
                 **kwargs
                 ):

        super().__init__()

        # z, query, key = [dim*expansion_ratio, dim*expansion_ratio - dim*conv_ratio, dim*conv_ratio]
        # total channel : 2*dim*expansion_ratio

        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)

        self.split_indices = (hidden, hidden - conv_channels, conv_channels)

        # dwconv
        self.conv = nn.Conv2d(conv_channels,
                              conv_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size//2,
                              groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # pyrmaid(downsize_ratio = [1/8, 1/4, 1/2, 1])
        self.downsize_idx = downsize_idx
        self.n_levels = n_levels


    def forward(self, x):
        batch, chan, H, W = x.shape

        x = x.permute(0,2,3,1).contiguous() # [B, H, W, C]
        z, query, key = torch.split(self.fc1(x).contiguous(), self.split_indices, dim=-1)

        ##################################
        # 1. Downsize
        ##################################
        key = key.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]

        ratio = self.n_levels - 1 - self.downsize_idx   # 3-2-1-0
        if ratio > 0:
              p_size = (H//2**ratio, W//2**ratio)
              key = F.adaptive_max_pool2d(key, p_size)  # [B, C, h, w]

        ##################################
        # 2. MambaOut and Upsample
        ##################################
        # st()
        key = self.conv(key)
        key = F.interpolate(key, size=(H, W), mode='nearest') # [B, C, H, W]
        key = key.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]

        ##################################
        # 3. Cat and return
        ##################################
        x = self.fc2(self.act(z) * torch.cat((query, key), dim=-1)).contiguous()
        x = self.drop_path(x)

        return x.permute(0,3,1,2)   # [B, H, W, C] -> [B, C, H, W]

##############################################################
## Pyramid Gate Block with Per Branch Conv@11
class GBC_little_conv(nn.Module):
    def __init__(self,
                 dim,
                 ffn_scale=2.0,
                 norm_layer=partial(nn.LayerNorm,eps=1e-6),
                 act_layer=nn.GELU,
                 kernel_size=7,
                 conv_ratio=1.0,
                 drop_path=0.,
                 expansion_ratio=8/3,
                 n_levels=4,
                 coef=[1,2,3,6],) :
        super().__init__()

        # variants ---------------------------------------
        self.n_levels = n_levels
        branch_coef = coef  # channel per branch
        self.coef_sum = sum(coef)  # full coef sum
        self.one_coef = dim // self.coef_sum  # channel per one coef
        self.branch_dim = [coef[3]*self.one_coef, coef[2]*self.one_coef, \
            coef[1]*self.one_coef, coef[0]*self.one_coef]

        # func -------------------------------------------
        self.gbc = nn.ModuleList([
            BlqMamba(self.branch_dim[n_levels-i-1],
                     expansion_ratio=expansion_ratio,
                     kernel_size=kernel_size,
                     conv_ratio=conv_ratio,
                     act_layer=nn.GELU,
                     drop_path=drop_path,
                     downsize_idx = i,
                     n_levels=n_levels,)

            for i in range(self.n_levels)])

        # self.aggr = CMX(dim, n_levels)
        self.aggr_1 = nn.Sequential(
            nn.Conv2d(dim, dim * int(ffn_scale), 1, 1, 0, groups=self.one_coef),
            nn.Conv2d(dim * int(ffn_scale), dim, 1, 1, 0, groups=self.one_coef),
        )
        self.norm1 = LayerNorm(dim)
        self.aggr_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.SiLU()

    def forward(self, x):
        batch, chan, H, W = x.shape

        ##########################################
        # 1. Mamba-Out architecture per feature
        ##########################################
        xc = torch.split(x, self.branch_dim, dim=1)

        for i in range(self.n_levels):
            branch = self.gbc[i](xc[i])
            branch = branch.view(batch, self.coef_sum, -1, H, W).contiguous() # [B,C,H,W] -> [B,coef_sum,-1,H,W]

            if i == 0:
              out = branch
            else :
              out = torch.cat([out, branch], dim=2) # [B, coef_sum, branch_chan_sum, H, W]

        ##########################################
        # 2. Channel-Mixing
        ##########################################
        out = out.view(batch, chan, H, W).contiguous()
        out = self.act(self.norm1(self.aggr_1(out)))

        out = self.aggr_2(out)
        out = self.act(out) * x
        return out
    
    
##############################################################
## Pyramid Gate Block
class GBC(nn.Module):
    def __init__(self,
                 dim,
                 ffn_scale=2.0,
                 norm_layer=partial(nn.LayerNorm,eps=1e-6),
                 act_layer=nn.GELU,
                 kernel_size=7,
                 conv_ratio=1.0,
                 drop_path=0.,
                 expansion_ratio=8/3,
                 n_levels=4,
                 coef=[1,2,3,6],) :
        super().__init__()

        # variants
        self.n_levels = n_levels
        branch_coef = coef  # channel per branch
        coef_sum = sum(coef)  # full coef sum
        one_coef = dim // coef_sum  # channel per one coef
        self.branch_dim = [coef[0]*one_coef, coef[1]*one_coef, \
            coef[2]*one_coef, coef[3]*one_coef]

        # func
        self.gbc = nn.ModuleList([
            BlqMamba(self.branch_dim[i],
                     expansion_ratio=expansion_ratio,
                     kernel_size=kernel_size,
                     conv_ratio=conv_ratio,
                     act_layer=nn.GELU,
                     drop_path=drop_path,
                     downsize_idx = i,
                     n_levels=n_levels,)

            for i in range(self.n_levels)])

        # mixing among branches ---------------------------
        # self.pairwise = nn.ModuleList([
        #     nn.Conv2d(self.branch_dim[i]+self.branch_dim[i+1],
        #               self.branch_dim[i+1],
        #               kernel_size=1,
        #               stride=1,
        #               padding=0,
        #               groups=1)
        # for i in range(self.n_levels - 1)])

        # self.alpha = nn.Parameter(torch.ones(3))

        # -------------------------------------------------

        # self.aggr = CMX(dim, n_levels)
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.SiLU()

    def forward(self, x):
        batch, chan, H, W = x.shape
        out = []

        ##########################################
        # 1. Mamba-Out architecture per feature
        ##########################################
        xc = torch.split(x, self.branch_dim, dim=1)

        for i in range(self.n_levels):
            # st()
            branch = self.gbc[i](xc[i])
            out.append(branch)

        ##########################################
        # 2. Channel-Mixing
        ##########################################
        out = torch.cat(out, dim=1)
        out = self.aggr(out)

        out = self.act(out) * x
        return out
    

##############################################################
## Block
class BasicBlock(nn.Module):
    def __init__(self,
                 dim,
                 ffn_scale=2.0,
                 norm_layer=partial(nn.LayerNorm,eps=1e-6),
                 act_layer=nn.GELU,
                 kernel_size=7,
                 conv_ratio=1.0,
                 drop_path=0.,
                 expansion_ratio=8/3,
                 coef=[1,2,3,6],
                 mode=0,
                 ):

        super().__init__()

        # norm
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        # self.norm1 = norm_layer(dim)
        # self.norm2 = norm_layer(dim)

        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # Multiscale Block

        if mode == 0:
          self.blqm = GBC(dim,
                          expansion_ratio=expansion_ratio,
                          kernel_size=kernel_size,
                          conv_ratio=conv_ratio,
                          act_layer=nn.GELU,
                          drop_path=drop_path,
                          coef=coef,
                          )
        elif mode == 1:
          self.blqm = GBC_little_conv(dim,
                          expansion_ratio=expansion_ratio,
                          kernel_size=kernel_size,
                          conv_ratio=conv_ratio,
                          act_layer=nn.GELU,
                          drop_path=drop_path,
                          coef=coef,
                          )


        # Feedforward layer
        # self.convffn = ConvFFN_v4(dim,
        #                ffn_scale,
        #                )
        self.convffn = PCFN(
            dim,
            growth_rate=ffn_scale,
        )

    def forward(self, x):
        # legacy code ---------------------------
        x = self.blqm(self.norm1(x)) + x
        x = self.convffn(self.norm2(x)) + x

        return x
    
    
##############################################################
## Overall Architecture
@ARCH_REGISTRY.register()
class LMLT(nn.Module):
    def __init__(self,
                 dim,
                 in_chans=3,
                 n_blocks=8,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,
                 conv_ratio=1.0,
                 kernel_size=7,
                 drop_path_rate=0.,
                 upscale=2,
                 ffn_scale=2,
                 expansion_ratio=8/3,
                 coef=[1,2,3,6],
                 mode=0,
                 **kwargs,
    ):
        super().__init__()

        # variants
        self.dim = dim
        self.ffn_scale = ffn_scale

        # stochastic depth
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]

        # Architecture
        self.to_feat = nn.Conv2d(in_chans, dim, 3, 1, 1)

        self.feats = nn.Sequential(*[BasicBlock(dim,
                                                ffn_scale,
                                                norm_layer=norm_layer,
                                                act_layer=act_layer,
                                                kernel_size=kernel_size,
                                                conv_ratio=conv_ratio,
                                                # drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                                                drop_path = dpr[i],
                                                expansion_ratio=expansion_ratio,
                                                coef=coef,
                                                mode=mode,
                                                )
                                     for i in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscale**2, 3, 1, 1),
            nn.PixelShuffle(upscale)
        )


    def check_img_size(self, x):
        _, _, h, w = x.size()
        downsample_scale = 8
        scaled_size = downsample_scale

        mod_pad_h = (scaled_size - h % scaled_size) % scaled_size
        mod_pad_w = (scaled_size - w % scaled_size) % scaled_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


    def forward(self, x):
        B, C, H, W = x.shape

        # check image size
        x = self.check_img_size(x)

        # patch embed
        x = self.to_feat(x)

        # module, and return to original shape
        x = self.feats(x) + x
        x = x[:, :, :H, :W]

        # reconstruction
        x = self.to_img(x)
        return x
    

if __name__== '__main__':
    #############Test Model Complexity #############
    # from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    # x = torch.randn(1, 3, 320, 180)
    # x = torch.randn(1, 3, 256, 256)

    mode = 0  # mode:0 - 일반 실행, 1 - little conv - global conv 실행

    model = LMLT(dim=84,
                    n_blocks=12,
                    upscale=2,
                    expansion_ratio=8/3,
                    coef=[1, 1, 1, 1], # [1, 2, 3, 6]
                    mode = 0)
    # model = LMLT(dim=36, n_blocks=12, ffn_scale=2.0, upscaling_factor=2)
    # print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)