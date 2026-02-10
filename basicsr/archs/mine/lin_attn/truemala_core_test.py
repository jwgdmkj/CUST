import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
import math

from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath

from itertools import repeat
import collections.abc
from typing import Tuple

from pdb import set_trace as st
import numpy as np
from einops import rearrange, repeat

from functools import partial
from pdb import set_trace as st
f"""
turemala_core_test.py

mala 자체의 모듈을 여러개 수정해가면서 성능변화 추이 체크
"""
# df2k download : https://github.com/dslisleedh/Download_df2k/blob/main/download_df2k.sh
# dataset prepare : https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md

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

# SE
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


# Channel MLP: Conv1*1 -> Conv1*1
class ChannelMLP(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.mlp(x)


# MBConv: Conv1*1 -> DW Conv3*3 -> [SE] -> Conv1*1
class MBConv(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.mbconv = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim),
            nn.GELU(),
            SqueezeExcitation(hidden_dim),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.mbconv(x)


# CCM
class CCM(nn.Module):
    def __init__(self, 
                 dim, 
                 growth_rate=2.0,
                 drop_path=0.,
                 ):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(), 
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)

class tTensor(torch.Tensor):
    @property
    def shape(self):
        shape = super().shape
        return tuple([int(s) for s in shape])
    
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

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
        x = self.act(self.conv_0(x))
        x1, x2 = torch.split(x,[self.p_dim,self.hidden_dim-self.p_dim],dim=1)
        x1 = self.act(self.conv_1(x1))
        x = self.conv_2(torch.cat([x1,x2], dim=1))

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
## MALA Attention
def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

class RoPE(nn.Module):

    def __init__(self, embed_dim):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    
    def forward(self, slen: Tuple[int]):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        # st()
        # index = torch.arange(slen[0]*slen[1]).to(self.angle)
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)
        # sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
        # sin = sin.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        sin = torch.cat([sin_h, sin_w], -1) #(h w d1)
        # cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
        # cos = cos.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        cos = torch.cat([cos_h, cos_w], -1) #(h w d1)

        retention_rel_pos = (sin.flatten(0, 1), cos.flatten(0, 1))

        return retention_rel_pos
    
def theta_shift(x, sin, cos):
    # st()
    return (x * cos) + (rotate_every_two(x) * sin)

class AddLinearAttention(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # self.num_heads = num_heads
        # self.head_dim = dim // num_heads
        
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = dim ** -0.5
        self.elu = nn.ELU()
        
        # PE 
        self.dwc_q = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.dwc_k = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.dwc_v = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)


    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        '''
        x: (b c h w)
        sin: ((h w) d1)
        cos: ((h w) d1)
        
        x - linear - QKV Linear Attn (+lepe) - * - proj
          |- linear - O(gate) -----------------|
        
        Mamba 형식을 갖추기 위해, 추가로 더해야 하는 것: 
        1) O(gate)에 activation을 추가해도 됨.
        2) QKV을 O와 함께 한번에 만드는게 아닌, Linear-Conv-Act 이후 QKV를 만들어도 됨.
        
        추가 실험 아이디어(DWConv는 gate형식 또는 그냥 DWConv) : 
        1) 두개의 Branch를 둬서, 하나는 MALA, 다른 하나는 DWConv로 구성. 이후에 합하기(또는 곱하기). 
        2) DWConv 이후에 DWConv를 추가로 더 쌓기. 
        3) MALA 그 자체를 ISR에 알맞게 변형.
        
        만일 DWConv 모듈 추가로 인해 파라미터가 많아진다면, PCFN의 ratio를 줄이는 방향으로도 접근 가능.
        '''
        
        ##################################
        # 1. make QKV, O(gate), LEPE
        ##################################
        B, C, H, W = x.shape
        qkvo = self.qkvo(x) #(b 3*c h w)
        qkv = qkvo[:, :3*self.dim, :, :]
        o = qkvo[:, 3*self.dim:, :, :]
        lepe = self.lepe(qkv[:, 2*self.dim:, :, :]) # (b c h w)

        # (b 3c h w) -> (3 b num_head (h w) c//num_head)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = q + self.dwc_q(q)
        k = k + self.dwc_k(k)
        v = v + self.dwc_v(v)
        
        q = q.view(B, C, H*W).permute(0,2,1).contiguous()
        k = k.view(B, C, H*W).permute(0,2,1).contiguous()
        v = v.view(B, C, H*W).permute(0,2,1).contiguous()
        
        ####################################
        # 2. ELU and make z(sclaing factor)
        ####################################
        q = self.elu(q) + 1
        k = self.elu(k) + 1
        z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) * self.scale

        # q = theta_shift(q, sin, cos)
        # k = theta_shift(k, sin, cos)

        ################################
        # 3. kv and output
        ################################
        kv = (k.transpose(-2, -1) * (self.scale / (H*W)) ** 0.5) @ (v * (self.scale / (H*W)) ** 0.5)

        res = q @ kv * (1 + 1/(z + 1e-6)) - z * v.mean(dim=2, keepdim=True)

        # (b (h w) c) -> (b c h w)
        res = rearrange(res, 'b (h w) d -> b d h w', h=H, w=W)
        res = res + lepe
        return self.proj(res * o)

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
                 branch_dim=[1,2,3,6],) :
        super().__init__()
        
        # variants
        self.n_levels = n_levels
        self.branch_dim = branch_dim

        # func
        self.gbc = nn.ModuleList([
            AddLinearAttention(self.branch_dim[i])
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
        # self.aggr = CMX(dim, n_levels)
        # -------------------------------------------------
        
        
        self.gate = nn.Linear(dim, sum(self.branch_dim)- self.branch_dim[n_levels-1])
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.SiLU()
        
    def forward(self, x, sin, cos):
        batch, chan, H, W = x.shape
        out = []
        
        ##########################################
        # 1. Channel-wise Pyramid Downsizing
        ##########################################
        xc = torch.split(x, self.branch_dim, dim=1)
        downsized_feats = []
        
        for i in range(self.n_levels):
            dw_ratio = self.n_levels - 1 - i   # 3-2-1-0
            if dw_ratio > 0:
                p_size = (H//2**dw_ratio, W//2**dw_ratio)
                downsized_feat = F.adaptive_max_pool2d(xc[i], p_size)
                downsized_feats.append(downsized_feat)
            else :
                downsized_feats.append(xc[i])

        ##########################################
        # 2. MALA architecture per feature
        ##########################################
        value = x.permute(0,2,3,1) # [B H W C]
        value = self.gate(value).permute(0,3,1,2).contiguous()
        gate_idx = 0
        
        for i in range(self.n_levels):
            branch = self.gbc[i](downsized_feats[i], sin[i], cos[i])      
            branch_original_shape = F.interpolate(branch, size=(H, W), mode='nearest')
            
            if i < self.n_levels - 1 :  # except last branch
                branch_original_shape = branch_original_shape \
                    * value[:, gate_idx:gate_idx+self.branch_dim[i]]
                # * self.act(value[:, gate_idx:gate_idx+self.branch_dim[i]])
                
                gate_idx = self.branch_dim[i]
                
            out.append(branch_original_shape)        

        ##########################################
        # 3. Channel-Mixing
        ##########################################
        out = torch.cat(out, dim=1)
        out = self.aggr(out)
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
                 branch_dim=[1,2,3,6],
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
        self.blqm = GBC(dim, 
                        expansion_ratio=expansion_ratio,
                        kernel_size=kernel_size, 
                        conv_ratio=conv_ratio,
                        act_layer=nn.GELU,
                        drop_path=drop_path,
                        branch_dim=branch_dim,
                        ) 
        
                                                
        # Feedforward layer
        # self.convffn = ConvFFN_v4(dim, 
        #                ffn_scale,
        #                ) 
        self.convffn = PCFN(
            dim
        )

    def forward(self, x, sin, cos):
        # legacy code ---------------------------      
        x = self.blqm(self.norm1(x), sin, cos) + x
        x = self.convffn(self.norm2(x)) + x
        
        return x
        

##############################################################
## Overall Architecture
# @ARCH_REGISTRY.register()
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
                 branch_dim=[1,2,3,6],
                 n_levels=4,
                 **kwargs,
    ):   
        super().__init__()
        
        # variants
        self.dim = dim
        self.ffn_scale = ffn_scale
        self.n_levels = n_levels
        self.branch_dim = branch_dim
        
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
                                                branch_dim=branch_dim,
                                                ) 
                                     for i in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscale**2, 3, 1, 1),
            nn.PixelShuffle(upscale)
        )
        
        # branch_dim and RoPE
        self.RoPEs = nn.Sequential(*[RoPE(self.branch_dim[i]) 
                                     for i in range(self.n_levels)])
        
        
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
        _, _, H_pad, W_pad = x.shape
        
        #########################################################################
        # sin, cos per branch
        size_list = []
        for i in range(self.n_levels):
            dw_ratio = self.n_levels - 1 - i   # 3-2-1-0
            p_size = (H_pad//2**dw_ratio, W_pad//2**dw_ratio)
            size_list.append(p_size)
        
        # st()
        sin, cos = [], []
        # st()
        for i in range(self.n_levels):
            ssin, ccos = self.RoPEs[i]((size_list[i][0], size_list[i][1])) 
            sin.append(ssin)
            cos.append(ccos)
        #########################################################################
        
        # patch embed
        x = self.to_feat(x)
        
        shortcut = x
        # module, and return to original shape
        for block in self.feats:
            x = block(x, sin, cos)
        x = x + shortcut

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

    model = LMLT(dim=84, 
                    n_blocks=12, 
                    upscale=2,
                    expansion_ratio=8/3,
                    branch_dim=[8,12,16,48],
                    kernel_size=9,)
    # model = LMLT(dim=36, n_blocks=12, ffn_scale=2.0, upscaling_factor=2)
    # print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)