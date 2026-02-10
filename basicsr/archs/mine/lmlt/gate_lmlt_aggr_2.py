import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from basicsr.utils.registry import ARCH_REGISTRY

from itertools import repeat
import collections.abc
from typing import Tuple

from pdb import set_trace as st
import numpy as np

f"""
gate_lmlt_aggr.py

aggregation 방법 실험실
1) conv@11 - dwconv@33 - (+x)

# 10000-30.93 / 25000-31.47 / 50000-31.75 / 100K-31.95 (222,224)
"""
# df2k download : https://github.com/dslisleedh/Download_df2k/blob/main/download_df2k.sh
# dataset prepare : https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md

# Layer Norm
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
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(), 
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

## PCFN
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
    
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class ConvPosEnc(nn.Module):
    """Depth-wise convolution to get the positional information.
    """
    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              to_2tuple(k),
                              to_2tuple(1),
                              to_2tuple(k // 2),
                              groups=dim)

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        feat = feat.flatten(2).transpose(1, 2)
        x = x + feat
        return x
    

##############################################################
## Downsample ViT
class downsample_vit(nn.Module):
    def __init__(self, 
                 dim, 
                 window_size=8, 
                 attn_drop=0., 
                 proj_drop=0.,
                 down_scale=2,
                 use_gate_like=True,):
        super().__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.scale = dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim*3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)
        
        # mamba-like gating
        self.use_gate_like = use_gate_like
        if self.use_gate_like:
            self.z = nn.Linear(dim, dim)
            self.act = nn.GELU()        
    
    
    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    
    def window_reverse(self, windows, window_size, h, w):
        """
        Args:
            windows: (num_windows*b, window_size, window_size, c)
            window_size (int): Window size
            h (int): Height of image
            w (int): Width of image

        Returns:
            x: (b, h, w, c)
        """
        b = int(windows.shape[0] / (h * w / window_size / window_size))
        x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
        return x
    
    
    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.window_size, self.window_size
        x = x.view(B, C, H//H_sp, H_sp, W//W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        lepe = func(x) ### B', C, H', W'
        lepe = lepe.reshape(-1, C, H_sp * W_sp).permute(0, 2, 1).contiguous()

        x = x.reshape(-1, C, H_sp* W_sp).permute(0, 2, 1).contiguous()
        return x, lepe
    
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        ################################
        # window partition
        ################################
        x = x.permute(0, 2, 3, 1)   # b h w c
        if self.use_gate_like:  
            z = self.act(self.z(x))
            
        x_window = self.window_partition(x, self.window_size).permute(0,3,1,2)
        x_window = x_window.permute(0,2,3,1).view(-1, self.window_size * self.window_size, C)
        
        ################################
        # make qkv
        ################################
        qkv = self.qkv(x_window)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        ################################
        # attn and PE
        ################################
        v, lepe = self.get_lepe(v, self.get_v)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v) + lepe
        
        ################################
        # proj, drop and reshape
        ################################
        x = x.reshape(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(x, self.window_size, H, W)
        
        if self.use_gate_like:
            x = x * z
            
        x = self.proj(x)
        x = self.proj_drop(x)
               
        return x.permute(0,3,1,2)   # b c h w


##############################################################
## LHSBv2 - split dim and define 4 attn blocks
class MultiAggregation(nn.Module):
    def __init__(self, 
                 dim,
                 groups=2,
                 branch_dim=[6,8,10,12]):
        super().__init__()
        
        # channel shuffle
        self.groups = groups
        heads = []
        tails = []
        cur_idx = 0
        for c in branch_dim:
            if c % groups != 0:
                raise ValueError(f"Channel {c} must be even for half-splitting.")
            
            branch_half = c // 2
            heads.extend(range(cur_idx, cur_idx + branch_half))    # 전반부 - cur ~ cur+half
            tails.extend(range(cur_idx+branch_half, cur_idx+c))    # 후반부 - cur+half ~ cur+c
            
            cur_idx = cur_idx + c
        final_indices = heads + tails
        self.register_buffer('shuffle_idx', torch.tensor(final_indices, dtype=torch.long))
        
        # channel mixing
        self.proj_in = nn.Conv2d(dim, dim*2, 1, 1, 0, groups=groups)
        
        # gating
        self.dwc1 = nn.Conv2d(dim, dim, (1,5), 1, (0,2), groups=dim)    # horizon
        self.dwc2 = nn.Conv2d(dim, dim, (5,1), 1, (2,0), groups=dim)    # vertical
        
        # self.act = nn.GELU()
        
        # final projection
        self.proj_out = nn.Conv2d(dim*2, dim, 1, 1, 0)
    
    def channel_shuffle(self, x):
        ##  shuffleNet
        # b, c, h, w = x.shape
        # x = x.reshape(b, self.groups, -1, h, w)
        # x = x.permute(0,2,1,3,4)
        # x = x.reshape(b, -1, h, w)
        
        ## half split
        x = x[:, self.shuffle_idx]
        return x
    
    def forward(self, x):
        
        # shuffle and proj_in
        x_shuffle = self.channel_shuffle(x)
        x_proj = self.proj_in(x_shuffle)
        
        # split and gating
        x_hor, x_ver = x_proj.chunk(2, dim=1)
        x_hor = self.dwc1(x_hor) + x_hor
        x_ver = self.dwc2(x_ver) + x_ver
        out = torch.cat([x_hor, x_ver], dim=1)
        
        # fusion
        out = self.proj_out(out)
        return out

class LargeKernelAggregation(nn.Module):
    def __init__(self, dim, expansion_ratio=1.5): # 1.5배 확장으로 파라미터 다이어트
        super().__init__()
        
        hidden_dim = int(dim * expansion_ratio)
        
        # [1] Expansion (dim -> 1.5*dim)
        # 2배 대신 1.5배만 늘려서 파라미터 폭증을 막음
        self.proj_in = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        
        # [2] Large Kernel Depthwise (7x7)
        # 커널 크기는 그대로 유지하여 성능(Receptive Field) 확보
        self.dwc = nn.Conv2d(hidden_dim, hidden_dim, 7, 1, 3, groups=hidden_dim)
        
        # [3] Gating 준비
        # hidden_dim을 반으로 쪼개야 하므로 짝수인지 확인 필요 없게 chunk 사용
        # proj_out의 입력 채널은 hidden_dim // 2 가 됨
        self.out_dim = hidden_dim // 2
        
        # [4] Final Projection
        self.proj_out = nn.Conv2d(self.out_dim, dim, 1, 1, 0)

    def forward(self, x):
        # 1. Expansion
        x_expanded = self.proj_in(x)
        
        # 2. Spatial Context (7x7)
        x_spatial = self.dwc(x_expanded)
        
        # 3. Simple Gating (Ratio 1.5)
        # 1.5배 된 채널을 반으로 나누므로, 각각 0.75배의 정보를 가짐
        # Baseline(0.5배 Bottleneck)보다 정보량이 많아 유리함
        x1, x2 = x_spatial.chunk(2, dim=1)
        x_gated = x1 * x2 
        
        # 4. Projection
        out = self.proj_out(x_gated)
        
        return out + x
    
class LHSBv2(nn.Module):
    def __init__(self,
                 dim,  
                 attn_drop=0.,
                 proj_drop=0.,
                 n_levels=4,
                 branch_dim=[8,12,16,48],
                 high_skip=False,
                 use_aggr_expand=True,
                 idx = 0,):
        
        super().__init__()
        self.n_levels = n_levels
        self.branch_dim = branch_dim
        
        self.high_skip = False   # high-freq skip connection
        self.use_aggr_expand = use_aggr_expand
        self.idx = idx

        # Spatial Weighting
        self.mfr = nn.ModuleList([
            downsample_vit(self.branch_dim[i],
                           window_size=8,
                           attn_drop=attn_drop,
                           proj_drop=proj_drop,
                           down_scale=2**i)
            for i in range(self.n_levels)])
        
        # Feature Aggregation
        if not self.use_aggr_expand:
            self.aggr = nn.Sequential(
                nn.Conv2d(dim, dim, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                # nn.GELU(),
                # nn.Conv2d(dim//2, dim, 1, 1, 0)
            )
        else :
            self.aggr = LargeKernelAggregation(dim)
        
        # Activation
        self.act = nn.GELU() 

        # Gating
        self.gate = nn.Linear(dim, dim)
        
        
    def forward(self, x, aggr=None):
        batch, chan, H, W = x.shape

        xc = torch.split(x, self.branch_dim, dim=1)
        SA_before_idx = None
        out = []
        
        ####################################
        ## gating
        ####################################
        gate = x.permute(0,2,3,1) # [B H W C]
        gate = self.gate(gate).permute(0,3,1,2).contiguous()
        gate_idx = 0 

        ####################################
        ## downsize features
        ####################################
        downsized_feats = []
        
        for i in range(self.n_levels):
            dw_ratio = self.n_levels - 1 - i   # 3-2-1-0
            if dw_ratio > 0:
                p_size = (H//2**dw_ratio, W//2**dw_ratio)
                downsized_feat = F.adaptive_max_pool2d(xc[i], p_size)
                downsized_feats.append(downsized_feat)
            else :
                downsized_feats.append(xc[i])
                
        ####################################
        ## multi-scale processing 
        ####################################        
        for i in range(self.n_levels):
            if i < self.n_levels - 1 :  # except last 
                branch = self.mfr[i](downsized_feats[i])      
                branch_original_shape = F.interpolate(branch, size=(H, W), mode='nearest')

                # branch에, only_high_freq 정보 추가
                if self.high_skip == True:
                    res = xc[i] - F.interpolate(downsized_feats[i], size=(H,W), mode='nearest')
                    branch_original_shape = res + branch_original_shape
                
                branch_original_shape = branch_original_shape \
                    * gate[:, gate_idx:gate_idx+self.branch_dim[i]]
                # * self.act(gate[:, gate_idx:gate_idx+self.branch_dim[i]])
                
                out.append(branch_original_shape)
                gate_idx = self.branch_dim[i]
            
            else :   # last branch : High-Freq Conv
                branch = self.mfr[i](downsized_feats[i])   
                branch = branch * gate[:, gate_idx:gate_idx+self.branch_dim[i]]   
                out.append(branch)        

        ####################################
        ## aggregation
        ####################################
        out = torch.cat(out, dim=1)
        out = self.aggr(out)
        out = self.act(out) * x
        
        return out
        
##############################################################
## Block
class AttBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 ffn_scale=2.0, 
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 branch_dim=[8,12,16,42],):
        
        super().__init__()

        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim) 

        # Multiscale Block
        self.lhsb = LHSBv2(dim, 
                         attn_drop=attn_drop, 
                         proj_drop=drop,
                         branch_dim=branch_dim,) 
        
        # Feedforward layer
        self.pcfn = PCFN(dim, growth_rate=ffn_scale) 

    def forward(self, x):
        x = self.lhsb(self.norm1(x)) + x
        x = self.pcfn(self.norm2(x)) + x
        return x
        

##############################################################
## Overall Architecture
# @ARCH_REGISTRY.register()
class LMLT(nn.Module):
    def __init__(self, 
                 dim, 
                 n_blocks=8, 
                 ffn_scale=2.0, 
                 upscaling_factor=4,
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 branch_dim=[8,12,16,42],):
        
        super().__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.window_size=8
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]  # stochastic depth decay rule
        
        self.feats = nn.Sequential(*[AttBlock(dim, 
                                              ffn_scale, 
                                              drop=drop_rate,
                                              attn_drop=attn_drop_rate,
                                              drop_path=dpr[i],
                                              branch_dim=branch_dim
                                              ) 
                                     for i in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )
        

    def check_img_size(self, x):
        _, _, h, w = x.size()
        downsample_scale = 8
        scaled_size = self.window_size * downsample_scale
        
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
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis    
    x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    # x = torch.randn(1, 3, 320, 180)
    # x = torch.randn(1, 3, 256, 256)

    # large
    # branch_dim=[8,12,16,44]
    # n_blocks = 12
    
    # tiny
    branch_dim = [6, 8, 10, 28]
    n_blocks = 6
    
    dim = sum(branch_dim)
    
    model = LMLT(dim=dim, 
                 n_blocks=n_blocks, 
                 ffn_scale=2.0, 
                 upscaling_factor=2,
                 branch_dim = branch_dim,)
    # model = LMLT(dim=36, n_blocks=12, ffn_scale=2.0, upscaling_factor=2)
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)
