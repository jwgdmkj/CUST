import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange

from itertools import repeat
import collections.abc
from typing import Tuple

from pdb import set_trace as st
import numpy as np

f"""
downsize.py

(ln - globalattn - ln - lightweight mlp) - (ln - neighborattn - ln - mlp)
"""
# df2k download : https://github.com/dslisleedh/Download_df2k/blob/main/download_df2k.sh
# dataset prepare : https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md

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


# LightweightConvFFN
class GatedFFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, split_rate=0.5):
        super().__init__()
        
        ### var
        self.dim = dim
        hidden_dim = int(dim * growth_rate)
        dwc_dim = int(hidden_dim * split_rate)
        self.split_dim = [dwc_dim, hidden_dim - dwc_dim]
    
        ### algorithm
        self.in_proj = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.dwc = nn.Conv2d(dwc_dim, dwc_dim, 5, stride=1, padding=2, groups=dwc_dim)
        self.out_proj = nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.in_proj(x) # b c h w
        x1, x2 = torch.split(x, self.split_dim, dim=1)
        x1 = self.dwc(x1)
        x2 = self.act(x2)
        x = self.out_proj(torch.cat([x1, x2], dim=1))
        return x


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
    x - conv11(x2) -GeLu & split -- x1 (채널 : h_dim/4) -- conv33 -|
                                 |- x2 (채널 : h_dim*(3/4)) ----- cat -- conv11(/2) - output
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

    
##################################################################
## Attn Share Func
def window_partition(x, window_size, batch_flag=False):
    """
    Returns :
        x : (b*nw, wh, wd, c)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    
def window_reverse(windows, window_size, h, w):
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

##############################################################
## NeighborAttn
class NeighborAttn(nn.Module):
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
            
        x_window = window_partition(x, self.window_size).permute(0,3,1,2)
        x_window = x_window.permute(0,2,3,1).view(-1, self.window_size * self.window_size, C)
        
        ################################
        # make qkv and attn, PE
        ################################
        qkv = self.qkv(x_window)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        v, lepe = self.get_lepe(v, self.get_v)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v) + lepe
        
        ################################
        # proj, drop and reshape
        ################################
        x = x.reshape(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W)
        
        if self.use_gate_like:
            x = x * z
            
        x = self.proj(x)
        x = self.proj_drop(x)
               
        return x.permute(0,3,1,2)   # b c h w

##############################################################
## lth - new low-to-high connection
class Low_to_high(nn.Module):
    def __init__(self,
                 lr_dim,
                 hr_dim,):
        
        super().__init__()
        self.down_conv = nn.Conv2d(hr_dim, lr_dim, 1)
        self.up_conv = nn.Conv2d(lr_dim, hr_dim, 1)
        self.scale = nn.Parameter(torch.zeros(1, hr_dim, 1, 1))
    
    def forward(self, lr_post, hr_pre):
        # 1. HR을 LR 공간으로 투영 (Downsample)
        hr_down = F.interpolate(hr_pre, size=lr_post.shape[2:], mode='bilinear')
        hr_down = self.down_conv(hr_down)
        
        # 2. 투영 오차(Residual) 계산
        # "LR이 갖고 있는 정보 중 HR에 반영 안 된 것"
        error = lr_post - hr_down 
        
        # 3. 오차를 다시 HR 공간으로 투영 (Back-Project)
        error_up = self.up_conv(error)
        error_up = F.interpolate(error_up, size=hr_pre.shape[2:], mode='bilinear')
        
        # 4. 보정
        return hr_pre + (self.scale * error_up)
    
##############################################################
## Aggregation among 4 branches
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
        self.dwc1 = nn.Conv2d(dim, dim, (1,5), 1, (0,2), groups=dim)
        self.dwc2 = nn.Conv2d(dim, dim, (5,1), 1, (2,0), groups=dim)
        
        self.act = nn.GELU()
        
        # final projection
        self.proj_out = nn.Conv2d(dim, dim, 1, 1, 0)
    
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
        x1, x2 = x_proj.chunk(2, dim=1)
        x1 = self.dwc1(x1) + x1
        x2 = self.dwc2(x2) + x2
        out = x1 * x2
        
        # fusion
        out = self.proj_out(out)
        return out

##############################################################
## Downsize
class MeanAttn(nn.Module):
    def __init__(self,
                 branch_dim,
                 window_size,
                 topk=3,
                 mean_size=64,) :
        super().__init__()
        
        ### var
        self.dim = sum(branch_dim)
        self.window_size = window_size
        self.topk = topk
        self.num_means = mean_size
        self.branch_dim = branch_dim
        self.mean_iter = 4
        
        ### ema 
        self.register_buffer('running_means', torch.randn(mean_size, self.dim))
        self.momentum = 0.99
        
        ### linear
        self.scale = [self.branch_dim[i] ** -0.5 for i in range(len(self.branch_dim))]
        self.q_linear = nn.ModuleList([
            nn.Linear(self.branch_dim[i], self.branch_dim[i])
            for i in range(len(self.branch_dim))
        ])  # q
        self.k_linear = nn.ModuleList([
            nn.Linear(self.branch_dim[i], self.branch_dim[i])
            for i in range(len(self.branch_dim))
        ])  # k
        self.v_linear = nn.ModuleList([
            nn.Linear(self.branch_dim[i], self.branch_dim[i])
            for i in range(len(self.branch_dim))
        ])  # v
        
        self.proj = nn.ModuleList([
            nn.Linear(self.branch_dim[i], self.branch_dim[i])
            for i in range(len(self.branch_dim))
        ])  # out_proj
        
        self.get_v = nn.ModuleList([
            nn.Conv2d(self.branch_dim[i], self.branch_dim[i], kernel_size=3, stride=1, padding=1,
                      groups=self.branch_dim[i])
            for i in range(len(self.branch_dim))
        ])  # lepe
        
        self.norms = nn.ModuleList([
            LayerNorm(self.branch_dim[i])
            for i in range(len(self.branch_dim))
        ])  # LN
        
        self.mlp = nn.ModuleList([
            GatedFFN(self.branch_dim[i])
            for i in range(len(self.branch_dim))
        ])  # MLP
        
        ### mamba-like gating
        use_gate_like = False
        self.use_gate_like = use_gate_like
        if self.use_gate_like:
            self.z = nn.Linear(dim, dim)
            self.act = nn.GELU()  
    
    
    def get_lepe(self, v, func):
        T, C = v.shape
        H = W = int(np.sqrt(T))
        v = v.transpose(-2,-1).contiguous().view(C, H, W)

        H_sp, W_sp = self.window_size, self.window_size
        v = v.view(C, H//H_sp, H_sp, W//W_sp, W_sp)
        v = v.permute(1, 3, 0, 2, 4).contiguous().reshape(-1, C, H_sp, W_sp) ### nW, C, H', W'

        lepe = func(v) ### nW, C, H', W'
        lepe = lepe.reshape(-1, C, H_sp * W_sp).permute(0, 2, 1).contiguous().squeeze(0) ## nW HW C

        v = v.reshape(C, H_sp* W_sp).permute(1, 0).contiguous()
        return v, lepe
        
        
    def attention(self, idx, dsx, means):
        """
        idx : 몇 번째인지
        dsx : downsize x. [B branch_dim H//2**i W//2**i]
        means : 평균. [T branch_dim]
        """
        
        batch, C, dh, dw = dsx.shape
        T, _ = means.shape
        
        ###################
        # window partition
        ###################
        dsx = dsx.permute(0,2,3,1).contiguous()
        x_window = window_partition(dsx, self.window_size).permute(0,3,1,2)
        x_window = x_window.permute(0,2,3,1).view(-1, self.window_size * self.window_size, C)

        ########################
        # make qkv and attn, pe
        ########################
        q = self.q_linear[idx](x_window)
        k = self.k_linear[idx](means)
        v = self.v_linear[idx](means)
        
        v, lepe = self.get_lepe(v, self.get_v[idx])
        
        attn = (q @ k.transpose(-2, -1)) * self.scale[idx]
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        
        dsx = (attn @ v) + lepe
        
        ################################
        # proj, drop and reshape
        ################################
        dsx = dsx.reshape(-1, self.window_size, self.window_size, C)
        dsx = window_reverse(dsx, self.window_size, dh, dw)
        
        dsx = self.proj[idx](dsx)
        # x = self.proj_drop(x)
               
        return dsx.permute(0,3,1,2)   # b c h w

         
    def mean_maker(self, x, means):
        ###########################################
        # calculate similarity
        # x : [B HW C]
        # means : [T C]
        # sim_map : 가장 유사한 클러스트와의 유사도
        # sim_idx : 가장 유사한 클러스터의 idx
        ###########################################
        batch, N, chan = x.shape
        x, means = F.normalize(x, dim=-1), F.normalize(means, dim=-1)

        sim = (x @ means.transpose(-2, -1)) / (chan ** (-0.5)) # [B HW T] (T = Token or Cluster)
        sim = F.softmax(sim, dim=-1)
        # sim_map, sim_idx = torch.max(means, dim=-1)  
        topk_probs, topk_indices = torch.topk(sim, k=self.topk, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        sparse_probs = torch.zeros_like(sim)
        sparse_probs.scatter_(dim=-1, index=topk_indices, src=topk_probs)

        ########################################
        # renew means using global aggregation
        # sparse prob : [B N T](topk 외엔 0)
        # x : [B N C]
        # sum(dim=(0,1))을 통해 가중치의 총합으로 나눔 -> 정확한 대표값 획득
        ########################################
        sim = torch.einsum('bnt, bnc -> tc', sparse_probs, x)
        sim_denom = sparse_probs.sum(dim=(0,1)).unsqueeze(-1)
        means = sim / (sim_denom + 1e-6)
        
        return means
    
    def mean_maker_optimized(self, x, means):
        batch, N, chan = x.shape    # [B HW C]
        T, _ = means.shape  # [T C]
        x, means = F.normalize(x, dim=-1), F.normalize(means, dim=-1)
        prev_means = means
        
        ### calculate sim
        sim = (x @ means.transpose(-2, -1)) / (chan ** -0.5) # [B, N, T]
        sim = F.softmax(sim, dim=-1)
        topk_probs, topk_indices = torch.topk(sim, k=self.topk, dim=-1) # [B, N, K]
        
        ### distribute x depend on prob(*연산). (dim=k)로 sum 시, 원래의 x가 나옴.
        weighted_x = x.unsqueeze(2).expand(-1, -1, self.topk, -1)   # [B N C] -> [B N topk C]
        weighted_x = weighted_x * topk_probs.unsqueeze(-1)  # 가중치 적용 = [B N topk C] * [B N topk 1]
        weighted_x = weighted_x.reshape(-1, chan)   # [BNk C]
        topk_indices = topk_indices.reshape(-1) # [BNk]
        
        ### renew means
        new_means = torch.zeros_like(means)
        new_means.index_add_(0, topk_indices, weighted_x)
        denom = means.new_zeros(T, 1)
        denom.index_add_(0, topk_indices, topk_probs.reshape(-1).unsqueeze(-1))
        means = new_means / (denom + 1e-6)
        
        ### check whether 0 or not
        is_updated = denom > 1e-6
        means = torch.where(is_updated, means, prev_means)
        
        ### norm
        means = F.normalize(means, dim=-1)
         
        return means
    
    def forward(self, x, dsx, means):
        f"""
        x : [B C H W]
        dsx : [[B C H//2**i W//2**i] x branch 개수]
        mean : 64개의 고정된 앵커를 기반으로, means 생성. [T, C]
        """
        batch, chan, H, W = x.shape
        
        #########################
        # renew and update means
        #########################
        if self.training:
        # if True:
            with torch.no_grad():
                for i in range(self.mean_iter):
                    means = self.mean_maker_optimized(x.permute(0,2,3,1).reshape(batch, -1, chan).contiguous(), means)
        
        if not hasattr(self, 'running_means'):
            self.register_buffer('running_means', means.detach().clone())
        
        if self.training:
            self.running_means = 0.99 * self.running_means + 0.01 * means.detach()
            means = self.running_means
        else:
            means = self.running_means
        
        means_list = torch.split(means, self.branch_dim, dim=-1)
        
        ###############
        # global attn
        ###############
        out = []
        for i in range(len(self.branch_dim)):
            branch = self.attention(i, dsx[i], means_list[i])
            out.append(branch)
        
        ###############
        # norm, MLP
        ###############
        for i in range(len(self.branch_dim)):
            out[i] = self.norms[i](out[i])  # b c' h' w'
            out[i] = self.mlp[i](out[i])
        
        return out, means

##############################################################
## Inter-Window Attn
class InterWindowAttn(nn.Module):
    def __init__(self,
                 dim, 
                 group_size=8, 
                 cluster_size=64):
        super().__init__()
        f"""
        group_size : 총 몇 개의 픽셀을 하나의 group으로 만들 것인가?
        cluster_size : 하나의 클러스터에 몇 개의 group를 넣을 것인가?
        """
        
        ### var
        self.dim = dim
        self.group_size = group_size
        self.cluster_size = cluster_size
        self.batch_consist_flag = True   # 배치 내 다른 이미지도 유사도 계산에 포함 여부

        self.topk = 3
        qkv_dim = dim
        
        ### algorithm
        self.toq = nn.Linear(dim, qkv_dim)
        self.tok = nn.Linear(dim, qkv_dim)
        self.tov = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.lepe_conv = nn.Conv2d(dim, dim, 
                                   kernel_size=3, 
                                   padding=1, 
                                   groups=dim)
    
    
    def get_lepe(self, x_windows):
        """
        x_windows: [Total_N, P, C] (Q의 원천)
        1) [BN, C, ws, ws]로 변경
        2) Conv2d 실행
        3) 다시 [BN, ws*ws, C]로 변경
        """
        x_img = rearrange(x_windows, 'n (h w) c -> n c h w', h=self.group_size, w=self.group_size)
        lepe = self.lepe_conv(x_img)
        lepe_flat = rearrange(lepe, 'n c h w -> n (h w) c')
        return lepe_flat
    
    def batch_consist_sim_calculate(self, dsx):
        """
        dsx: [itr, num_win, chan] 
         - itr: 클러스터(블록) 개수
         - cluster_size: 하나의 클러스터 내 윈도우 개수 (사용자님의 cluster_num)
         - chan: 채널
        
        topk: 상위 몇 개를 뽑을 것인지
        """
        itr, cluster_size, chan = dsx.shape
        topk = self.topk if self.cluster_size >= self.topk else self.cluster_size # 클러스터
        
        # 최종 결과를 담을 리스트 (각 쿼리 클러스터별 결과)
        final_global_indices = []
        final_global_scores = []
        
        #### Outer Loop : query cluster
        for i in range(itr):  
            candidate_score = []
            candidate_index = []
            
            ### Ineer Loop : key cluster            
            for j in range(itr):
                attn = dsx[i] @ dsx[j].transpose(-2, -1)
                sim_score, sim_idx = torch.topk(attn, k=topk, dim=-1)
                global_idx = sim_idx + (j * cluster_size)
                candidate_score.append(sim_score)
                candidate_index.append(global_idx)

            ### Final selection for Cluster i
            ### inner loop을 돌며 모인 후보를 합침
            ### 이후에, 최종 후보들 중 topk를 찾음
            cat_score = torch.cat(candidate_score, dim=1)
            cat_index = torch.cat(candidate_index, dim=1)
            final_val, final_idx = torch.topk(cat_score, k=topk, dim=-1)
            final_idx = torch.gather(cat_index, 1, final_idx)
            
            ### 현재 i에 대한 최종 topk 선별
            final_global_indices.append(final_idx)
            final_global_scores.append(final_val)
            
        
        ### 모든 그룹(batch*h*w)에 대한 유사도, 그 인덱스 결과를 한데 모음
        out_score = torch.cat(final_global_scores, dim=0)
        out_index = torch.cat(final_global_indices, dim=0)            
               
        return out_score, out_index
    
    def batch_non_consist(self, dsx):
        
        return dsx
    
    def forward(self, x):
        """
        x: [B, C, H, W] (이미지 전체)
        hw : 풀링된 h, w(각 값 하나하나가 원래는 64 윈도우인 하나의 그룹)
        """
        batch, chan, H, W = x.shape
        
        #####################
        # 풀링
        #####################
        gh, gw = H//self.group_size, W//self.group_size
        ds_size = (gh, gw)
        dsx = F.adaptive_max_pool2d(x, ds_size)  # [B C h w]
        
        ##############################
        # global similarity search
        ##############################
        # 배치도 비교그룹군 내에 포함 시
        if self.batch_consist_flag :
            cluster_num = (batch*gh*gw) // self.cluster_size
            dsx = dsx.permute(0,2,3,1).reshape(cluster_num, self.cluster_size, chan)    # [num_cls, cls_size, c]

            sim_score, sim_index = self.batch_consist_sim_calculate(dsx)
            x_window = window_partition(x.permute(0,2,3,1), self.group_size, batch_flag=True).flatten(1,2)
            
            ### aggregate - 유사도 비율대로 합치기
            sim_score = sim_score.softmax(dim=-1).unsqueeze(-1).unsqueeze(-1)   # 합 1로, [B*nw, topk, 1, 1]
            
            f"""
            1. k번째 이웃 데이터의 인덱스 [Total_N] 가져오기
            2. k번쨰 이웃 데이터 가져오기 [Total_N, P, C]
            3. k번째 가중치 가져오기, [Total_N, 1, 1]
            4. 누적(by in-place)
            """
            # neighbors = x_window[sim_index.long()]
            # context_windows = torch.sum(neighbors * sim_score, dim=1)
            weights = sim_score
            context_windows = torch.zeros_like(x_window)
            for k in range(self.topk):
                k_idx = sim_index[:, k].long()
                curr_neighbor = x_window[k_idx]
                curr_weight = weights[:, k, :, :] 
                context_windows.add_(curr_neighbor * curr_weight)
        
        # 배치는 포함 안할 시
        else :
            dsx = x
        
        ##########################
        # Attn and Aggregation
        ##########################
        q = self.toq(x_window)
        k, v = self.tok(context_windows), self.tov(context_windows)
        lepe_bias = self.get_lepe(x_window) # [Total_N, P, C]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v) + lepe_bias # [B, N, H, P, D]
        out = self.proj(out)
        out = rearrange(out, '(b h w) (w1 w2) c -> b c (h w1) (w w2)', 
                            b=batch, h=gh, w=gw, w1=self.group_size, w2=self.group_size)
        
        return out
    

##############################################################
## LHSBv2 - split dim and define 4 attn blocks 
class LHSBv2(nn.Module):
    def __init__(self,
                 dim,  
                 attn_drop=0.,
                 proj_drop=0.,
                 n_levels=4,
                 branch_dim=[8,12,16,48],
                 high_skip=False,
                 use_aggr_expand=True,
                 idx = 0,
                 window_size=8,
                 ):
        
        super().__init__()
        self.n_levels = n_levels
        self.branch_dim = branch_dim
        
        self.high_skip = False   # high-freq skip connection
        self.use_aggr_expand = use_aggr_expand
        self.idx = idx

        ### Downsize ###
        self.dslist = nn.ModuleList()
        mean_topk = 3
        self.num_means = 64
        # self.meanattn = MeanAttn(self.branch_dim,
        #                        window_size,
        #                        topk=mean_topk,
        #                        mean_size=self.num_means,
        #                        )
        
        ### IntraWindowAttn ###
        self.interattn = InterWindowAttn(dim)
        
        ### Spatial Weighting ###
        self.mfr = nn.ModuleList([
            NeighborAttn(self.branch_dim[i],
                           window_size=window_size,
                           attn_drop=attn_drop,
                           proj_drop=proj_drop,
                           down_scale=2**i)
            for i in range(self.n_levels)])
        
        ### Feature Aggregation ###
        # self.aggr = MultiAggregation(dim,
        #                             groups=2,
        #                             branch_dim=branch_dim,)
        self.aggr = nn.Sequential(
                nn.Conv2d(dim, dim//2, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(dim//2, dim//2, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim//2, dim, 1, 1, 0)
            )
        
        ### Activation ###
        self.act = nn.GELU() 

        ### Gating ###
        self.gate = nn.Linear(dim, dim)
        
        ### low-to-high ###
        self.lth = nn.ModuleList()
        for i in range(self.n_levels - 1):
            # dim: 채널 수 (LR과 HR 채널 수가 같다면 그대로 사용)
            # 파라미터 최소화를 위해 Depthwise Convolution 사용
            self.lth.append(
                Low_to_high(lr_dim=self.branch_dim[i], 
                            hr_dim=self.branch_dim[i+1],
                            )
            )
        
        
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
        
        ### downsize global key/value ###
        # means = x.view(batch, chan, H*W)
        # means = torch.mean(rearrange(means, 'b c (mns ptc) -> mns (b ptc) c', mns=self.num_means), 
        #                    dim=-2)
        # downsized_feats, means = self.meanattn(x, downsized_feats, means)
        
        ### InterWindowAttn
        # inter_out = self.interattn(x)
        
        ####################################
        ## multi-scale processing 
        ####################################        
        for i in range(self.n_levels):
            if i < self.n_levels - 1 :  # except last 
                branch = self.mfr[i](downsized_feats[i])      
                branch_original_shape = F.interpolate(branch, size=(H, W), mode='bilinear')

                ## make to original (H,W) for final output
                branch_original_shape = branch_original_shape \
                    * gate[:, gate_idx:gate_idx+self.branch_dim[i]]
                # * self.act(gate[:, gate_idx:gate_idx+self.branch_dim[i]])
    
                out.append(branch_original_shape)
                gate_idx = self.branch_dim[i]
                
                # ------------------------- low to high ------------------------------
                downsized_feats[i+1] = self.lth[i](branch, downsized_feats[i+1])
                # --------------------------------------------------------------------
            
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
                 branch_dim=[8,12,16,42],
                 window_size=8,):
        
        super().__init__()

        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim)
        self.norm3 = LayerNorm(dim)
        self.norm4 = LayerNorm(dim)

        ### InterWindow Block and FFN
        group_size=8
        cluster_size=64
        self.iwb = InterWindowAttn(
            dim,
            group_size=8,
            cluster_size=64,
        )
        # self.ffn = GatedFFN(
        #     dim,
        # )
        
        ### Multiscale Block
        # self.lhsb = LHSBv2(dim, 
        #                  attn_drop=attn_drop, 
        #                  proj_drop=drop,
        #                  branch_dim=branch_dim,
        #                  window_size=window_size,
        #                  ) 
        
        ### Feedforward layer
        self.pcfn = PCFN(dim, growth_rate=ffn_scale) 

    def forward(self, x):
        x = self.iwb(self.norm1(x)) + x
        # x = self.ffn(self.norm2(x)) + x
        
        # x = self.lhsb(self.norm3(x)) + x
        x = self.pcfn(self.norm4(x)) + x
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
                 branch_dim=[8,12,16,42],
                 window_size=8,):
        
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
                                              branch_dim=branch_dim,
                                              window_size=window_size,
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
    branch_dim = [8, 12, 16, 44]
    n_blocks = 12
    window_size = 8
    
    dim = sum(branch_dim)
    
    model = LMLT(dim=dim, 
                 n_blocks=n_blocks, 
                 ffn_scale=2.0, 
                 upscaling_factor=2,
                 branch_dim = branch_dim,
                 window_size=window_size,)
    # model = LMLT(dim=36, n_blocks=12, ffn_scale=2.0, upscaling_factor=2)
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)
