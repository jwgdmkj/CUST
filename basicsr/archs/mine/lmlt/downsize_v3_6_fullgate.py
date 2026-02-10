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
downsize_v3_6_fullgate.py

그룹핑 버전
v3_6 : v3과 기본적으로 같되, v_3_5처럼, 이전 청크 절반, 다음 청크를 절반 사용.
또한 fullgate
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
    def __init__(self, dim, growth_rate=2.0, p_rate=0.2):
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
## Inter-Window Attn
class InterWindowAttn(nn.Module):
    def __init__(self, 
                 dim, 
                 window_size=8, 
                 group_size=9):
        super().__init__()
        
        ### var
        self.window_size = window_size # 8
        self.group_size = group_size   # 32
        self.scale = dim ** -0.5
        
        # pe
        self.pe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        # Query와 Key를 변환하는 레이어
        hidden_dim = 32
        
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
        # gate before proj
        self.gate_proj = nn.Linear(dim, dim)
        self.act = nn.Sigmoid()
        

    def window_group_partition(self, x):
        """
        이미지를 윈도우 단위로, 그리고 다시 그룹 단위로 재정렬하는 함수
        x: [B, C, H, W]
        Return: [B, Num_Groups, 32(Wins), 64(Pixels), C]
        """
        B, C, H, W = x.shape
        ws = self.window_size
        gs = self.group_size    # 얼마나 많은 윈도우를 그룹할 것인가(제곱)
        
        ####################
        # Pad
        ####################
        target_unit = ws * gs
        pad_h = (target_unit - H % target_unit) % target_unit
        pad_w = (target_unit - W % target_unit) % target_unit

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        H_pad, W_pad = x.shape[2], x.shape[3]
        
        ####################
        # Partition
        ####################
        x = x.view(B, C, H_pad // ws, ws, W_pad // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()    
        x = x.view(B, H_pad // ws, W_pad // ws, ws * ws, C)   # [B, (num_win), (win_size), C]

        ####################
        # Grouping
        ####################
        H_wins, W_wins = x.shape[1], x.shape[2] # 윈도우 개수 (H, W 방향)
        x = x.view(B, H_wins // gs, gs, W_wins // gs, gs, -1, C)    # [B, grp_h, gs, grp_w, w, win_size, c]
        
        x = x.permute(0,1,3,2,4,5,6).contiguous()   # [B, grp_h, grp_w, gs, gs, win_size, c]
        x = x.view(B, -1, gs*gs, ws*ws, C)
        
        return x, pad_h, pad_w
    
    def window_group_reverse(self, x, original_shape, padded_size):
        b, ng, gs_pad, ws_pad, chan = x.shape
        ws, gs = self.window_size, self.group_size
        _,_,H,W = original_shape
        
        ########################
        # Pad 크기 계산
        ########################
        target_unit = ws * gs
        H_pad = H + padded_size[0]
        W_pad = W + padded_size[1]
        
        ###############################
        # 그리드 차원 재배치 & 차원 재배치
        ###############################
        gh = H_pad // target_unit
        gw = W_pad // target_unit
        x = x.view(b, gh, gw, gs, gs, ws, ws, chan)
        
        # 목표 : [b, c, h_pad, w_pad]
        # h_pad는 gh -> gs -> ws 순서,
        # w_pad는 gw -> gs -> ws 순서로 합쳐져야 함
        x = x.permute(0,7,1,3,5,2,4,6).contiguous()
        x = x.view(b, chan, H_pad, W_pad)
        
        ########################
        # 패딩 제거
        ########################
        if padded_size[0] > 0 or padded_size[1] > 0 :
            x = x[:,:,:H,:W]
        
        return x
    
    def cluster_cat_kv(self, x_grouped, sim):
        f"""
        다음 청크를 키/밸류에 추가. 
        단, 다른 window를 높은 유사도로 갖는 패치는 -inf 처리
        
        sim : [B, ng, gs, ws, gs] : 각 패치들(ws)과, 그룹 내의 윈도우들(gs) 간의 유사도
        """
        B, ng, gs, ws, chan = x_grouped.shape
        device = x_grouped.device
        
        #######################################################################################
        # assignment, flatten & sort : 유사도 기반 ID 부여 후, 그룹 별로 펴서 정렬
        #                              idx : [0~(gs*ws)-1]
        
        # argmax : 가장 높은 값을 가진 값의 idx 반환.
        #          즉 각 패치와 가장 유사한 윈도우의 idx. [배치*그룹수, 그룹사이즈*윈도우사이즈]
        
        # sorting_indices: 그룹 내 패치들을 비슷한 친구들끼리 정렬했을 때의 패치들의 "원래 인덱스."
        #                  torch.argsort(dim=1) : -> 방향으로, 오름차순으로 둘 때 각 요소의 '원래 idx'
        #                  ex) [5,9,3]의 경우 오름차순은 [3,5,9]이고 이 값들의 원래 idx는 [2,0,1]
        
        # gather_idx : sorting_indices는 픽셀 위치만 알려주므로, 채널들이 모두 같이 이동해야 하니 C로 확장
        
        # x_sorted : sorting_indices를 기반으로 x를 옮김. 
        #            [2,0,1]을 idx삼아 [5,9,3]을 옮기면 [3,5,9]가 됨.
        # id_sorted: 정렬된 x의 각 패치들의 원래 가장 가까운 window의 idx
        #######################################################################################
        x_grouped = x_grouped.view(B*ng, gs*ws, chan)   # [배치*그룹수, 그룹사이즈*윈도우사이즈, 채널]
        
        assign_id = sim.argmax(dim=-1).view(B*ng, gs*ws)
        sorting_indices = torch.argsort(assign_id, dim=1)
         
        ### x와 id를 정렬된 순서대로 재배열
        gather_idx = sorting_indices.unsqueeze(-1).expand(-1, -1, chan)   # [B*ng, gs*ws, C]: 채널 방향으로 expand
        x_sorted = torch.gather(x_grouped, 1, gather_idx)   # [B*ng, gs*ws, C]
        id_sorted = torch.gather(assign_id, 1, sorting_indices)   # [B*ng, gs*ws]

        ###################################################
        # Chunking
        
        # pad_id: 마지막 청크를 위해 패딩된 데이터가 실제 데이터와 매칭되는 것을 막기 위해, 
        #         id(가장 유사한 윈도우의 아이디)를 -1로 부여
        ###################################################
        cs = self.window_size ** 2  # chunk_size
        nc = (gs*ws) // cs  # num_chunk
        
        # A) Query
        q_chunks = x_sorted.view(B * ng, nc, cs, chan)
        q_ids = id_sorted.view(B * ng, nc, cs)
        
        # B) Key/val(이전 반 청크 + 현 청크 + 다음 반 청크)
        pad_x = torch.zeros(B*ng, cs//2, chan, device=device)
        pad_x = torch.cat([pad_x, x_sorted, pad_x], dim=1)
        
        pad_id = torch.full((B*ng, cs//2), -1, device=device)
        pad_id = torch.cat([pad_id, id_sorted, pad_id], dim=1)  # [B*ng, gs*ws+64]
        
        # Unfold 통해 슬라이딩 윈도우 생성(win=128, stride=64)
        kv_chunks = pad_x.unfold(1, cs*2, cs).permute(0, 1, 3, 2)
        kv_ids = pad_id.unfold(1, cs*2, cs) # [B*ng, 128, nc]
        
        ###############################################################################
        # Attn with Masking
        
        # q_ids == kv_ids[:,:,:64] -> 쿼리의 id == 키의 초반 id(후반 id는 패딩으로, q[i+1]의 id와 같음)
        
        # mask : [B*ng, gs, ws, 2*ws] 
        # q_ids와 kv_ids를 unsqueeze 시, [b*ng, gs, ws, 1]와 [b*ng, gs, 1, 2*ws] 
        # q\k  0 1 0 2
        #  0   T F T F
        #  0   T F T F
        #  1   F T F F 
        # 다음과 같은 효과를 주기 위한 unsqueeze
        # attn 계산 결과는 [q의 ws, k의 ws]이므로, 그에 대응하기 위한 mask
        # 이후 false에 해당하는 idx는 -inf로 채운 뒤 softmax, 그 이후 v 행렬곱 진행
        ###############################################################################
        q = self.to_q(q_chunks)      # [BG, Chunks, 64, C]
        k = self.to_k(kv_chunks)     # [BG, Chunks, 128, C]
        v = self.to_v(kv_chunks)     # [BG, Chunks, 128, C]
        attn = (q @ k.transpose(-2, -1)) * self.scale # [BG, Chunks, 64, 128]
        
        # *** Masking ***
        # Query의 ID와 Key의 ID가 같을 때만 True (같은 그룹끼리만)
        # q_ids: [..., 64, 1], kv_ids: [..., 1, 128]
        mask = (q_ids.unsqueeze(-1) == kv_ids.unsqueeze(-2))
        
        # False인 부분(ID 불일치)을 작은값으로 마스킹 -> 불순물 완벽 차단
        min_val = -1e4
        attn = attn.masked_fill(~mask, min_val)

        # Softmax & Aggregate
        attn = attn.softmax(dim=-1)
        out = attn @ v # [b*ng, Chunks, 64, C]

        gate = self.act(self.gate_proj(x_grouped)).view(B*ng, gs, ws, -1)  # [B*ng, gs*ws, 1]
        out = out * gate
        
        #####################################
        # Unsort & Restore(원래 순서로 복구)
        
        # argsort : 뒤섞인 패치들의 원래 위치(인덱스, sorting_indices) 배열을, 오름차순으로 둠.
        #           ex) [5,9,3]을 [3,5,9]로 뒀을 때, sorting_indices는 [2,0,1]이 됨.
        #               이를 다시 오름차순, 즉 [0,1,2]로 두면, [3,5,9]는 다시 [5,9,3]이 됨.
        
        # out_flat : 이후 원래 순서로 복구
        # out : 원래 텐서 형태로 복구
        #####################################
        out = out.view(B * ng, gs*ws, chan)
        out = self.proj(out)

        inverse_indices = torch.argsort(sorting_indices, dim=1)
        inverse_indices = inverse_indices.unsqueeze(-1).expand(-1, -1, chan)

        out = torch.gather(out, 1, inverse_indices)
        out = out.view(B, ng, gs, ws, chan)
        
        return out
    
    def cluster_pool_kv(self, x_grouped, sim):
        f"""
        다른 청크들을 풀링해서 키/밸류에 추가
        """
        return x

    def forward(self, x):
        # x: [B, C, H, W]
        batch, chan, H, W = x.shape

        ###################
        # 이미지 -> 그룹화된 윈도우 텐서로 변환([B, num_group, group_size, win_size, c]) 뒤 키 풀링
        ###################
        x = x + self.pe(x)
        x_grouped, pad_h, pad_w = self.window_group_partition(x)
        
        ###################
        # 유사도 계산
        ###################
        sim = x_grouped.mean(dim=3) # [B, num_group, group_size, c]
        sim = torch.einsum('b g w p c, b g k c -> b g w p k', x_grouped, sim)   # [B, ng, gs, ws, gs]
        global_out = self.cluster_cat_kv(x_grouped, sim)
        
        #####################
        # 원래 shape로 되돌림
        #####################
        global_out = self.window_group_reverse(global_out, x.shape, (pad_h, pad_w))

        return global_out
    

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
        
        ### IntraWindowAttn ###
        # self.group_size = 8
        # self.cluster_size = 64
        # self.interwins = nn.ModuleList()
        # for i in range(self.n_levels - 1):
        #     self.interwins.append(
        #         InterWindowAttn(dim=self.branch_dim[i],
        #                         group_size=2**(3-i),)
        #     )
        
        self.interattn = InterWindowAttn(dim,
                                         window_size=window_size,)
        
        ### Spatial Weighting ###
        self.mfr = nn.ModuleList([
            NeighborAttn(self.branch_dim[i],
                           window_size=window_size,
                           attn_drop=attn_drop,
                           proj_drop=proj_drop,
                           down_scale=2**i)
            for i in range(self.n_levels)])
        
        ### Feature Aggregation ###
        self.aggr = MultiAggregation(dim,
                                    groups=2,
                                    branch_dim=branch_dim,)
        # self.aggr = nn.Sequential(
        #         nn.Conv2d(dim, dim//2, 1, 1, 0),
        #         nn.GELU(),
        #         nn.Conv2d(dim//2, dim//2, 3, 1, 1),
        #         nn.GELU(),
        #         nn.Conv2d(dim//2, dim, 1, 1, 0)
        #     )
        
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
        
        # self.attnfusion = nn.Conv2d(dim*2, dim, 1)
        
        
    def forward(self, x, aggr=None):
        f"""
        x -- x1 - Interwin(group=8) - NeighborAttn - C
          |- x2 - Interwin(group=4) - NeighborAttn --|
          |- x3 - Interwin(group=2) - NeighborAttn --|
          |- x4 --------------------- NeighborAttn --|
        """
        batch, chan, H, W = x.shape
        
        ####################################
        ## InterWindow attn
        ####################################
        x = self.interattn(x) + x
 
        ####################################
        ## split & ngating
        ####################################
        xc = torch.split(x, self.branch_dim, dim=1)
        SA_before_idx = None
        out = []
        
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
        
        # out = self.attnfusion(torch.cat([out, interwin_output], dim=1))  
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

        # self.norm1 = LayerNorm(dim) 
        # self.norm2 = LayerNorm(dim)
        self.norm3 = LayerNorm(dim)
        self.norm4 = LayerNorm(dim)
        
        ### Multiscale Block
        self.lhsb = LHSBv2(dim, 
                         attn_drop=attn_drop, 
                         proj_drop=drop,
                         branch_dim=branch_dim,
                         window_size=window_size,
                         ) 
        
        ### Feedforward layer
        self.pcfn = PCFN(dim, growth_rate=ffn_scale) 

    def forward(self, x):
        # x = self.iwb(self.norm1(x)) + x
        # x = self.ffn(self.norm2(x)) + x
        
        x = self.lhsb(self.norm3(x)) + x
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
    # x = torch.randn(1, 3, 640, 360)
    # x = torch.randn(1, 3, 427, 240)
    x = torch.randn(1, 3, 320, 180)
    # x = torch.randn(1, 3, 256, 256)

    # large
    # branch_dim=[8,12,16,44]
    # n_blocks = 12
    
    # tiny
    branch_dim = [8, 12, 16, 32]
    n_blocks = 12
    window_size = 8
    
    dim = sum(branch_dim)
    
    model = LMLT(dim=dim, 
                 n_blocks=n_blocks, 
                 ffn_scale=2.0, 
                 upscaling_factor=4,
                 branch_dim = branch_dim,
                 window_size=window_size,)
    # model = LMLT(dim=36, n_blocks=12, ffn_scale=2.0, upscaling_factor=2)
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)
