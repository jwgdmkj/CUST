import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange
from basicsr.archs.arch_util import trunc_normal_

from itertools import repeat
import collections.abc
from typing import Tuple

from pdb import set_trace as st
import numpy as np

f"""
CUST_only_hpi.py

CUST 기반, GMLT 대신 HPI-Net 사용
"""
# df2k download : https://github.com/dslisleedh/Download_df2k/blob/main/download_df2k.sh
# dataset prepare : https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md
        
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, channel_first=True):
        super().__init__()
        self.channel_first = channel_first
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # [핵심] 그냥 nn.LayerNorm을 씁니다. (GroupNorm 아님!)
        # 이 모듈은 C++로 최적화되어 있어 메모리를 적게 씁니다.
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        if self.channel_first == False:
            return self.norm(x)

        elif self.channel_first == True:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)
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

class lightFFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.in_proj = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.act = nn.GELU()
        self.dwc = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.out_proj = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.act(self.in_proj(x))
        x = self.dwc(x) + x
        return self.out_proj(x)

# LightweightConvFFN
class GatedFFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, split_rate=0.2):
        super().__init__()
        
        ### var
        self.dim = dim
        hidden_dim = int(dim * growth_rate)
        dwc_dim = int(hidden_dim * split_rate)
        self.split_dim = [dwc_dim, hidden_dim - dwc_dim]
    
        ### algorithm
        self.in_proj = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.dwc = nn.Conv2d(dwc_dim, dwc_dim, 3, stride=1, padding=1)
        self.out_proj = nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.in_proj(x)) # b c h w
        x1, x2 = torch.split(x, self.split_dim, dim=1)
        x1 = self.dwc(x1)
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
## Inter-Window Attn
class CUST(nn.Module):
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
        hidden_dim = dim
        
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

        x_grouped = x_grouped.view(B*ng, gs*ws, chan)   # [배치*그룹수, 그룹사이즈*윈도우사이즈, 채널]
        
        assign_id = sim.argmax(dim=-1).view(B*ng, gs*ws)
        sorting_indices = torch.argsort(assign_id, dim=1)
         
        ### x와 id를 정렬된 순서대로 재배열
        gather_idx = sorting_indices.unsqueeze(-1).expand(-1, -1, chan)   # [B*ng, gs*ws, C]: 채널 방향으로 expand
        x_sorted = torch.gather(x_grouped, 1, gather_idx)   # [B*ng, gs*ws, C]
        id_sorted = torch.gather(assign_id, 1, sorting_indices)   # [B*ng, gs*ws]

        cs = self.window_size ** 2  # chunk_size
        nc = (gs*ws) // cs  # num_chunk
        
        # A) Query
        q_chunks = x_sorted.view(B * ng, nc, cs, chan)
        q_ids = id_sorted.view(B * ng, nc, cs)
        
        ###########################################################################
        # B-2) (전 반 청크 + 현 청크 + 다음 반 청크)
        pad_x = torch.zeros(B*ng, cs//2, chan, device=device)
        pad_x = torch.cat([pad_x, x_sorted, pad_x], dim=1)
        pad_id = torch.full((B*ng, cs//2), -1, device=device)
        pad_id = torch.cat([pad_id, id_sorted, pad_id], dim=1)  # [B*ng, gs*ws+64]
        ###########################################################################
        
        # Unfold 통해 슬라이딩 윈도우 생성(win=128, stride=64)
        kv_chunks = pad_x.unfold(1, cs*2, cs).permute(0, 1, 3, 2)
        kv_ids = pad_id.unfold(1, cs*2, cs) # [B*ng, 128, nc]
        
        ###############################################################################
        # Attn with Masking
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

        gate = self.act(self.gate_proj(x_sorted)).view(B*ng, gs, ws, -1)  # [B*ng, gs*ws, 1]
        out = out * gate
        
        #####################################
        # Unsort & Restore(원래 순서로 복구)
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
        
        #####################################
        # 유사도 계산 및 다른 패치와 엮기 계산
        #####################################
        sim = x_grouped.mean(dim=3) # [B, num_group, group_size, c]
        sim = torch.einsum('b g w p c, b g k c -> b g w p k', x_grouped, sim)   # [B, ng, gs, ws, gs]
        cana_out = self.cluster_cat_kv(x_grouped, sim)
        
        #####################
        # 원래 shape로 되돌림
        #####################
        global_out = self.window_group_reverse(cana_out, x.shape, (pad_h, pad_w))

        return global_out
    
    
    
##############################################################
## Intra-Window Attn
def patch_divide(x, step, ps):
    """Crop image into patches(이미지를 지정된 크기(ps)로 자르되, 서로 겹치게 자른다.)
    Args:
        x (Tensor): Input feature map of shape(b, c, h, w).
        step (int): Divide step. 'ps-2'
        ps (int): Patch size. [16, 20, 24, 28, 16, 20, 24, 28]
    Returns:
        crop_x (Tensor): Cropped patches.
        nh (int): Number of patches along the horizontal direction.
        nw (int): Number of patches along the vertical direction.
    """
    b, c, h, w = x.size()
    if h == ps and w == ps: # h==w==patch_size일 경우, step은 ps-2가 아닌 ps
        step = ps
    crop_x = []
    nh = 0
    
    ##########################################################################
    # if h == 100 : range(0, 98, 14) --> i = [0, 14, 28, ... 84] 이렇게 루프를 돔.
    # top, down = (0, 16), ... (84, 100)
    
    # down > h : 이미지 끝(h)가 패치 이동간격(step)으로 딱 떨어지지 않을 때, 자투리 공간이 남을 때 T
    # h=101일 때, range(0,99,14)이므로 i=[0,14,...98], 즉 i=98이 추가됨.
    # 이 때, down=98+16>h로, 범위를 벗어남. 그러면 (top,down)=(85,101)로, 이미지 맨 끝을 down으로 갖게 됨.
    
    # right > w : 마찬가지로, 이미지 너비(w)가 패치 이동간격(step)으로 떨어지지 않을 때
    # w=75일 때, range(0, 73, 14)이므로 j=[0,14,...70], 즉 j=70이 추가됨.
    # 그러면 right=70+16>w이 되므로, (right,left) = (56,70)로, 이미지 맨 끝을 right로 갖게 됨.
    
    # 이렇게 하나의 높이에서 여러 개의 이미지를 crop으로 잘라냄.
    # nh =세로방향으로 잘라낸 개수 / nw = 총 크롭된 이미지 / 세로로 잘라낸 수 = 가로로 잘라낸 수
    ##########################################################################
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        nh += 1
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    
    #####################################
    # crop_x : [(총 crop된 횟수) x (B, dim, ps, ps)] = 42 x [B 40 16 16]
    # stack 및 permute로, [b 42 40 16 16]으로 만들고, nh, nw와 함께 반환
    #####################################
    crop_x = torch.stack(crop_x, dim=0)  # (n, b, c, ps, ps)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()  # (b, n, c, ps, ps)
    return crop_x, nh, nw


def patch_reverse(crop_x, x, step, ps):
    """Reverse patches into image.
    Args:
        crop_x (Tensor): Cropped patches. [B, num_crop, dim, ps, ps]
        x (Tensor): Feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        output (Tensor): Reversed image. [B, dim(40), H, W]
    """
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    
    ####################################################
    # 크롭된 이미지를 순서대로 다시 집어넣기(output에).
    # 순서가 range(crop_x[1]=num_crop)이 아닌, 
    # 집어넣을 간격을 먼저 정하고, 거기에 crop_x[:,index]를 더함
    ####################################################
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
            
    ####################################################
    # patch overlap으로 인해, 중첩되어 2번 더해진 영역들을 2로 나눔.
    # [height, 2]만큼, 또는 [2, width]만큼 더해진 영역은 2번 더해졌음.
    # [2, 2] 영역은 4번 더해졌음. 이는 for문 2개를 돌면서 4로 나눠짐.
    ####################################################
    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
        
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2
        
    return output

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, 
                      hidden_features, 
                      kernel_size=kernel_size, 
                      stride=1, 
                      padding=(kernel_size - 1) // 2, 
                      dilation=1,
                      groups=hidden_features), 
            nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class ConvFFN(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 kernel_size=5, 
                 act_layer=nn.GELU):
        
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        f"""
        Linear(40, 96) -- act -- DWConv@5 -- + -- Linear(96, 40)
                               |-------------|
        """ 
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x
    

class Attention(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
    """

    def __init__(self, dim, heads, qk_dim):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
        

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
       
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        
        out = F.scaled_dot_product_attention(q,k,v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)



class LRSA(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        num (int): Number of blocks.
        qk_dim (int): Channels of query and key in Attention.
        mlp_dim (int): Channels of hidden mlp in Mlp.
        heads (int): Head numbers of Attention.
        
    TAB이 Global한 정보 파악이 위주였다면,
    LRSA는 로컬한 정보 확보가 목표.
    
    patch_divide 및 reverse (with overlapping)의 목표 :
        step(stride)를 ps(patch_size)보다 작게 해서, 경계면에 있는 애들의 정보를 더 잘 파악하기 위함.
    """

    def __init__(self, 
                 dim, 
                 qk_dim, 
                 heads=1):
        super().__init__()

        self.norm1 = LayerNorm(dim, channel_first=False)
        self.norm2 = LayerNorm(dim, channel_first=False)
        
        self.attn = Attention(dim, heads, qk_dim)
        self.ffn = ConvFFN(dim, dim*2)


    def forward(self, x, ps):
        
        ############################
        # Patch Divide - LN - ATTN
        
        # ps(patch_size) : [16, 20, 24, 28, 16, 20, 24, 28]
        # 만들어진 q,k,v(=[크롭된 이미지 개수xB, head, ps*ps, head_dim]) 간의 attn 진행
        ############################
        step = ps - 2
        crop_x, nh, nw = patch_divide(x, step, ps)  # (b, n, c, ps, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')

        crop_x = self.attn(self.norm1(crop_x)) + crop_x
        crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)
        
        ################################
        # Patch Reverse - LN - MLP(ConvFFN)
        
        # patch_reverse input : crop_x, x(첫 input), step(=patch_size - 2), ps)
        ################################
        x = patch_reverse(crop_x, x, step, ps)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = self.ffn(self.norm2(x), x_size=(h, w)) + x
        x = rearrange(x, 'b (h w) c->b c h w', h=h)
        
        return x
    
    
##############################################################
## Block
class MainBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 ffn_scale=2.0, 
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 branch_dim=[8,12,16,42],
                 window_size=8,
                 patch_size=16,):
        
        super().__init__()

        self.norm1 = LayerNorm(dim) 
        self.norm2 = LayerNorm(dim)
        # self.norm3 = LayerNorm(dim)
        # self.norm4 = LayerNorm(dim)
        
        self.patch_size = patch_size
        
        ### cust
        # self.cust = CUST(dim, window_size=window_size,)
        
        ### ffn
        # self.ffn1 = lightFFN(dim, growth_rate=ffn_scale) 
        
        ### Multiscale Block
        self.attns = LRSA(dim, 
                         dim,
                         ) 
        
        ### Feedforward layer
        # self.ffn2 = lightFFN(dim, growth_rate=ffn_scale) 
        self.mid_conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        # x = self.cust(self.norm1(x)) + x
        # x = self.ffn1(self.norm2(x)) + x
        
        # x = self.attns(self.norm3(x)) + x
        # x = self.ffn2(self.norm4(x)) + x
        x = self.attns(x, self.patch_size)
        x = self.mid_conv(x) + x
        return x
        

##############################################################
## Overall Architecture
# ARCH_REGISTRY.register()
class CUSTNet(nn.Module):
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
        self.dim = dim
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]  # stochastic depth decay rule
        
        patch_size = [16, 20, 24, 28, 16, 20, 24, 28, 16, 20, 24, 28]
        self.feats = nn.Sequential(*[MainBlock(dim, 
                                              ffn_scale, 
                                              drop=drop_rate,
                                              attn_drop=attn_drop_rate,
                                              drop_path=dpr[i],
                                              branch_dim=branch_dim,
                                              window_size=window_size,
                                              patch_size=patch_size[i],
                                              ) 
                                     for i in range(n_blocks)])

        # reconstruction
        self.upscale = upscaling_factor
        if self.upscale == 4:
            self.upconv1 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
            
        elif self.upscale == 2 or self.upscale == 3:
            self.upconv = nn.Conv2d(self.dim, self.dim * (self.upscale ** 2), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
    
        self.last_conv = nn.Conv2d(self.dim, 3, 3, 1, 1)
        if self.upscale != 1:
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            

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
        
        if self.upscale != 1: 
            base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        else: 
            base = x
            
        # check image size
        x = self.check_img_size(x)  
        
        # patch embed
        x = self.to_feat(x)
        
        # module, and return to original shape
        x = self.feats(x) + x
        x = x[:, :, :H, :W]
        
        ## reconstruction
        if self.upscale == 4:
            x = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
            x = self.lrelu(self.pixel_shuffle(self.upconv2(x)))
        elif self.upscale == 1:
            x = x
        else:
            x = self.lrelu(self.pixel_shuffle(self.upconv(x)))
        x = self.last_conv(x) + base
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
    branch_dim = [44]
    n_blocks = 12
    window_size = 8
    
    dim = sum(branch_dim)
    
    model = CUSTNet(dim=dim, 
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
