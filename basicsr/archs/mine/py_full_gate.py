import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
import math

# from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath

from itertools import repeat
import collections.abc
from typing import Tuple

from pdb import set_trace as st
import numpy as np
from einops import rearrange, repeat

f"""
피라미드 + VSSD + Coef + 키만 풀링 + gate 추가
py_full_gate.py
"""

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

##############################################################
## HSM-SSD
to_ttensor = lambda *args: tuple([tTensor(x) for x in args]) if len(args) > 1 \
    else tTensor(args[0])

class BlqSSM(nn.Module):
    def __init__(self,
                 dim,
                 state_dim,
                 ssd_expand,
                 drop_path=0.1,
                 downsize_idx=0,
                 type='spatial',
                 A_init_range=(1, 16),
                 device=None,
                 dtype=None,
                 bias=False,
                 conv_bias=True,
                 nheads=4,
                 n_levels=4,
                 ):
        super().__init__()

        self.dim = dim
        self.state_dim=state_dim
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * dim)

        # var for downsize
        self.downsize_idx = downsize_idx
        self.n_levels = n_levels

        self.ngroups = 1
        self.conv_dim = self.d_inner + 2*self.ngroups * self.state_dim  # embed + 2*state_dim
        self.nheads = nheads
        self.head_dim = self.d_inner // self.nheads # 임시

        factory_kwargs = {"device": device, "dtype": dtype}
        self.type = 'spatial' if type=='spatial' else 'channel'

        self.ssd_positive_dA = True
        self.d_in_proj = 2*self.d_inner + 2*self.ngroups*state_dim + self.nheads

        ## Parameter ----------------------------------------
        # A
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # dt
        f"""
        필수는 아니지만, 초기 안정화를 위해 존재.
        만일 bias=False/True라면, 모든 head가 거의 같은 시간 스케일에서 시작하지만,
        head 별로 log-uniform으로 뽑은 dt0을 정확히 초기출력으로 만들어, 학습 안정성과 속도가 증가.

        dt_bias를 만들기 위함. 그 파라미터 사이즈는 nhead
        u = torch.rand(...) -> [0,1] 균일분포
        u * (log(dt_max) ...) -> [0, Δ] 균일분포(여기서 Δ = log(dt_max) - log(dt_min)).
        + log(dt_min) -> [log(dt_min), log(dt_max)] 균일분포(즉 값이 [0, 5]).
        exp(log_dt) -> [dt_min, dt_max]에서 log-균일분포(즉 값이 [1, 32]). 원래 우리가 원하는 범위값.
        clamp : 텐서 값이 dt_init_floor보다 작다면, 이를 dt_init_floor로 올려줌으로써 dt가 너무 작음을 방지.
        """
        dt_max = 0.1
        dt_min = 0.001
        dt_init_floor= 1e-4

        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)

        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        f"""
        dt를 softplus의 역함수값으로 변환 후, 이를 파라미터화.
        즉 dt라는 양수값을 받아, 그 값이 softplus를 통과했을 때 다시 dt가 나오도록 변환.
        dt0 = softplus(dt_bias)이기 위해,
        dt_bias = softplus^(-1)(dt_0)을 만든 것.

        초기에 log-uniform으로 뽑은 특정한 값 dt0이 있다.
        forward에서 dt=dt_0이길 원함.
        위의 과정을 거치면, dt가 초기에 0에 가까울 때,
        F.softplus(0 + dt_bias) = dt0를 성립시켜,
        여기에서 샘플링한 log-uniform dt(=dt0)가 정확히 초기값이 되게 함.
        """
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        ## Algorhithm ----------------------------------------
        # Processing
        self.in_proj = nn.Linear(self.dim, int(self.d_in_proj), bias=bias, **factory_kwargs)
        # print(f'0) init - in_proj : {int(self.d_in_proj)} due to 2*d_inner + 2*ngroups*state_dim + nheads')
        # print(f'here, d_inner:{self.d_inner}, state_dim:{self.state_dim}, nhead:{self.nheads}')
        self.act = nn.SiLU()
        self.conv2d = nn.Conv2d(
            in_channels = self.conv_dim,
            out_channels=self.conv_dim,
            groups=self.conv_dim,
            bias=conv_bias,
            kernel_size=3,
            padding=1,
            **factory_kwargs,
        )   # conv_dim = embedding + 2*state_dim

        self.norm = nn.LayerNorm(self.d_inner)  # 입력이 [B, HW, C]임을 전제
        self.out_proj = nn.Linear(self.d_inner, self.dim, bias=bias, **factory_kwargs)


    #######################################################################
    ## Non-causal SSM (단, head를 제거한 상태)
    def non_causal_linear_attn(self, x, dt, A, B, C, D, H=None, W=None):
        ''''
        non-casual attention duality of mamba v2
        x: (B, L, H, D), equivalent to V in attention
        dt: (B, L, nheads)
        A: (nheads) or (d_inner, d_state)
        B: (B, L, d_state), equivalent to K in attention
        C: (B, L, d_state), equivalent to Q in attention
        D: (nheads), equivalent to the skip connection
        '''

        batch, seqlen, head, dim = x.shape    # original : [B, L, head, dim//head]
        dstate = B.shape[2]
        V = x.permute(0, 2, 1, 3) # (B, H, L, D)
        dt = dt.permute(0, 2, 1) # (B, H, L)
        dA = dt.unsqueeze(-1) * A.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
        if self.ssd_positive_dA: dA = -dA

        # print(f'6) x:{x.shape}, V:{V.shape}, dt:{dt.shape}, dA:{dA.shape}')

        V_scaled = V * dA
        K = B.view(batch, 1, seqlen, dstate)# (B, 1, L, D)
        if getattr(self, "__DEBUG__", False):
            A_mat = dA.cpu().detach().numpy()
            A_mat = A_mat.reshape(batch, -1, H, W)
            setattr(self, "__data__", dict(
                dA=A_mat, H=H, W=W, V=V,))

        ## get kv via transpose K and V
        KV = K.transpose(-2, -1) @ V_scaled # (B, H, dstate, D)
        Q = C.view(batch, 1, seqlen, dstate)#.repeat(1, head, 1, 1)
        x = Q @ KV # (B, H, L, D)
        x = x + V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)

        return x

    #######################################################################
    ## Overall SSM
    def forward(self, x):
        if self.type == 'spatial':
            batch, chan, H, W = x.shape
            x = x.flatten(2).contiguous()   # [B, C, HW]
        elif self.type == 'channel':
            batch, H, W, chan = x.shape
            x = x.flatten(1, 2).contiguous()    # [B, HW, C]

        x = x.transpose(1, 2)   # [B, HW(=L), C] <- [B, C, HW] vice versa

        # print('1) x shape : ', x.shape)

        zxbcdt = self.in_proj(x)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)

        # print('2) A and zxbcdt : ', A.shape, zxbcdt.shape)

        #############################################
        # 1. Z, A, B, c, delta 만들기
        #############################################
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.state_dim, self.nheads],
            dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)


        #############################################
        # 2. xBC, dt downsize
        #############################################
        ratio = self.n_levels - 1 - self.downsize_idx # 3-2-1-0

        #2D Convolution
        xBC = xBC.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # downsize
        if ratio > 0 :
          pool_H = H//2**ratio
          pool_W = W//2**ratio
          pool_size = (pool_H, pool_W)

          xBC = F.adaptive_max_pool2d(xBC, pool_size)

          dt = dt.view(batch, H, W, -1).permute(0,3,1,2).contiguous()
          dt = F.adaptive_max_pool2d(dt, pool_size)
          dt = dt.permute(0,2,3,1).view(batch, pool_H * pool_W, -1) # [B, hw, C]

        else :
          pool_H, pool_W = H, W

        st()
        
        xBC = self.act(self.conv2d(xBC))
        xBC = xBC.permute(0, 2, 3, 1).view(batch, pool_H * pool_W, -1).contiguous() # [b, hw, c]
        # print('4) after conv, xBC : ', xBC.shape)

        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.d_inner,
                                    self.ngroups * self.state_dim,
                                    self.ngroups * self.state_dim], dim=-1)
        x, dt, A, B, C = to_ttensor(x, dt, A, B, C)
        # print('5) x b and c : ', x.shape, B.shape, C.shape)

        #############################################
        # 3. Execute Non-Causal SSM, and Upsample
        #############################################
        # st()
        y = self.non_causal_linear_attn(
            rearrange(x, "b l (h p) -> b l h p", p=self.head_dim),
            dt, A, B, C, self.D, pool_H, pool_W
        )

        y = rearrange(y, "b l h p -> b l (h p)")
        y = F.interpolate(y.permute(0,2,1).view(batch, -1, pool_H, pool_W).contiguous(),
                          size=(H, W), mode='nearest')  #[B C H W]
        y = y.view(batch, chan, H*W).permute(0, 2, 1).contiguous()  # [B HW C]

        ######################################################
        # 3. Gate branch를 곱하고, extra normalization 실행
        ######################################################
        # y = self.norm(y, z)
        y = self.norm(y)
        y = y*z
        out = self.out_proj(y)
        out = out.view(batch, H, W, chan).permute(0, 3, 1, 2).contiguous()
        return out
    
    


##############################################################
## EfficientViMBlock
class BlqMamba(nn.Module):
    def __init__(self,
                 dim,
                 state_dim=15,
                 chan_state_dim=1,
                 ssd_expand=1,
                 n_levels=1,
                 drop_path=0.1,
                 nheads=4,
                 coef=[1,1,1,1],
                 ):

        super().__init__()

        # variants
        self.dim = dim
        self.n_levels = n_levels
        self.state_dim = state_dim
        self.chan_state_dim = chan_state_dim
        self.ssd_expand = ssd_expand
        self.drop_path=drop_path
        self.nheads = nheads

        # branch dim
        branch_coef = coef  # channel per branch
        coef_sum = sum(coef)  # full coef sum
        one_coef = dim // coef_sum  # channel per one coef
        self.branch_dim = [coef[i]*one_coef for i in range(self.n_levels)]

        # Mamba Branch
        self.mmb = nn.ModuleList([
            BlqSSM(self.branch_dim[i],
                           state_dim=state_dim,
                           ssd_expand=ssd_expand,
                           drop_path=drop_path,
                           downsize_idx=i,
                           type='spatial',
                           nheads=self.nheads,
                           n_levels=self.n_levels,)
            for i in range(self.n_levels)])

        ##################################################
        # Feature Aggregation Methods
        # 1) Conv 1x1
        # self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # 2) Channel-wise SSM
        # self.chan_mmb = BlqSSM(1,
        #                        state_dim=chan_state_dim,
        #                        ssd_expand=ssd_expand,
        #                        idx=0,
        #                        type='channel')
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        ##################################################

        self.gate = nn.Linear(dim, self.n_levels-1)
        
        # Activation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
      batch, chan, H, W = x.shape
      out = []

      ######################################
      # 1. VSSD architecture per feature
      ######################################
      xc = torch.split(x, self.branch_dim, dim=1)
      
      value = x.permute(0,2,3,1)    # [B H W C]
      value = self.gate(value).permute(0,2,3,1).contiguous()

      for i in range(self.n_levels):
        # process vssd and insert original shape into output
        branch = self.mmb[i](xc[i])
        
        if i < self.n_levels - 1:
            branch = brnach * value[:, i:i+1]
            
        out.append(branch)

      ######################################
      # 2. Channel-Mixing
      ######################################
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
                 state_dim=15,
                 chan_state_dim=1,
                 ssd_expand=1,
                 drop_path=0.,
                 nheads=4,
                 n_levels=1,  # branch 개수
                 coef=[1,1,1,1], # 각 branch의 비율
                 ):

        super().__init__()

        # norm
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # Multiscale Block
        self.blqm = BlqMamba(dim,
                             state_dim=state_dim,
                             chan_state_dim=chan_state_dim,
                             ssd_expand=ssd_expand,
                             drop_path=drop_path,
                             nheads=nheads,
                             n_levels=n_levels,
                             coef=coef,
                             )

        # Feedforward layer
        self.convffn = PCFN(dim,
                       growth_rate = ffn_scale,
                       )

        # PE
        # self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x):
        # batch, chan, height, width = x.shape
        # x = x + self.cpe1(x)
        # shortcut = x

        # first
        # x = self.norm1(x)
        # x = self.blqm(x)
        # x = shortcut + self.drop_path(x)

        # second
        # x = x + self.cpe2(x)
        # x = x + self.drop_path(self.convffn(self.norm2(x)))

        # legacy code ---------------------------
        x = self.blqm(self.norm1(x)) + x
        x = self.convffn(self.norm2(x)) + x
        # ---------------------------------------

        return x
    

##############################################################
## Overall Architecture
# @ARCH_REGISTRY.register()
class LMLT(nn.Module):
    def __init__(self,
                 dim,
                 in_chans=3,
                 n_blocks=8,
                 ffn_scale=2.0,
                 ssd_expand=1,
                 state_dim=15,
                 chan_state_dim=1,
                 upscale=4,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 nheads=4,
                 n_level=1,
                 coef=[1,1,1,1],
    ):


        f"""
        원본 : in_dim : 전체적인 dim(60, 84...)
        out_dim : 다음 stage에서의 dim. 여기에서는 in_dim과 같게. --> 두개 다 dim으로 퉁치기.
        depths : 각 stage 별 block 개수. 일단 stage=4, block=8로 설정 --> n_blocks
        mlp_ratio : 2 --> ffn_scale
        ssd_expand : 1
        state_dim : B, C, dt의 dim.
        + channel-wise state_dim : 4개의 branch의 mixing 용 mamba의 state_dim.
        + upscaling_factor : (2, 3, 4)

        dim=60 -> state_dim=15(1/4) / chan_state_dim=1
        """

        super().__init__()

        # variants
        self.dim = dim
        self.ffn_scale = ffn_scale
        self.state_dim = state_dim
        self.chan_state_dim = chan_state_dim

        # stochastic depth
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]

        # Architecture
        self.to_feat = nn.Conv2d(in_chans, dim, 3, 1, 1)

        self.feats = nn.Sequential(*[BasicBlock(dim,
                                                ffn_scale,
                                                state_dim=state_dim,
                                                chan_state_dim=chan_state_dim,
                                                ssd_expand=ssd_expand,
                                                # drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                                                drop_path = dpr[i],
                                                nheads=nheads,
                                                n_levels=n_level,
                                                coef=coef,
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

    model = LMLT(dim=84,
                    n_blocks=12,
                    state_dim = 64,
                    ffn_scale=2.0,
                    upscale=2,
                    nheads=1,
                    n_level=4,
                    coef=[1,2,3,6]) # [1,1,1,1] / [1,2,3,6]
    # model = LMLT(dim=36, n_blocks=12, ffn_scale=2.0, upscaling_factor=2)
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)
    
    