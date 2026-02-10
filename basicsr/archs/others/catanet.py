import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from inspect import isfunction
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import trunc_normal_
import math

from pdb import set_trace as st

f"""
CATANet: 
Efficient Content-Aware Token Aggregation for Lightweight Image Super-Resolution
"""

f"""
TAB의 IASA와 LRSA의 local_attn의 차이 :
    TAB은 scatter_add, gather 함수 등을 통해, 비슷한 픽셀들끼리 뒤섞어놓은 상태에서 local_attn을 진행
    이후 scatter(idx=idx_last)를 통해 다시 원래대로 돌려놓음.
    
    이후 LRSA에서는, 원래 순서대로 복원된 feat을 상대로 local_attn을 진행. 
"""


f"""
scatter :
    out = scatter(dim=-2, index=idx_last.expand(out.shape), src=out) 일 때,
    [[[544, 544, ... 544],
    [1379, 1379, .. 1379],
    ....
    [2761, 2761, .. 2761]]] 
    이렇게 되어 있다면, 패치방향(세로방향)으로 인덱스를 보며 값을 옮김.
    패치방향으로 쭉 가면 544, 1379, 2789... 이렇게 되면, 
    out의 첫번째 패치의 내용들(40개의 채널값)을 544번째 패치로 옮김.
    
    [[[544, 87, 597, ..],
    [1379, 441, ..],
    [2789, 594, ..]]]
    이렇게 되어 있다면, 세로 방향으로 내려오면 
    src의 [0:3, 0]은 각각 x의 [544,0], [1379,0], [2789,0]로 옮겨지고
    [0:3, 1]은 [87,1], [441,1], [594,1]로 옮겨지게 됨.
    
    dim=-1일 경우에는 채널방향(가로방향)으로 인덱스를 보면서,
    src의 [0, 0:3]을 out의 [0, 544], [0, 87], [0, 597] ...
    src의 [1, 0:3]을 out의 [1,1379], [1,594], ... 이렇게 옮기게 됨.
"""

def exists(val):
    return val is not None

def is_empty(t):
    return t.nelement() == 0

def expand_dim(t, dim, k):
    f"""
    t : [B, N] = [1, 7500] (buckets=각 패치가 어떤 덩어리와 유사한지 그 idx)
    dim : -1
    k : D = 40
    
    output : [B, N, D] = [1, 7500, 40]
             [B N 1]의 1을 D(=40)만큼 복사
             즉 t[0,0]이 3이었다면, t[0,0,:]은 [3 3 3 ... 3(총 D개)]가 됨
             t[0, 7499]가 15였다면, t[0, 7499,:]은 [15 15 15 ... 15(총 D개)]가 됨
             bucket id(=덩어리 id)를 채널 40개로 복제한 것이 됨.
             
    목표 : scater_add 연산을 수행하기 위함.
    """
    t = t.unsqueeze(dim)    # [B N 1]
    expand_shape = [-1] * len(t.shape)  # [-1, -1, -1]
    expand_shape[dim] = k   # [-1, -1, 40(=k=D)]
    return t.expand(*expand_shape)

def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x

def ema(old, new, decay):
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)

def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha= (1 - decay))
    
    
def similarity(x, means):
    return torch.einsum('bld,cd->blc', x, means)

def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets

def batched_bincount(index, num_classes, dim=-1):
    # index: [B, N]. 각 패치가 몇 번째 bucket과 가장 유사한지 그 idx가 있음.
    # num_classes: 덩어리 수(16, 32, 64, 128, 16 ...)
    # shape : [1, 16(=num_classes)](type=list)
    # out : [1, 16(=버켓 수)]짜리 0으로 채워진 텐서를 만든 다음,
    # index에 해당하는 위치에 1(torch.ones_like[i])을 더함.
        
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    
    f"""
    index가 [[3, 6, 14, ...]]이라면, 
    out의 [0,3] 위치에 torch.ones_like(index)[0]의 값을, 
    [0,6] 위치에 torch.ones_like(index)[1] 값을, 
    [0,14] 위치에 torch.ones_like(index)[2] 값을 추가
    최종적으로 out shape는 [1,16]이며,
    n번째 덩어리가 얼마나 많은 패치(=7500)과 연관되어 있는지를 알려줌
    즉 out 요소들의 총합은 7500이 됨.
    """
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out

def center_iter(x, means, buckets = None):
    f"""
    x : [B, N, D] = [1 7500 40]
    means : [num_tokens, D] = [16 40]
    
    scatter_add example:
        x : [[[1,1],
              [2,2]]]
              
      idx : [[[0,0],
              [0,1]]]
                
    --> means : [[[3,1],
                  [0,2]]]
                  
      idx : [[[0,0],
              [1,1]]]
    
    --> means : [[[1,1],
                  [2,2]]]
    """
    
    b, l, d, dtype, num_tokens = *x.shape, x.dtype, means.shape[0]
    
    ##########################################################
    # dist and buckets : 각 패치가 어떤 덩어리와 유사한지 idx 찾기
    # x와 means 사이의 유사도를 @를 통해 구한 후, 가장 큰 값의 index를 buckets로 반환
    # buckets : [B N] = [1 7500] (하나의 패치가 어떤 means와 유사한지, 즉 [0, 15] 사이의 idx)
    ##########################################################
    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    ################################################################
    # batched_bincount : 한 덩어리가 몇개의 패치와 연관되어 있는지 계산
    # bins : [B num_tokens] = [1 16] (총합은 패치(=7500)와 같음)
    # zero_mask : [B num_tokens] = bucket이 0, 즉 어떤 패치와도 닮지 않은 덩어리의 위치를 True로 표시
    ################################################################
    bins = batched_bincount(buckets, num_tokens).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    ###############################################################
    # means_ : 덩어리별로 묶인 패치들의 평균을 구함
    # means_ : [b num_tokens d] = [1 16 40](0으로만 이뤄짐)
    # 이후, expand_dim(buckets, -1, d)를 통해 만들어진 [B N D](=[1 7500 40]) 텐서와 x(=[1 7500 40])
    # 을 scatter_add_ 연산 수행.
    # scatter_add : idx와 같은 위치에 있는 src 값을, 현 idx의 element와 같은 means_열의 idx에 해당하는 값의 행에 넣음.
    #               즉 idx와 src의 shape가 같아야 하며, idx의 값은 [0, 행(means_)] 사이여야 함.
    #               means_의 dim은 x나 idx와 같거나 커야 함.
    #               예를 들어 idx[0,4,0]이 1이고 src[0,4,0]이 3이라면, means_[0,1,0]에 3이 더해지고,
    #               idx[0,4,1]이 2이고 src[0,4,1]이 4라면, means_[0,2,1]에 4가 더해짐.
    # 따라서 means_([1 16 40])은 각 덩어리별로 묶인 패치들의 합이 됨(각 버킷에 할당된 토큰들의 feature 합(sum)).
    # idx를 통해 x의 7500개 패치는 자신과 닮은 덩어리의 위치([0, 15])에 해당하는 means_에 더해짐.
    # 최종적으로 총 7500개가 자신과 가장 닮은 버킷 16개에 분산되며 골고루 더해짐.
    # D(=40)은 x의 채널 수와 같고, x의 패치값을 온전히 옮기기 위한 사이즈.
    ###############################################################
    means_ = buckets.new_zeros(b, num_tokens, d, dtype=dtype)
    means_.scatter_add_(-2, index=expand_dim(buckets, -1, d), src=x)

    ###############################################################
    # norm: sum을 통해 batch차원을 없애고, 전체 배치기준으로 center를 하나로 합침
    #       (dim=-1) 정규화를 통해, 하나의 덩어리로 묶인 수많은 패치들의 합을 1로 만듦
    # where(condition, x, y): condition이 True인 위치에선 x, False인 위치에선 y를 선택
    # means : T, 즉 이번 패치가 어떤 버킷에도 할당되지 않은 경우, 기존 means를 유지
    #         F, 즉 하나 이상의 패치가 할당된 버킷의 경우, means_를 선택
    #         means - 유사도 기준으로 묶인 버킷들
    #         means_ - 이 버킷들과 유사한 패치들을 한데 모아 누적합한 것
    ###############################################################
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means
    
#######################################################################
## TAB(IASA + IRCA)
class IASA(nn.Module):
    def __init__(self, dim, qk_dim, heads, group_size):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.group_size = group_size
        
    
    def forward(self, normed_x, idx_last, k_global, v_global):
        f"""
        normed_x = Input x(단 한번도 변한 적 없음). [B, 7500, 40]
        idx_last = 뒤섞여진 x들의 첫 위치를 삼고 있음. [B, 7500, 1]
                   예) idx_last[532] = 2178일 때, 531번째 패치는 뒤섞이기전에는 2177번째 패치였음
        k_global, v_global = Linear(x_means). [head, 16(버킷), 9] / [head, 16, 10]
                             k는 linear(40, qk_dim=36)으로 설정되어, (head=4, dim//head=9)이며,
                             v는 linear(40, 40)이므로 dim//head=10
        """
        x = normed_x
        B, N, _ = x.shape
        
        ###########################################
        # gather: qkv를 각각 연관된 애들끼리 묶일 수 있도록 뒤섞음.
        # expand -> [B, 7500, 1]이 dim(=40)만큼 복사됨. 
        # 즉 [B 7500 40], equal(idx[batch,:,0], idx[batch,:,15])=T
        # expand = [[[1351,1351,...],[6420,6420,...], .. [531,531,...(36개)](7500개)] 
        
        # gather의 작동원리 : (dim=-2)이므로, 행을 선택
        # 0번 채널의 idx(행)가 1351 -> 1351줄의 0번째 채널을 갖고옴. new_q[0,0,0] = q[0,1351,0]
        # 1번 채널의 idx(행)가 1351 -> 1351줄의 1번재 채널을 갖고옴. new_q[0,0,1] = q[0,1351,1]
        # ... 마지막 행 역시 1351 -> 1351줄의 35번째 채널을 갖고옴. new_q[0,0,35] = q[0,1351,35]
        # 즉, src의 idx번째(1351) 열의 행(채널)을 통째로 옮김. 패치들을 통째로, 벡터단위로 순서 옮기는 과정.
        #####################################
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)  # q,k,v = [B, 7500, 36/36/40]
        q = torch.gather(q, dim=-2, index=idx_last.expand(q.shape))
        k = torch.gather(k, dim=-2, index=idx_last.expand(k.shape))
        v = torch.gather(v, dim=-2, index=idx_last.expand(v.shape))

        # N=패치 수 / group_size = [256, 128, 64, 32, 256, 128, 64, 32] 중 작은 걸 택함.
        # 패치 수 또는 그룹사이즈에 비례하게 패딩할 수 있는 수치
        gs = min(N, self.group_size)  # group size
        ng = (N + gs - 1) // gs
        pad_n = ng * gs - N
        
        #####################################
        # 1. local 정보의 attention
        
        # paded_q : ng(그룹 개수)로 잘린 q([b 패치 dim] -> [b 그룹수 head 그룹패치수 dim//head])
        # 현재 패치들은 연관된 애들끼리 묶여있지만, 하나의 버킷과 관련된 패치들은 각각 수가 다름. 
        # Fig3(b)의 서브그루핑이 여기서 나옴. (ng=256, 128, ..)으로, 서브그루핑해서, 수가 갖게 함.
        # 0번 버킷과 관련된 패치가 [0,225], 1번 버킷과 관련된 패치가 [226, 356]이었다면,
        # 이 그루핑을 통해 [0, 255], [256, 512] .. 등을 통해 서로 관련 없는 버킷과 닮은 패치들끼리 강제 그루핑
        
        # paded_k, v : 위 과정을 통해 강제 그루핑 되었으니, 이렇게 나뉜 패치가 자신의 원래 영역(이전것 또는 다음것)도
        # 볼 수 있도록, 추가로 패딩(gs)하고, unfold를 통해 중복 영역을 추가하게 됨.
        # gs(그룹 사이즈)만큼 추가로 패딩하고, 패치 단위로 unfold.
        # unfold : (dim=-2) 방향으로 size(2*gs)만큼 자르기. stride는 gs.
        # 즉, [b 패치 36(40)]에서, 패치를 2*gs(=512)만큼 가져오고, 그 사이즈는 [36(40), 512].
        # 세로방향으로 512개씩 총 30(by stride)번 가져오므로, 그 사이즈는 [B 30 36(40) 512]가 됨.
        # 자동으로 Transpose 효과까지 가짐. 
        
        # 이후 (q@k)@V 실행.
        # output : [B, 30, 4, 256, 10] = [배치, num_grp, head, grp_size, head_dim]]
        #####################################
        paded_q = torch.cat((q, torch.flip(q[:,N-pad_n:N, :], dims=[-2])), dim=-2)
        paded_q = rearrange(paded_q, "b (ng gs) (h d) -> b ng h gs d",ng=ng,h=self.heads)

        paded_k = torch.cat((k, torch.flip(k[:,N-pad_n-gs:N, :], dims=[-2])), dim=-2)
        paded_k = paded_k.unfold(-2,2*gs,gs)
        paded_k = rearrange(paded_k, "b ng (h d) gs -> b ng h gs d",h=self.heads)
        
        paded_v = torch.cat((v, torch.flip(v[:,N-pad_n-gs:N, :], dims=[-2])), dim=-2)
        paded_v = paded_v.unfold(-2,2*gs,gs)
        paded_v = rearrange(paded_v, "b ng (h d) gs -> b ng h gs d",h=self.heads)
        
        out1 = F.scaled_dot_product_attention(paded_q,paded_k,paded_v)
        
        #######################################
        # 2. global 정보의 attn 계산
        
        # k/v_global : [head 버킷 head_dim] -> [1 1 head 버킷 head_dim] -> [B num_grp(30) head 버킷 head_dim]
        # [head 버킷 head_dim]이 총 B*30번 반복됨.
        
        # (q @ global_k) @ global_v
        # q는 [B, num_win(30), head, win_size(256), head_dim(9)]
        # global_k는 [head 버킷 head_dim]을 B*30번 반복. 
        # q@k     = [B, num_win, head, win_size, 버킷]
        # (q@k)@v = [B, num_win, head, win_size, head_dim]
        # 즉, q@k = "하나의 윈도우가, 모든 버킷과 attn을 진행" 버킷 하나는 이미 global한 정보를 담고 있음.
        
        # local attn 정보와 global attn 정보를 더함.
        #######################################
        k_global = k_global.reshape(1,1,*k_global.shape).expand(B,ng,-1,-1,-1)
        v_global = v_global.reshape(1,1,*v_global.shape).expand(B,ng,-1,-1,-1)
        out2 = F.scaled_dot_product_attention(paded_q,k_global,v_global)
        out = out1 + out2
        
        ######################################
        # out -> [B, num_win*win_size, head*head_dim][:, :N, :] (패딩했던 부분은 제거. N=7500)
        # = [B, 7500, 40]
        # scatter -> idx_last를 활용, out의 뒤섞여있던 패치들을 dim=-2(패치) 즉 세로방향의 원래 위치로 되돌림.
        # 이후 proj_out
        ######################################
        out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")[:, :N, :]
        out = out.scatter(dim=-2, index=idx_last.expand(out.shape), src=out)
        out = self.proj(out)
    
        return out

class IRCA(nn.Module):
    f"""
    objective : Local feature가, global한 정보들 중 필요한 것을 골라가게 하기 위해 k(v)_global을 만듦.
                나중에 Q(local info)와 MM을 해서, (q@k)@v = ([N, token])@[token, d] = [N,d]를 통해,
                패치와 토큰들 간의 연관성 파악.
    """
    def __init__(self, dim, qk_dim, heads):
        super().__init__()
        self.heads = heads
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
      
    def forward(self, normed_x, x_means):
        x = normed_x
        
        # global pattern 확보를 위해, 추가로 한번 더 center_iter 진행
        if self.training:
            x_global = center_iter(F.normalize(x,dim=-1), F.normalize(x_means,dim=-1))
        else:
            x_global = x_means

        # x_global = [16 40] = [버킷 채널]
        # 이를 rearrange를 통해 [head 버킷 채널//head]로 변경
        k, v = self.to_k(x_global), self.to_v(x_global)
        k = rearrange(k, 'n (h dim_head)->h n dim_head', h=self.heads)
        v = rearrange(v, 'n (h dim_head)->h n dim_head', h=self.heads)

        return k,v, x_global.detach()
    

class TAB(nn.Module):
    def __init__(self, dim, qk_dim, mlp_dim, heads, n_iter=3,
                 num_tokens=8, group_size=128,
                 ema_decay = 0.999):
        super().__init__()
        f"""
        x -- (center_iter *5) -- IASA - + - Conv -+- LN - MLP
           |                  |- ICRA --|         |
           | -------------------------------------|
        """
        self.n_iter = n_iter
        self.ema_decay = ema_decay
        self.num_tokens = num_tokens
        
        
        self.norm = nn.LayerNorm(dim)
        self.mlp = PreNorm(dim, ConvFFN(dim,mlp_dim))
        self.irca_attn = IRCA(dim,qk_dim,heads)
        self.iasa_attn = IASA(dim,qk_dim,heads,group_size)
        self.register_buffer('means', torch.randn(num_tokens, dim))
        self.register_buffer('initted', torch.tensor(False))
        self.conv1x1 = nn.Conv2d(dim,dim,1, bias=False)

    
    def forward(self, x):
        _,_,h, w = x.shape
        x = rearrange(x, 'b c h w->b (h w) c')
        residual = x
        x = self.norm(x)
        B, N, _ = x.shape
        
        f"""
        idx_last : [B, N](각 토큰의 idx를 [0,N-1]사이로 부여)
        paded_x : [B, N+pad_n, C](두 번째 dim이 num_tokens의 배수가 되도록, 뒤에서부터 거꾸로 반사패딩)
                  인풋이 [B 100 75]이고 num_toknes=16일 때, pad_n=4가 되어 [B 100*75+4 C]가 됨
        x_means : [num_tokens, C](토큰 덩어리) = [16, 40]
                  paded_x를 num_tokens개만큼 묶어서 평균을 구함. 즉, 7504개 패치를 469개씩 묶어 16개 덩어리로 만듦.
                  token은 [16 32 64 128 16 32 64 128]로, 묶이는 패치는 갈수록 적어짐(더 local해짐)
        """
        idx_last = torch.arange(N, device=x.device).reshape(1,N).expand(B,-1)
        if not self.initted:
            pad_n = self.num_tokens - N % self.num_tokens
            paded_x = torch.cat((x, torch.flip(x[:,N-pad_n:N, :], dims=[-2])), dim=-2)
            x_means=torch.mean(rearrange(paded_x, 'b (cnt n) c->cnt (b n) c',cnt=self.num_tokens),
                               dim=-2).detach()   
        else:  
            x_means = self.means.detach()

        f"""
        n_iter : [5,5,...5]
        
        여러번 iteration을 하는 이유 : 
            matrix multiplication 연산과 softmax를 통해, 각 means는 N개의 패치와 얼마나 관련이 있는지, 그 확률을 획득
            softmax 연산을 통해 획득한 확률이 확실할 경우, 그 값은 e^x를 통해 매우 커지기 때문에, 즉시 유사한 패치 사이에 낌
            그러나 어중간할 경우에는, 1등 패치모음(토큰)과 2등 토큰 사이에 어정쩡하게 낌.
            여러번 연산함으로써 하나의 토큰에 확실히 끼게 됨.
    
        x_means : 인접한 패치들의 묶음, 또는 그 묶음과 유사한 패치들끼리 다시 묶은 버킷들, [16 40]
                  배치단위까지 함께 묶었기에 [버킷개수 패치채널] 사이즈가 나옴.
                  n_iter 횟수만큼 반복해서 버킷을 k-nn 정제.
        """
        if self.training:
            with torch.no_grad():
                for _ in range(self.n_iter-1):
                    # x(B, N, C), x_means(num_tokens, C)를 C에서 정규화한 후, center_iter 수행
                    x_means = center_iter(F.normalize(x,dim=-1), F.normalize(x_means,dim=-1))
                        
        
        ###################################
        # IRCA
        
        # 1) icra_attn
        # x_means = 한번더 center_iter 한 x_means
        # k, v_global = Linear(x_means)
        
        # 2) einsum : x@x_means : [B 7500 16](b, num_patch, num_bucket)
        # argmax: 최대값의 인덱스 반환(값을 반환하는 max와는 다름)
        # x_belong_idx: 각 패치 별로 자신과 가장 유사한 버킷의 인덱스를 반환
        # 예- [7, 4, 0, ... 15, 14, 3] -> 0번쨰 패치는 7번버킷, ... 7500번째 패치는 3번버킷과 유사
        
        # 3) argsort : 값을 정렬했을 때의 idx 순서를 반환. 같은 버킷에 속하는 패치(x)끼리 모이도록 만듦.
        # 자신과 유사한 패치들끼리 묶이게 됨(같은 버킷을 가장 관련성 큰 버킷으로 삼는애들끼리 모임)
        # 예- [3503, 3443, 1628, .. 7196, 1462, 1388] -> 3503, 3443, 1638...번쨰 패치는 0번버킷,
        # ... 7196, 1462, 1388번째 패치는 15번 버킷과 유사
        # stable=True로 하면 오름차순을 보장하지만 default는 랜덤임.
        
        # 4) gather : 위치 추적.
        # 앞서 argsort를 통해, 패치를 내용끼리 섞었음. 
        # index에 해당하는 값을 idx_last(=arange(N))에서 추출.
        # index는 서로 유사한 패치들끼리 인접시켰을때 그 인덱스들이며, 
        # 후에 이 x를 그 인덱스 순서로 뒤섞음(IASA 내에서)
        # 그 x를 다시 원래 순서대로 되돌리기 위해 idx_last를 사용.
        # gather 함수 : input 내의 element들을 index 순서대로 추출함.
        ###################################
        k_global, v_global, x_means = self.irca_attn(x, x_means)
        
        with torch.no_grad():
            x_scores = torch.einsum('b i c,j c->b i j', 
                                        F.normalize(x, dim=-1), 
                                        F.normalize(x_means, dim=-1))
            x_belong_idx = torch.argmax(x_scores, dim=-1)

            idx = torch.argsort(x_belong_idx, dim=-1)
            idx_last = torch.gather(idx_last, dim=-1, index=idx).unsqueeze(-1)
        
        ##################################
        # IASA + token center update
        # y = [b 7500 40] - [b h w c] - conv@11 - (+x) - MLP
        
        # 첫 실행일 경우(not self.initted) 기존의 전역토큰(self.means)을 그냥 사용.
        # 아닐 경우, 기존 센터(self.means)에 새로운 정보(new_means)를 살짝 섞음(self.ema_decay만큼).
        ##################################
        y = self.iasa_attn(x, idx_last,k_global,v_global)
        y = rearrange(y,'b (h w) c->b c h w',h=h).contiguous()
        y = self.conv1x1(y)
        x = residual + rearrange(y, 'b c h w->b (h w) c')
        x = self.mlp(x, x_size=(h, w)) + x
        
        if self.training:
            with torch.no_grad():
                new_means = x_means
                if not self.initted:
                    self.means.data.copy_(new_means)
                    self.initted.data.copy_(torch.tensor(True))
                else: 
                    ema_inplace(self.means, new_means, self.ema_decay)
            
    
        return rearrange(x, 'b (h w) c->b c h w',h=h)
        
        
        
########################################################################
## LRSA(WindowAttn + ConvFFN)
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


class PreNorm(nn.Module):
    """Normalization layer.
    Args:
        dim (int): Base channels.
        fn (Module): Module after normalization.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
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

    def __init__(self, dim, qk_dim, mlp_dim,heads=1):
        super().__init__()
     

        self.layer = nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, qk_dim)),
                PreNorm(dim, ConvFFN(dim, mlp_dim))])

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

        attn, ff = self.layer
        crop_x = attn(crop_x) + crop_x
        crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)
        
        ################################
        # Patch Reverse - LN - MLP(ConvFFN)
        
        # patch_reverse input : crop_x, x(첫 input), step(=patch_size - 2), ps)
        ################################
        x = patch_reverse(crop_x, x, step, ps)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = ff(x, x_size=(h, w)) + x
        x = rearrange(x, 'b (h w) c->b c h w', h=h)
        
        return x


##########################################################################
## Main 
# @ARCH_REGISTRY.register()
class CATANet(nn.Module):
    setting = dict(dim=40, block_num=8, qk_dim=36, mlp_dim=96, heads=4, 
                     patch_size=[16, 20, 24, 28, 16, 20, 24, 28])

    def __init__(self,
                 in_chans=3,
                 n_iters=[5,5,5,5,5,5,5,5],
                 num_tokens=[16,32,64,128,16,32,64,128],
                 group_size=[256,128,64,32,256,128,64,32],
                 upscale: int = 4):
        super().__init__()
        
    
        self.dim = self.setting['dim']
        self.block_num = self.setting['block_num']
        self.patch_size = self.setting['patch_size']
        self.qk_dim = self.setting['qk_dim']
        self.mlp_dim = self.setting['mlp_dim']
        self.upscale = upscale
        self.heads = self.setting['heads']
        
        


        self.n_iters = n_iters
        self.num_tokens = num_tokens
        self.group_size = group_size
    
        #-----------1 shallow--------------
        self.first_conv = nn.Conv2d(in_chans, self.dim, 3, 1, 1)

        #----------2 deep--------------
        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
   
        for i in range(self.block_num):
          
            self.blocks.append(nn.ModuleList([TAB(self.dim, 
                                                  self.qk_dim, 
                                                  self.mlp_dim,
                                                  self.heads, 
                                                  self.n_iters[i], 
                                                  self.num_tokens[i],
                                                  self.group_size[i]), 
                                            
                                              LRSA(self.dim, 
                                                   self.qk_dim,
                                                   self.mlp_dim,
                                                   self.heads)
                                              ]))
            
            self.mid_convs.append(nn.Conv2d(self.dim, self.dim,3,1,1))
            
        #----------3 reconstruction---------
        
        if upscale == 4:
            self.upconv1 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
            
        elif upscale == 2 or upscale == 3:
            self.upconv = nn.Conv2d(self.dim, self.dim * (upscale ** 2), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale)
    
        self.last_conv = nn.Conv2d(self.dim, in_chans, 3, 1, 1)
        if upscale != 1:
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

    def forward_features(self, x):
        f"""
        deep feature extraction(8 blocks)
        
        x -- TAB(global_attn) -- LRSA(local_attn) -- Conv(mid_conv) - + - output
           |----------------------------------------------------------|
        """
        for i in range(self.block_num):
            residual = x
            global_attn, local_attn = self.blocks[i]
            x = global_attn(x)
            x = local_attn(x, self.patch_size[i])
            x = residual + self.mid_convs[i](x)
        return x
        
    def forward(self, x):

        if self.upscale != 1: 
            base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        else: 
            base = x
            
        ## shallow feature extraction
        x = self.first_conv(x)
        
        ## deep feature extraction
        x = self.forward_features(x) + x
    
        ## reconstruction
        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 1:
            out = x
        else:
            out = self.lrelu(self.pixel_shuffle(self.upconv(x)))
        out = self.last_conv(out) + base
        return out
    
    
    # def __repr__(self):
    #     num_parameters = sum(map(lambda x: x.numel(), self.parameters()))
    #     return '#Params of {}: {:<.4f} [K]'.format(self._get_name(),
    #                                                   num_parameters / 10 ** 3) 
  
  


if __name__ == '__main__':
    model = CATANet(upscale=2)
    x = torch.randn(1, 3, 640, 360)
    y = model(x)
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print('output: ', y.shape)