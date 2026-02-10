def window_group_partition_overlap(self, x):
        B, C, H, W = x.shape
        ws, gs = self.window_size, self.group_size
        K = ws * gs      # 그룹 커널 크기 (예: 72)
        S = K - 2        # S = overlap = 겹치는 영역이 2가 되도록 설정

        ########################
        # Padding: 모든 픽셀을 커버하면서 stride만큼 이동 가능하도록
        # (H + pad - K) % S == 0 을 만족해야 함
        ########################
        pad_h = (S - (H - K) % S) % S if H > K else K - H
        pad_w = (S - (W - K) % S) % S if W > K else K - W

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        H_pad, W_pad = x.shape[2:]

        #############################
        # 2. Partition(Unfold:) [B, C*K*K, L]
        #############################
        x_unfold = F.unfold(x, kernel_size=(K, K), stride=(S, S))
        
        ########################################
        # 3. Reshape: [B, L, GS*GS, WS*WS, C], grouping
        ########################################
        L = x_unfold.shape[-1]
        x = rearrange(x_unfold, 'b (c gh wh gw ww) l -> b l (gh gw) (wh ww) c', 
                    c=C, gh=gs, wh=ws, gw=gs, ww=ws)
        
        return x, pad_h, pad_w
    
    
def window_group_reverse_overlap(self, x, original_shape, padded_size):
        B, C, H, W = original_shape
        ws, gs = self.window_size, self.group_size
        K = ws * gs
        S = K - 2  # 동일한 stride 적용
        H_pad, W_pad = H + padded_size[0], W + padded_size[1]

        ########################################
        # 1. Fold를 위한 Flatten: [B, C*K*K, L]
        ########################################
        x = rearrange(x, 'b l (gh gw) (wh ww) c -> b (c gh wh gw ww) l',
                    gh=gs, gw=gs, wh=ws, ww=ws)

        #############################################
        # 2. Summation (오버래핑 2픽셀 영역은 값이 더해짐)
        #############################################
        x_sum = F.fold(x, output_size=(H_pad, W_pad), kernel_size=(K, K), stride=(S, S))

        ###########################################
        # 3. Normalization Mask (몇 번 겹쳤는지 계산)
        ###########################################
        ones = torch.ones_like(x)
        counts = F.fold(ones, output_size=(H_pad, W_pad), kernel_size=(K, K), stride=(S, S))

        ################################################
        # 4. Average & Crop
        # counts는 대부분 1이지만, 겹치는 2픽셀 라인은 2(혹은 모서리에서 4)의 값을 가짐.
        ################################################
        x_avg = x_sum / counts
        
        if padded_size[0] > 0 or padded_size[1] > 0:
            x_avg = x_avg[:, :, :H, :W]
            
        return x_avg