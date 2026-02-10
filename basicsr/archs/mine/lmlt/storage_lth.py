####################################################################################################
        ## low-to-high ##
        self.compress_low = nn.ModuleList([
            nn.Conv2d(self.branch_dim[i], 1, 1, 1, 0, bias=False)
            for i in range(self.n_levels - 1)
        ])  # low branch
        self.compress_high = nn.ModuleList([
            nn.Conv2d(self.branch_dim[i+1], 1, 1, 1, 0, bias=False)
            for i in range(self.n_levels - 1)
        ]) # high branch
        self.alphas = nn.Parameter(torch.full((self.n_levels - 1, 2), 0.5))
        

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
                
                ## make to next branch shape(h*2, w*2) for low-to-high
                low_compress = self.compress_low[i](branch)
                high_compress = self.compress_high[i](downsized_feats[i+1])
                low_compress = F.interpolate(low_compress, size=high_compress.shape[2:], mode='bilinear', align_corners=False)
                lth_map = torch.sigmoid(self.alphas[i, 0] * low_compress + self.alphas[i, 1] * high_compress)
                next_branch = lth_map * downsized_feats[i+1]
                downsized_feats[i+1] = next_branch + downsized_feats[i+1]
            
            else :   # last branch : High-Freq Conv
                branch = self.mfr[i](downsized_feats[i])   
                branch = branch * gate[:, gate_idx:gate_idx+self.branch_dim[i]]   
                out.append(branch)          
####################################################################################################

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
    
####################################################################################################
class Low_to_high(nn.Module):
    def __init__(self,
                 lr_dim,
                 hr_dim,):
        
        super().__init__()
        self.compress = nn.Conv2d(lr_dim, hr_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.zeros(1, hr_dim, 1, 1))
    
    def forward(self, lr_pre, lr_post, hr_pre):
        delta = lr_post - lr_pre
        delta_up = F.interpolate(delta, size=hr_pre.shape[2:], mode='bilinear', align_corners=False)
        attn_map = self.compress(delta_up)
        hr_out = self.alpha * attn_map + hr_pre
        return hr_out
    

    
    ## low-to-high init ##
        self.lth = nn.ModuleList()
        for i in range(self.n_levels - 1):
            # dim: 채널 수 (LR과 HR 채널 수가 같다면 그대로 사용)
            # 파라미터 최소화를 위해 Depthwise Convolution 사용
            self.lth.append(
                Low_to_high(lr_dim=self.branch_dim[i], 
                            hr_dim=self.branch_dim[i+1],
                            )
            )
        
        # forward 
                gate_idx = self.branch_dim[i]
                
                # ------------------- [수정 핵심] -------------------
                downsized_feats[i+1] = self.lth[i](downsized_feats[i], branch, downsized_feats[i+1])
                # ---------------------------------------------------
            
            else :   # last branch : High-Freq Conv
                
####################################################################################################

class Low_to_high(nn.Module):
    def __init__(self,
                 lr_dim,
                 hr_dim,):
        
        super().__init__()
        self.compress = nn.Conv2d(lr_dim, hr_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, lr_pre, lr_post, hr_pre):
        delta = lr_post - lr_pre
        delta_up = F.interpolate(delta, size=hr_pre.shape[2:], mode='bilinear', align_corners=False)
        attn_map = self.sigmoid(self.compress(delta_up))
        hr_out = hr_pre * attn_map + hr_pre
        return hr_out
####################################################################################################

### 제일 효과 높음
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