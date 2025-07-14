import torch
import torch.nn as nn
import torch.nn.functional as F



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=12, dropout=0, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, kernel_size=3, padding=dilation, dilation=dilation)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dropout=0, norm_groups=12, dilation=1, kernel_size=3):
        super().__init__()
        padding = dilation if kernel_size == 3 else 0
        self.block1 = Block(dim, dim, groups=norm_groups, dilation=dilation)
        self.block2 = Block(dim, dim, groups=norm_groups, dropout=dropout, dilation=dilation)
        self.res_conv = nn.Identity()

    def forward(self, x):
        return self.block2(self.block1(x)) + self.res_conv(x)


class ARB(nn.Module):
    def __init__(self, channels):
        super(ARB, self).__init__()

        self.activation = Swish()


        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)


        self.rc1 = ResnetBlock(channels, kernel_size=1)


        self.rdc7 = ResnetBlock(channels, dilation=7)


        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)


        self.dc5 = nn.Conv2d(channels, channels, kernel_size=3, padding=5, dilation=5)
        self.dc3 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3)

        self.rc3 = ResnetBlock(channels, kernel_size=3)

    def forward(self, x):
        residual_outer = x  

        out = self.activation(self.conv1(x))     
        residual_inner = out                   

        out = self.rc1(out)              
        out = self.rdc7(out)                
        out = out + residual_inner           

        out = self.activation(self.conv2(out))
        out = self.activation(self.dc5(out))    
        out = self.activation(self.dc3(out))   

        out = out + residual_outer           
        out = self.rc3(out)               

        return self.activation(out)
    

class CAU(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CAU, self).__init__()
        self.dw_conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)  # dilation=1
        self.dw_conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4, groups=channels)  # dilation=4

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        c1 = self.dw_conv1(x)
        c = self.dw_conv2(c1)

        y = self.global_pool(c).view(x.size(0), -1)  
        y = self.relu(self.fc1(y))
        w = self.sigmoid(self.fc2(y))  
        a = c * w.view(x.size(0), -1, 1, 1)  
        return a

class SAU(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SAU, self).__init__()
        inter_channels = channels // reduction
        self.q = nn.Conv2d(channels, inter_channels, kernel_size=1)
        self.k = nn.Conv2d(channels, inter_channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).view(B, -1, H * W).permute(0, 2, 1)  
        k = self.k(x).view(B, -1, H * W)              
        v = self.v(x).view(B, -1, H * W)              

        attn = torch.bmm(q, k) / (k.size(1) ** 0.5)     
        attn = self.softmax(attn)                     

        out = torch.bmm(v, attn.permute(0, 2, 1))      
        out = out.view(B, C, H, W)
        return out


class CAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CAM, self).__init__()
        self.cau = CAU(channels, reduction=reduction)
        self.sau = SAU(channels)

    def forward(self, x):
        cau_feat = self.cau(x)            
        sau_feat = self.sau(cau_feat)  
        return sau_feat, cau_feat
    


class Encoder(nn.Module):
    def __init__(self, channels):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, channels//4, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(channels//4, channels//2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(channels//2, channels, kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feats = []

        x = self.relu(self.conv1(x))
        feats.append(x)

        x = self.relu(self.conv2(x))
        feats.append(x)

        x = self.relu(self.conv3(x))
        feats.append(x)

        return x, feats

    
class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(channels, channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(channels//2, channels//4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(channels//4, 3, kernel_size=3, padding=1)
        )

        self.skip3_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.skip2_conv = nn.Conv2d(channels//2, channels//2, kernel_size=1)
        self.skip1_conv = nn.Conv2d(channels//4, channels//4, kernel_size=1)

    def forward(self, x, feats):
        skip3 = feats.pop()
        x = x + self.skip3_conv(skip3)
        x = self.up1(x)

        skip2 = feats.pop()
        x = x + self.skip2_conv(skip2)
        x = self.up2(x)

        skip1 = feats.pop()
        x = x + self.skip1_conv(skip1)
        x = self.up3(x)

        return x




class TeacherNet(nn.Module):
    def __init__(self, dim):
        # dim = 96
        super(TeacherNet, self).__init__()

        self.patch_size = 4

        self.encoder = Encoder(channels=dim)

        self.attention = CAM(channels=dim)

        self.bottleneck1 = nn.Sequential(
            ARB(channels=dim),
            ARB(channels=dim)
        )
        self.bottleneck2 = nn.Sequential(
            ARB(channels=dim),
            ARB(channels=dim)
        )
        self.decoder = Decoder(channels=dim)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


    def forward(self, x):
        x = self.check_image_size(x)
        x, skips = self.encoder(x)                 


        sau_feat, cau_feat = self.attention(x)            

        bot_feat1 = self.bottleneck1(x) 

        bot_feat2 = self.bottleneck2(bot_feat1)  

        feat = sau_feat * bot_feat2  

        out = self.decoder(feat, skips)           

        return out, sau_feat, cau_feat, bot_feat2, bot_feat1
