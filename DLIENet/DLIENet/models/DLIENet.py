import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super(AdaIN, self).__init__()
        self.eps = eps

    def forward(self, content_feat, style_feat):
        style_mean = style_feat.mean(dim=[2, 3], keepdim=True)
        style_std = style_feat.std(dim=[2, 3], keepdim=True) + self.eps
        content_mean = content_feat.mean(dim=[2, 3], keepdim=True)
        content_std = content_feat.std(dim=[2, 3], keepdim=True) + self.eps

        normalized = (content_feat - content_mean) / content_std
        return normalized * style_std + style_mean

class FFB(nn.Module):
    def __init__(self, channels):
        super(FFB, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

        self.adain = AdaIN()
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(channels, affine=True)

        self.res_conv = nn.Identity()

    def forward(self, x, encoder_feat):
        residual = self.res_conv(x)

        out = self.conv1(x)
        out = self.adain(out, encoder_feat)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm(out)

        out = out + residual
        out = self.relu(out)

        return out

    


class AFU(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm_layer=None):
        super(AFU, self).__init__()
        out_channels = out_channels or in_channels
        norm_layer = norm_layer or (lambda num_channels: nn.InstanceNorm2d(num_channels, affine=True))  # 默认 InstanceNorm2d

        self.afu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.afu(x)

class CAM_AFU(nn.Module):
    def __init__(self, channels):
        super(CAM_AFU, self).__init__()
        norm_fn = lambda c: nn.InstanceNorm2d(c, affine=True)
        self.afu1 = AFU(in_channels=channels, norm_layer=norm_fn)
        self.afu2 = AFU(in_channels=channels, norm_layer=norm_fn)

    def forward(self, x):
        feat1 = self.afu1(x)
        feat2 = self.afu2(feat1)
        return feat2, feat1



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


class StudentNet(nn.Module):
    def __init__(self,dim):
        super(StudentNet, self).__init__()
        self.patch_size = 4
        self.encoder = Encoder(channels=dim)
        self.attention = CAM_AFU(channels=dim)
        self.bottleneck1 = FFB(channels=dim)
        self.bottleneck2 = FFB(channels=dim)
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
        feat2, feat1 = self.attention(x)          
        bot_feat1 = self.bottleneck1(x, x)  
        bot_feat2 = self.bottleneck2(bot_feat1, x)  
        feat = feat2 * bot_feat2  
        out = self.decoder(feat, skips)            
        return out, feat2, feat1, bot_feat2, bot_feat1
