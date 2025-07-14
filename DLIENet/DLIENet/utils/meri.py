import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pytorch_msssim import ssim  
import lpips                     
loss_fn_alex = lpips.LPIPS(net='alex')

def rgb_to_ycbcr(img):
    r = img[:, 0:1, :, :]
    g = img[:, 1:2, :, :]
    b = img[:, 2:3, :, :]
    y = 0.257 * r + 0.504 * g + 0.098 * b + 0.0625
    return y

def calculate_metrics(output, target):

    loss_fn_alex.cuda()

    output = output * 0.5 + 0.5
    target = target * 0.5 + 0.5

    output_256 = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)
    target_256 = F.interpolate(target, size=(256, 256), mode='bilinear', align_corners=False)


    output_y = rgb_to_ycbcr(output_256)
    target_y = rgb_to_ycbcr(target_256)


    mse = F.mse_loss(output_y, target_y)
    psnr_val = 10 * torch.log10(1 / mse).item()


    ssim_val = ssim(output_y, target_y, data_range=1, size_average=False).item()

    output_y_3c = output_y.repeat(1, 3, 1, 1)
    target_y_3c = target_y.repeat(1, 3, 1, 1)
    lpips_val = loss_fn_alex(output_y_3c, target_y_3c).item()

    return psnr_val, ssim_val, lpips_val
