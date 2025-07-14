import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *

from utils import AverageMeter, write_img, chw_to_hwc
from utils import calculate_metrics
from datasets.loader import PairLoader
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='', type=str, help='model name')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
parser.add_argument('--data_dir', default='', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='', type=str, help='path to results saving')
parser.add_argument('--dataset', default='', type=str, help='dataset name')
parser.add_argument('--exp', default='', type=str, help='experiment setting')
args = parser.parse_args()


def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict

def test(test_loader, network, result_dir):
    PSNR = AverageMeter()
    SSIM_meter = AverageMeter()
    LPIPS_meter = AverageMeter()

    
    torch.cuda.empty_cache()

    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    for idx, batch in enumerate(test_loader):
        input = batch['source'].cuda()
        target = batch['target'].cuda()

        filename = batch['filename'][0]

        with torch.no_grad():
            output, sau_feat, cau_feat, bot_feat2, bot_feat1 = network(input)
            output = output.clamp_(-1, 1)
			
            psnr_val,ssim_val,lpips_val = calculate_metrics(output, target)

        PSNR.update(psnr_val)
        SSIM_meter.update(ssim_val)
        LPIPS_meter.update(lpips_val)

        print(f'Test: [{idx}]\tPSNR: {PSNR.val:.02f} ({PSNR.avg:.02f})\tSSIM: {SSIM_meter.val:.03f} ({SSIM_meter.avg:.03f})\tLPIPS: {LPIPS_meter.val:.04f} ({LPIPS_meter.avg:.04f})')

        f_result.write(f'{filename},{psnr_val:.02f},{ssim_val:.03f},{lpips_val:.04f}\n')

        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, 'imgs', filename), out_img)

    f_result.close()

    os.rename(os.path.join(result_dir, 'results.csv'),
              os.path.join(result_dir, f'{PSNR.avg:.02f} | {SSIM_meter.avg:.04f} | {LPIPS_meter.avg:.04f}.csv'))



if __name__ == '__main__':
	network = eval(args.model.replace('-', '_'))()
	network.cuda()
	saved_model_dir = os.path.join(args.save_dir, args.exp, args.model+'.pth')

	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing trained model!')
		exit(0)

	print("data_dir:",args.data_dir)
	print("dataset:",args.dataset)

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	print("dataset_dir:",dataset_dir)
	test_dataset = PairLoader(dataset_dir, 'test', 'test')
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)

	result_dir = os.path.join(args.result_dir, args.dataset, args.model)
	test(test_loader, network, result_dir)

