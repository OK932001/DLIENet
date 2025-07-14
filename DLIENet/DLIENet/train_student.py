import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import AverageMeter
from datasets.loader import PairLoader
from models import *

from utils.CR_res import ContrastLoss_res
import numpy as np

from loss.losses import *
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='', type=str, help='model name')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='', type=str, help='path to logs')
parser.add_argument('--dataset', default='', type=str, help='dataset name')
parser.add_argument('--exp', default='', type=str, help='experiment setting')
parser.add_argument('--gpu', default='', type=str, help='GPUs used for training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict

from PIL import Image
def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(tensor_data, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_numpy = tensor2im(tensor_data)

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)

    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)

    image_pil.save(image_path)

def train(train_loader, network, network_teacher, criterion, optimizer, scaler):
	losses = AverageMeter()

	torch.cuda.empty_cache()
	
	network.train()

	network_teacher.eval()
	for p in network_teacher.parameters():
		p.requires_grad = False


	for batch in train_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()



		with autocast(args.no_autocast):
			output, sau_feat, cau_feat, bot_feat2, bot_feat1  = network(source_img)
			output_teacher, sau_feat_teacher, cau_feat_teacher, bot_feat2_teacher, bot_feat1_teacher  = network_teacher(source_img)



			loss_hard = criterion[0](output, target_img)
			loss_soft = criterion[0](output, output_teacher)
			loss_feat = criterion[0](bot_feat2, bot_feat2_teacher) + criterion[0](bot_feat1, bot_feat1_teacher)
			loss_att = criterion[0](sau_feat, sau_feat_teacher) + criterion[0](cau_feat, cau_feat_teacher)
			#
			loss = loss_feat + loss_att*0.5 + loss_hard*0.03 + loss_soft*0.001

			loss_hard = criterion[0](output, target_img)
			loss_soft = criterion[0](output, output_teacher)
			loss_feat = criterion[0](bot_feat2, bot_feat2_teacher) + criterion[0](bot_feat1, bot_feat1_teacher)
			loss_att = criterion[0](sau_feat, sau_feat_teacher) + criterion[0](cau_feat, cau_feat_teacher)
			# 
			loss = loss_hard * 1.0 + loss_soft * 0.005 + loss_feat * 0.05 + loss_att * 0.1


		losses.update(loss.item())

		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

	return losses.avg


def valid(val_loader, network):
	PSNR = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	for batch in val_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with torch.no_grad():						
			output, sau_feat, cau_feat, bot_feat2, bot_feat1 = network(source_img)
			output = output.clamp_(-1, 1)		


		mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
		psnr = 10 * torch.log10(1 / mse_loss).mean()
		PSNR.update(psnr.item(), source_img.size(0))

	return PSNR.avg


if __name__ == '__main__':
	setting_filename = os.path.join('configs', args.exp, args.model+'.json')
	if not os.path.exists(setting_filename):
		setting_filename = os.path.join('configs', args.exp, 'default.json')
	with open(setting_filename, 'r') as f:
		setting = json.load(f)

	checkpoint=None
	print("model name:",args.model.replace('-', '_'))

	network = eval(args.model.replace('-', '_'))()
	network = nn.DataParallel(network).cuda()
	if checkpoint is not  None:
		network.load_state_dict(checkpoint['state_dict'])
	

	network_teacher = eval('TeacherNet')()
	network_teacher.cuda()
	saved_teacherNet_dir = os.path.join(args.save_dir, args.exp, 'TeacherNet'+'.pth')
	print("saved_teacherNet_dir:",saved_teacherNet_dir)

	if os.path.exists(saved_teacherNet_dir):
		print('==> existing trained model: ' + 'TeacherNet')
		network_teacher.load_state_dict(single(saved_teacherNet_dir))
	else:
		print('==> No existing trained model!')
		exit(0)
	

	criterion = []
	criterion.append(nn.L1Loss())
	P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}).cuda()
	criterion.append(P_loss) 

	if setting['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
	elif setting['optimizer'] == 'adamw':
		optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
	else:
		raise Exception("ERROR: unsupported optimizer")


	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
	scaler = GradScaler()

	if checkpoint is not None:
		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler.load_state_dict(checkpoint['lr_scheduler'])
		scaler.load_state_dict(checkpoint['scaler'])
		best_psnr = checkpoint['best_psnr']
		start_epoch = checkpoint['epoch'] + 1
	else:
		best_psnr = 0
		start_epoch = 0

	best_psnr = 0

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	train_dataset = PairLoader(dataset_dir, 'train', 'train', 
								setting['patch_size'],
							    setting['edge_decay'],
							    setting['only_h_flip'])
	train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
	val_dataset = PairLoader(dataset_dir, 'valid', setting['valid_mode'], 
							  setting['patch_size'])
	val_loader = DataLoader(val_dataset,

							batch_size=1,
                            num_workers=args.num_workers,
                            pin_memory=True)

	save_dir = os.path.join(args.save_dir, args.exp)
	os.makedirs(save_dir, exist_ok=True)


	print('==> Start training, current model name: ' + args.model)


	train_ls, test_ls, idx = [], [], []

	for epoch in tqdm(range(start_epoch,setting['epochs'] + 1)):
		loss = train(train_loader, network, network_teacher, criterion, optimizer, scaler)

		train_ls.append(loss)
		idx.append(epoch)

		scheduler.step()

		if epoch % setting['eval_freq'] == 0:
			avg_psnr = valid(val_loader, network)
			if avg_psnr > best_psnr:
				best_psnr = avg_psnr
				print(avg_psnr)

				torch.save({'state_dict': network.state_dict(),
							'optimizer':optimizer.state_dict(),
							'lr_scheduler':scheduler.state_dict(),
							'scaler':scaler.state_dict(),
							'epoch':epoch,
							'best_psnr':best_psnr
							},
						   os.path.join(save_dir, args.model+'.pth'))

