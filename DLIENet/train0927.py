#!/usr/bin/python3
# zheshi 0920 shezhi le kongdongjuanji de daima
import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from model0927 import Generator
from model0927 import Discriminator
from untils import ReplayBuffer
from untils import LambdaLR
# from untilCeshi import Logger
from untils import weights_init_normal
from myDatalader0927 import ImageDataset
from torchvision.transforms import InterpolationMode
from logs import write_infor, Write_time
import time
import torch.nn.functional as F
import torch.nn as nn

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
#device_ids = [3, 4]

parser = argparse.ArgumentParser()

parser.add_argument('--savePath', type=str, default='./output/output20220927', help='weight save path')
parser.add_argument('--dataroot', type=str, default='./dataset', help='root directory of the dataset')
parser.add_argument('--model_name', type=str, default='model0927', help='root directory of the dataset')
parser.add_argument('--trainInfo_filename', type=str, default='trainInfo_log.txt')
parser.add_argument('--mode', type=str, default='train', help='number of epochs of training')

# parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
# parser.add_argument('--dataroot', type=str, default='./dataset/train2', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
##学习率下降的epoch
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='epoch to start linearly decaying the learning rate to 0')
# parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
# parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
# parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
# parser.add_argument('--batchSize', type=int, default=1)
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--size', type=int, default=256)
parser.add_argument('--train_dataroot', type=str, default='./dataset/train2', help='data root path')
parser.add_argument('--weight', type=str, default='', help='load pre train weight')
# parser.add_argument('--savePath', type=str, default='./output/output2022092303', help='weight save path')
# parser.add_argument('--numworker', type=int, default=2)
parser.add_argument('--every', type=int, default=20, help='plot train result every * iters')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
# parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### =============================================定义变量 ######
# 网络名称 # 初始化生成器和判别器
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()
    # netG_A2B = nn.DataParallel(netG_A2B,device_ids).cuda()
    # netG_B2A= nn.DataParallel(netG_B2A,device_ids).cuda()
    # netD_A= nn.DataParallel(netD_A,device_ids).cuda()
    # netD_B= nn.DataParallel(netD_B,device_ids).cuda()

# 初始化权重
netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# 定义损失函数
criterion_GAN = torch.nn.MSELoss()  # 均方损失函数
criterion_patchGAN = torch.nn.MSELoss()
# L1 loss用来让生成的图片和训练的目标图片尽量相似,而图像中高频的细节部分则交由GAN来处理,
# 图像中的低频部分有patchGAN处理
# 创建一个标准来测量输入xx和目标yy中每个元素之间的平均绝对误差（MAE），源码中的解释。
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()


#====================== Loss==================

scale1_loss = nn.MSELoss()
scale2_loss = nn.MSELoss()
scale3_loss = nn.MSELoss()
laplace_loss2 = nn.L1Loss()
laplace_loss3 = nn.L1Loss()
scale1_color = nn.CosineSimilarity(dim= 1, eps = 1e-6)
scale2_color = nn.CosineSimilarity(dim= 1, eps = 1e-6)
scale3_color = nn.CosineSimilarity(dim= 1, eps = 1e-6)


scale1_loss_A = nn.MSELoss()
scale2_loss_A = nn.MSELoss()
scale3_loss_A = nn.MSELoss()
laplace_loss2_A = nn.L1Loss()
laplace_loss3_A = nn.L1Loss()
scale1_color_A = nn.CosineSimilarity(dim= 1, eps = 1e-6)
scale2_color_A = nn.CosineSimilarity(dim= 1, eps = 1e-6)
scale3_color_A = nn.CosineSimilarity(dim= 1, eps = 1e-6)

#=============================================

# Optimizers & LR schedulers
# ===========================================定义优化器
# 优化器的设置保证了只更新生成器或判别器，不会互相影响
# 1.Adm算法  torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# 2itertools.chain 迭代器能够将多个可迭代对象合并成一个更长的可迭代对象
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))
# computing running averages of gradient and its square (default: (0.9, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
# =========================================定义学习率更新方式
# torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
# 将每个参数组的学习率设置为给定函数的初始lr倍。 当last_epoch = -1时，将初始lr设置为lr
# lr_lambda：在给定整数参数empoch或此类函数列表的情况下计算乘法因子的函数，针对optimizer中的每个组一个函数.param_groups
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

# 首先定义好buffer   来源于工具包13行
# 是为了训练的稳定，采用历史生成的虚假样本来更新判别器，而不是当前生成的虚假样本
# 定义了一个buffer对象，有一个数据存储表data，大小预设为50，
# 它的运转流程是这样的：数据表未填满时，每次读取的都是当前生成的虚假图像，
# 当数据表填满时，随机决定 1. 在数据表中随机抽取一批数据，返回，并且用当前数据补充进来 2. 采用当前数据
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# =========================================Dataset loader
transforms_ = [transforms.Resize(int(opt.size * 1.12), InterpolationMode.BICUBIC),  # 调整输入图片的大小
               transforms.RandomCrop(opt.size),  # 随机裁剪
               transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
               # 归一化，这两行不能颠倒顺序呢，归一化需要用到tensor型
               ]

# =========================================加载训练数据集的方式transforms_=transforms_,
dataloader = DataLoader(ImageDataset(opt, unaligned=False),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu,drop_last=True)  # 两个数据集均打乱

# =========================================损失图
# logger = Logger(opt.n_epochs, len(dataloader))
###################################

if not os.path.exists(opt.savePath):
    os.makedirs(opt.savePath)

# 训练开始时间
start_time = time.time()
# 写入开始时间
write_infor(opt.savePath, opt.trainInfo_filename, 'start_time', start_time)

# ==========================================================Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    write_infor(opt.savePath, opt.trainInfo_filename, '\nEpoch', epoch)
    print('\n第 {} 轮训练'.format(epoch))
    for i, batch in enumerate(dataloader):
        # 输入
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # ===================================Generators A2B and B2A ######
        # 生成器损失函数：损失函数=身份损失+对抗损失+循环一致损失+patchGan损失
        optimizer_G.zero_grad()
        # print("开始")








        # Identity loss
        # 如果输入实数B，则G_A2B（B）应等于B
        x_a1,x_a2,same_B,y_a1,y_a2 = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        # 如果输入实数A，则G_B2A（A）应等于A
        x_a3,x_a4,same_A,y_a3,y_a4 = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        #======================================B的金字塔========================================================================
        Scale1,Scale2,fake_B,res2,res3 = netG_A2B(real_A)
        pred_fake, patch_G_A2B = netD_B(fake_B)
        patch_real = torch.ones_like(patch_G_A2B)
        loss_patch_A2B = criterion_patchGAN(patch_G_A2B, patch_real)
        loss_GAN_A2B = criterion_GAN(pred_fake.squeeze(-1), target_real)


        
        gt_down2 = F.interpolate(real_B, scale_factor=0.5, mode='bilinear')  # 128
        gt_down4 = F.interpolate(gt_down2, scale_factor=0.5, mode='bilinear')  # 64

        in_down2 = F.interpolate(fake_B, scale_factor=0.5, mode='bilinear')  # 128
        in_down4 = F.interpolate(in_down2, scale_factor=0.5, mode='bilinear')  # 64

        reup2 = F.interpolate(gt_down4, scale_factor=2, mode='bilinear')  # 128
        reup3 = F.interpolate(gt_down2, scale_factor=2, mode='bilinear')  # 256

        laplace2 = gt_down2 - reup2
        laplace3 = real_B - reup3

        scale3loss = scale3_loss(fake_B, real_B)
        scale2loss = scale2_loss(Scale2, gt_down2)
        scale1loss = scale1_loss(Scale1, gt_down4)
        scale1color = torch.mean(-1 * scale1_color(Scale1, gt_down4))
        scale2color = torch.mean(-1 * scale2_color(Scale2, gt_down2))
        scale3color = torch.mean(-1 * scale3_color(fake_B, real_B))
        laplaceloss2 = laplace_loss2(res2, laplace2)
        laplaceloss3 = laplace_loss3(res3, laplace3)
        loss1_B = 2 * scale1loss + scale2loss + scale3loss + 2 * scale1color + scale2color + scale3color + laplaceloss2 + laplaceloss3
        # =============================================================================================================================



        #===============================================A的金字塔===========================================================



        Scale1_A,Scale2_A,fake_A,res2_A,res3_A = netG_B2A(real_B)
        pred_fake, patch_G_B2A = netD_A(fake_A)
        patch_real_B2A = torch.ones_like(patch_G_B2A)
        loss_patch_B2A = criterion_patchGAN(patch_G_B2A, patch_real_B2A)
        loss_GAN_B2A = criterion_GAN(pred_fake.squeeze(-1), target_real)
        
        gt_down2_A = F.interpolate(real_A, scale_factor=0.5, mode='bilinear')  # 128
        gt_down4_A = F.interpolate(gt_down2_A, scale_factor=0.5, mode='bilinear')  # 64

        in_down2_A = F.interpolate(fake_A, scale_factor=0.5, mode='bilinear')  # 128
        in_down4_A = F.interpolate(in_down2_A, scale_factor=0.5, mode='bilinear')  # 64

        reup2_A = F.interpolate(gt_down4_A, scale_factor=2, mode='bilinear')  # 128
        reup3_A = F.interpolate(gt_down2_A, scale_factor=2, mode='bilinear')  # 256

        laplace2_A = gt_down2_A - reup2_A
        laplace3_A = real_A - reup3_A

        scale3loss_A = scale3_loss_A(fake_A, real_A)
        scale2loss_A = scale2_loss_A(Scale2_A, gt_down2_A)
        scale1loss_A = scale1_loss_A(Scale1_A, gt_down4_A)
        scale1color_A = torch.mean(-1 * scale1_color_A(Scale1_A, gt_down4_A))
        scale2color_A = torch.mean(-1 * scale2_color_A(Scale2_A, gt_down2_A))
        scale3color_A = torch.mean(-1 * scale3_color_A(fake_A, real_A))
        laplaceloss2_A = laplace_loss2_A(res2_A, laplace2_A)
        laplaceloss3_A = laplace_loss3_A(res3_A, laplace3_A)
        loss1_A = 2 * scale1loss_A + scale2loss_A + scale3loss_A + 2 * scale1color_A + scale2color_A + scale3color_A + laplaceloss2_A + laplaceloss3_A




        #=================================================================================================================




        x_a7,x_a8,recovered_B,y_a7,y_a8 = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        x_a9,x_a10,recovered_A,y_a9,y_a10 = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_patch_A2B + loss_patch_B2A  +loss1_A +loss1_B
        loss_G.backward()

        optimizer_G.step()
        ###################################

        # ===================================Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real, patch_D_A_real = netD_A(real_A)
        patch_real_A = torch.zeros_like(patch_D_A_real)
        loss_patch_A_real = criterion_patchGAN(patch_D_A_real, patch_real_A)
        loss_D_real = criterion_GAN(pred_real.squeeze(-1), target_real)

        # Fake loss

        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake, patch_D_A_fake = netD_A(fake_A.detach())
        patch_fake_A = torch.zeros_like(patch_D_A_fake)
        loss_patch_A_fake = criterion_patchGAN(patch_D_A_fake, patch_fake_A)

        loss_D_fake = criterion_GAN(pred_fake.squeeze(-1), target_fake)

        # Total loss
        #print('loss_D_real:{} ,loss_D_fake :{},loss_patch_A_real :{},loss_patch_A_fake:{}'.format(loss_D_real.data,loss_D_fake , loss_patch_A_real ,loss_patch_A_fake))
        loss_D_A = (loss_D_real + loss_D_fake + loss_patch_A_real + loss_patch_A_fake) * 0.5
        # print(loss_D_A)
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        # ===================================Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real, patch_D_B_real = netD_B(real_B)
        patch_real_B = torch.zeros_like(patch_D_B_real)
        loss_patch_B_real = criterion_patchGAN(patch_D_B_real, patch_real_B)
        loss_D_real = criterion_GAN(pred_real.squeeze(-1), target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake, patch_D_B_fake = netD_B(fake_B.detach())
        patch_fake_B = torch.zeros_like(patch_D_B_fake)
        loss_patch_B_fake = criterion_patchGAN(patch_D_B_fake, patch_fake_B)
        loss_D_fake = criterion_GAN(pred_fake.squeeze(-1), target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake + loss_patch_B_real + loss_patch_B_fake) * 0.5

        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################
        # print('loss_G:{},loss_G_identity:{},loss_G_GAN:{},loss_G_cycle:{},loss_D:{}'.format(loss_G
        # ,(loss_identity_A + loss_identity_B)
        # ,(loss_GAN_A2B + loss_GAN_B2A)
        # , (loss_cycle_ABA + loss_cycle_BAB)
        # ,(loss_D_A + loss_D_B)
        # ))
        print('\rEpoch: [{}/{}]'.format(epoch, opt.n_epochs - opt.epoch), "iter: [{}/{}]".format(i, len(dataloader)),
              'loss_G:{}, loss_G_identity:{}, loss_G_GAN:{}, loss_G_cycle:{}, loss_D:{}'.format(loss_G
                                                                                                , (
                                                                                                            loss_identity_A + loss_identity_B)
                                                                                                , (
                                                                                                            loss_GAN_A2B + loss_GAN_B2A)
                                                                                                , (
                                                                                                            loss_cycle_ABA + loss_cycle_BAB)
                                                                                                , (loss_D_A + loss_D_B)
                                                                                                ), flush=True)
        if i % 100 == 0:
            write_infor(opt.savePath, opt.trainInfo_filename, 'batch_{}: loss_D'.format(i).ljust(10),
                        loss_D_A + loss_D_B)
            write_infor(opt.savePath, opt.trainInfo_filename, 'model_name', opt.model_name)

        # Progress report (http://localhost:8097)
        '''  
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),
                    'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
                   images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})'''

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    pathG_A2B = '%s/netG_A2B.pth' % (opt.savePath)
    pathG_B2A = '%s/netG_B2A.pth' % (opt.savePath)
    pathD_A = '%s/netD_A.pth' % (opt.savePath)
    pathD_B = '%s/netD_B.pth' % (opt.savePath)
    if not os.path.exists(opt.savePath):
        os.makedirs(opt.savePath)

    # Save models checkpoints
    # torch.save(netG_A2B.state_dict(), pathG_A2B)
    # torch.save(netG_B2A.state_dict(), pathG_B2A)
    # torch.save(netD_A.state_dict(), pathD_A)
    # torch.save(netD_B.state_dict(), pathD_B)
    torch.save(netG_A2B.state_dict(), '{}/{}_netG_A2B.pth'.format(opt.savePath, epoch))
    torch.save(netG_B2A.state_dict(), '{}/{}_netG_B2A.pth'.format(opt.savePath, epoch))
    torch.save(netD_A.state_dict(), '{}/{}_netD_A.pth'.format(opt.savePath, epoch))
    torch.save(netD_B.state_dict(), '{}/{}_netD_B.pth'.format(opt.savePath, epoch))
###################################

end_time = time.time()
# write_infor(save_root, filename, 'end_time', end_time)
write_infor(opt.savePath, opt.trainInfo_filename, 'savePath', opt.savePath)
write_infor(opt.savePath, opt.trainInfo_filename, 'shujuji', 'train2')
write_infor(opt.savePath, opt.trainInfo_filename, 'end_time', end_time)

# wenjian cunru output2022092101

