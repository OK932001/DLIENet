#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from model import Generator
from model import Discriminator
from untils import ReplayBuffer
from untils import LambdaLR
#from untilCeshi import Logger
from untils import weights_init_normal
from myDataloader import ImageDataset
from torchvision.transforms import InterpolationMode

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
parser = argparse.ArgumentParser()
#parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./dataset/train2', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
##学习率下降的epoch
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='epoch to start linearly decaying the learning rate to 0')
#parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
#parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
#parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
#parser.add_argument('--batchSize', type=int, default=1)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--size', type=int, default=256)
parser.add_argument('--train_dataroot', type=str, default='./dataset/train2', help='data root path')
parser.add_argument('--weight', type=str, default='', help='load pre train weight')
parser.add_argument('--savePath', type=str, default='./weights', help='weight save path')
#parser.add_argument('--numworker', type=int, default=2)
parser.add_argument('--every', type=int, default=20, help='plot train result every * iters')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
#parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
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

netD_Day = Discriminator(opt.input_nc)

#***********************加载鉴定白天的鉴别器****************************
pthfile = r'./output/2022091401/netD_A.pth'
device = torch.device('cuda:0')
netD_Day.load_state_dict(torch.load(pthfile))
netD_Day .to(device)
netD_Day.eval()

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()
# 初始化权重
netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# 定义损失函数
criterion_GAN = torch.nn.MSELoss()  # 均方损失函数
# L1 loss用来让生成的图片和训练的目标图片尽量相似,而图像中高频的细节部分则交由GAN来处理,
# 图像中的低频部分有patchGAN处理
# 创建一个标准来测量输入xx和目标yy中每个元素之间的平均绝对误差（MAE），源码中的解释。
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

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
dataloader = DataLoader(ImageDataset(opt,  unaligned=False),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)  # 两个数据集均打乱

# =========================================损失图
#logger = Logger(opt.n_epochs, len(dataloader))
###################################

# ==========================================================Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # 输入
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # ===================================Generators A2B and B2A ######
        # 生成器损失函数：损失函数=身份损失+对抗损失+循环一致损失
        optimizer_G.zero_grad()
        print("开始")

        # Identity loss
        # 如果输入实数B，则G_A2B（B）应等于B
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        # 如果输入实数A，则G_B2A（A）应等于A
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake.squeeze(-1), target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_Day(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake.squeeze(-1), target_real)


        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA=criterion_cycle(recovered_A, real_A) * 10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()
        ###################################

        # ===================================Discriminator A ######

        ###################################

        # ===================================Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real.squeeze(-1), target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake.squeeze(-1), target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        print(loss_D_B)
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

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

    # Save models checkpoints
    torch.save(netG_A2B, './output/output2022091503/netG_A2B.pth')
    torch.save(netG_B2A, './output/output2022091503/netG_B2A.pth')
    torch.save(netD_A, './output/output2022091503/netD_A.pth')
    torch.save(netD_B, './output/output2022091503/netD_B.pth')
###################################

