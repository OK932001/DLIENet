
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import argparse
import random
import glob
from torch import Tensor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
class ImageDataset(Dataset):
    def __init__(self, opt, unaligned=False):
        self.root_a=opt.dataroot
        self.transform = transforms.Compose([
        transforms.Resize(int(opt.size*1.2), interpolation = InterpolationMode.BICUBIC), #调整输入图片的大小
        transforms.RandomCrop(opt.size), #随机裁剪
        transforms.RandomHorizontalFlip(),#随机水平翻转图像
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                    #归一化，这两行不能颠倒顺序呢，归一化需要用到tensor型
    ])
        self.unaligned = unaligned
        self.root_a=opt.dataroot

        self.files_A = sorted(glob.glob(opt.dataroot + '/train/trainA' + '/*.png'))
        self.files_B = sorted(glob.glob(opt.dataroot + '/train/trainB' + '/*.png'))
        #print(self.files_A)
        #print(self.files_B)

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

def cfg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--batchSize', type=int, default=1)
    parse.add_argument('--epoch', type=int, default=100)
    parse.add_argument('--size', type=int, default=256)
    parse.add_argument('--dataroot', type=str, default='./dataset', help='data root path')
    parse.add_argument('--weight', type=str, default='', help='load pre train weight')
    parse.add_argument('--savePath', type=str, default='./weights', help='weight save path')
    parse.add_argument('--numworker', type=int, default=4)
    parse.add_argument('--every', type=int, default=20, help='plot train result every * iters')
    parse.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parse.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parse.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    opt = parse.parse_args()
    return opt
def train_data_loader(arg):
    train_data_loader = DataLoader(ImageDataset(arg,unaligned=True),batch_size=arg.batchSize, shuffle=False, pin_memory=True, drop_last=True)
    return train_data_loader


#def train(opt):

if __name__ == '__main__':
    opt = cfg()
    print(opt)
    #print(opt)
    #print(glob.glob(opt.train_dataroot + '/trainA/' + '*.jpg'))

    model=train_data_loader(opt)
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    print(model.__len__())
    #print(model.files_A)
    #model=ImageDataset(opt,unaligned=True)
    #print(model.files_A)
    #print(model.__len__())
    #print(model.root_a)

    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(model):
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))
            #print(real_A)
            print(i)
    #print(model.cfg())

    #train(opt)