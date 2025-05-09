import argparse
import sys
import os
import time
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
#from PSNR_SSIM import PSNR,SSIM

from model0927 import Generator
from myDatalader0927 import ImageDataset
import  numpy as np
import cv2
from psnr_dataset import psnr,ssim
import  glob
import os
import numpy as np
from glob import glob
import cv2
from logs import Write_time,write_infor

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='model0927', help='size of the batches')
parser.add_argument('--generator_path', type=str, default='./output/output2022092701')
parser.add_argument('--order', type=str, default='1', help='save path number file')

#按模型划分存储的文件夹
parser.add_argument('--dataroot', type=str, default='./dataset', help='root directory of the dataset')
#parser.add_argument('--save_path', type=str, default='./test_result/img', help='weight save path')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--mode', type=str, default='caijian', help='number of epochs of training')
parser.add_argument('--result_path', type=str, default='./result')
parser.add_argument('--save_path', type=str, default='./test_result/img', help='weight save path')
parser.add_argument('--result_filename', type=str, default='result_log.txt')





parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
# parser.add_argument('--model_name', type=str, default='model0921', help='size of the batches')
# parser.add_argument('--dataroot', type=str, default='./dataset', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='netG_B2A.pth', help='B2A generator checkpoint file')
#parser.add_argument('--generator_path_ceshi', type=str, default='./output/output2022092104')
# parser.add_argument('--order', type=str, default='25', help='save path number file')

#parser.add_argument('--generator_path', type=str, default='./output/output2022092104')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--weight', type=str, default='', help='load pre train weight')
#parser.add_argument('--save_path', type=str, default='./test_result/img', help='weight save path')
parser.add_argument('--every', type=int, default=20, help='plot train result every * iters')
# parser.add_argument('--model_name', type=str, default='model0921', help='size of the batches')
# parser.add_argument('--dataroot', type=str, default='./dataset', help='root directory of the dataset')
# parser.add_argument('--generator_path', type=str, default='./output/output2022092104')
# parser.add_argument('--save_path', type=str, default='./test_result/img', help='weight save path')
# parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
# parser.add_argument('--mode', type=str, default='caijian', help='number of epochs of training')
# parser.add_argument('--result_path', type=str, default='./result')
# parser.add_argument('--result_filename', type=str, default='result_log.txt')
opt = parser.parse_args()
print(opt)


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# # Networks
# netG_A2B = Generator(opt.input_nc, opt.output_nc)
# netG_B2A = Generator(opt.output_nc, opt.input_nc)
#
# if opt.cuda:
#     netG_A2B.cuda()
#     netG_B2A.cuda()
#
# # Load state dicts
# pathG_A2B=opt.generator_A2B
# netG_A2B.load_state_dict(torch.load(opt.generator_path+opt.generator_A2B), False)
# netG_B2A.load_state_dict(torch.load(opt.generator_path+opt.generator_B2A), False)
#
# # Set model's test mode
# netG_A2B.eval()
# netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt),
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################
###################################

def read_img(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)




###### Testing######
# Create output dirs if they don't exist
# if not os.path.exists('{}/A{}'.format(opt.savePath,opt.order)):
#     os.makedirs('{}/A{}'.format(opt.savePath,opt.order))
# if not os.path.exists('{}/B{}'.format(opt.savePath,opt.order)):
#     os.makedirs('{}/B{}'.format(opt.savePath,opt.order))

print("opt.generator_path:",opt.generator_path)

files_A = sorted(glob(opt.generator_path + '/*netG_A2B.pth'), key=lambda x: eval(os.path.basename(x).split("_")[0]))
files_B = sorted(glob(opt.generator_path + '/*netG_B2A.pth'), key=lambda x: eval(os.path.basename(x).split("_")[0]))
print(len(files_A))
print(files_A)
num=0
index_num=0
best_ssim=0
#print(len(files_A))
#print(len(files_B))

while(index_num<len(files_A)):
    print('\n第 {} 轮测试'.format(index_num+1))
    pathNetD_A2B = files_A[index_num]
    pathNetD_B2A = files_B[index_num]
    file_num = os.path.basename(pathNetD_A2B).split("_")[0]
    if not os.path.exists('{}/{}/{}/A{}' .format(opt.save_path,opt.model_name,opt.order,file_num)):
        os.makedirs('{}/{}/{}/A{}' .format(opt.save_path,opt.model_name,opt.order,file_num))
    if not os.path.exists('{}/{}/{}/B{}' .format(opt.save_path,opt.model_name,opt.order,file_num)):
        os.makedirs('{}/{}/{}/B{}' .format(opt.save_path,opt.model_name,opt.order,file_num))

    # Networks
    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_B2A = Generator(opt.output_nc, opt.input_nc)

    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()

    # Load state dicts
    netG_A2B.load_state_dict(torch.load(pathNetD_A2B), False)
    netG_B2A.load_state_dict(torch.load(pathNetD_B2A), False)

    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()
    for i, batch in enumerate(dataloader):
        # Set model input
        # print("ceshi zai xunhuanlimian")
        # print("for limian %s", file_num)
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        print("real_A:",real_A.shape)
        print("real_B:",real_B.shape)

        # Generate output
        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

        # Save image files
        save_image(fake_A, '{}/{}/{}/A{}/{}.png'.format(opt.save_path,opt.model_name,opt.order,file_num,(i + 1)) )
        save_image(fake_B, '{}/{}/{}/B{}/{}.png'.format(opt.save_path,opt.model_name,opt.order,file_num,(i + 1)) )

        sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))

    sys.stdout.write('\n')
    num = num + 4
    if not os.path.exists('{}/{}/{}/A{}'.format(opt.result_path,opt.model_name,opt.order, file_num)):
        os.makedirs('{}/{}/{}/A{}'.format(opt.result_path, opt.model_name,opt.order,file_num))
    if not os.path.exists('{}/{}/{}/best'.format(opt.result_path,opt.model_name,opt.order)):
        os.makedirs('{}/{}/{}/best'.format(opt.result_path,opt.model_name,opt.order))
    start_time = time.time()
    write_infor('{}/{}/{}/A{}'.format(opt.result_path,opt.model_name,opt.order, file_num), opt.result_filename, 'start_time', start_time)
    write_infor('{}/{}/{}/A{}'.format(opt.result_path,opt.model_name,opt.order, file_num), opt.result_filename, 'model_name', opt.model_name)
    write_infor('{}/{}/{}/A{}'.format(opt.result_path,opt.model_name,opt.order, file_num), opt.result_filename, 'parameter_path', pathNetD_A2B)

    WSI_MASK_PATH1 = './dataset/test2/testB'
    WSI_MASK_PATH2 = '{}/{}/{}/A{}'.format(opt.save_path,opt.model_name,opt.order, file_num)
    path_real = sorted(glob(os.path.join(WSI_MASK_PATH1, '*.png')),
                       key=lambda x: eval(os.path.basename(x).split(".")[0]))
    path_fake = sorted(glob(os.path.join(WSI_MASK_PATH2, '*.png')),
                       key=lambda x: eval(os.path.basename(x).split(".")[0]))
    list_psnr = []
    list_ssim = []
    #print(path_real)
    #print(path_fake)
    #print(len(path_fake))
    ##list_mse = []
    for i in range(len(path_real)):
        t1 = read_img(path_real[i])
        # print(type(t1))
        t2 = read_img(path_fake[i])
        result1 = np.zeros(t1.shape, dtype=np.float32)
        result2 = np.zeros(t2.shape, dtype=np.float32)
        cv2.normalize(t1, result1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.normalize(t2, result2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        ##mse_num = mse(result1, result2)
        psnr_num = psnr(result1, result2)
        ssim_num = ssim(result1, result2)
        list_psnr.append(psnr_num)
        list_ssim.append(ssim_num)
        ##list_mse.append(mse_num)

        # 输出每张图像的指标：
        print("{}/".format(i + 1) + "{}:".format(len(path_real)))
        str = "\\"
        #print("image:" + path_real[i][(path_real[i].index(str) + 1):] + "---" + path_fake[i][(path_real[i].index(str) + 1):])
        print("PSNR:", psnr_num)
        print("SSIM:", ssim_num)

    ##print("MSE:", mse_num)
    print("测试图片数：", len(path_real))
    print("平均PSNR:", np.mean(list_psnr))  # ,list_psnr)
    print("平均SSIM:", np.mean(list_ssim))  # ,list_ssim)
    if(best_ssim<np.mean(list_ssim)):
        best_ssim=np.mean(list_ssim)
        #'{}/best'.format(opt.result_path)
        start_time1 = time.time()
        write_infor('{}/{}/{}/best'.format(opt.result_path,opt.model_name,opt.order), opt.result_filename, 'start_time', start_time1)
        write_infor('{}/{}/{}/best'.format(opt.result_path,opt.model_name,opt.order), opt.result_filename, '\nEpoch', index_num)
        write_infor('{}/{}/{}/best'.format(opt.result_path,opt.model_name,opt.order), opt.result_filename, 'psnr1', np.mean(list_psnr))
        write_infor('{}/{}/{}/best'.format(opt.result_path,opt.model_name,opt.order), opt.result_filename, 'ssim1', np.mean(list_ssim))
        write_infor('{}/{}/{}/best'.format(opt.result_path,opt.model_name,opt.order), opt.result_filename, 'model_name', opt.model_name)
        write_infor('{}/{}/{}/best'.format(opt.result_path,opt.model_name,opt.order), opt.result_filename, 'parameter_path', pathNetD_A2B)
        write_infor('{}/{}/{}/best'.format(opt.result_path,opt.model_name,opt.order), opt.result_filename, 'generator_path', opt.generator_path)
        end_time1 = time.time()
        write_infor('{}/{}/{}/best'.format(opt.result_path,opt.model_name,opt.order), opt.result_filename, 'end_time', end_time1)
        

    # write_infor('{}/A{}'.format(opt.result_path,opt.order), opt.result_filename, 'mse1', mse1_mean)
    write_infor('{}/{}/{}/A{}'.format(opt.result_path,opt.model_name,opt.order, file_num), opt.result_filename, 'psnr1', np.mean(list_psnr))
    write_infor('{}/{}/{}/A{}'.format(opt.result_path,opt.model_name,opt.order, file_num), opt.result_filename, 'ssim1', np.mean(list_ssim))  
    write_infor('{}/{}/{}/A{}'.format(opt.result_path,opt.model_name,opt.order, file_num), opt.result_filename, 'generator_path', opt.generator_path)
    end_time = time.time()
    write_infor('{}/{}/{}/A{}'.format(opt.result_path,opt.model_name,opt.order, file_num), opt.result_filename, 'end_time', end_time)
    write_infor('{}/{}/{}/A{}'.format(opt.result_path,opt.model_name,opt.order, file_num), opt.result_filename, 'duration', end_time - start_time)
    
    index_num=index_num+1


