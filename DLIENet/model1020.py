import torch.nn as nn
import torch
import torch.nn.functional as F

import math

class SEnet(nn.Module):
    def __init__(self,channels,ratio=16):
        super(SEnet, self).__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        # 经过两次全连接层，一次较小，一次还原
        self.fc=nn.Sequential(
            nn.Linear(channels,channels//ratio,False),
            nn.ReLU(),
            nn.Linear(channels//ratio, channels, False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_=x.size() #取出batch size和通道数
        # b,c,w,h->b,c,1,1->b,c 以便进行全连接
        avg=self.avgpool(x).view(b,c)
        #b,c->b,c->b,c,1,1 以便进行线性加权
        fc=self.fc(avg).view(b,c,1,1)
        return fc*x



class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        # conv_block = [  nn.ReflectionPad2d(1),#参数是padding，使用输入 tensor的反射来填充
        #                 nn.Conv2d(in_features, in_features, 3),
        #                 nn.InstanceNorm2d(in_features),
        #                 #torch.nn.InstanceNorm2d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        #                 #Applies Instance Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper Instance Normalization: The Missing Ingredient for Fast Stylization .
        #                 nn.ReLU(inplace=True),
        #                 nn.ReflectionPad2d(2),
        #                 nn.Conv2d(in_features, in_features, 3,dilation = 2),
        #                 nn.InstanceNorm2d(in_features)  ]
        #torch.nn.Sequential(*args)
        #A sequential container.模型将按照在构造函数中传递的顺序添加到模型中。 或者，也可以传递模型的有序字典。
        #self.conv_block = nn.Sequential(*conv_block)
        self.SEnt1=SEnet(in_features)
        self.Refle1=nn.ReflectionPad2d(1)
        self.conv1=nn.Conv2d(in_features, in_features, 3)
        self.inst1=nn.InstanceNorm2d(in_features)
        # torch.nn.InstanceNorm2d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # Applies Instance Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper Instance Normalization: The Missing Ingredient for Fast Stylization .
        self.relu=nn.ReLU(inplace=True)
        self.Refle2=nn.ReflectionPad2d(1)
        self.conv2=nn.Conv2d(in_features, in_features, 3, dilation=1)
        self.inst2=nn.InstanceNorm2d(in_features)

        self.Refle3 = nn.ReflectionPad2d(2)
        self.conv3 = nn.Conv2d(in_features, in_features, 3, dilation=2)
        self.inst3 = nn.InstanceNorm2d(in_features)

        self.Refle4 = nn.ReflectionPad2d(4)
        self.conv4 = nn.Conv2d(in_features, in_features, 3, dilation=4)
        self.inst4 = nn.InstanceNorm2d(in_features)

        self.conv5 = nn.Conv2d(in_features*3, in_features, 3, padding=1)


        self.out =nn.Conv2d(in_features*2, in_features, 3, padding=1)
        self.inst_out=nn.InstanceNorm2d(in_features)

        self.relu_out=nn.ReLU(inplace=True)


    def forward(self, x):
        y=x
        z= self.SEnt1(x)
        
        x=self.Refle1(x)
        x=self.conv1(x)
        x=self.inst1(x)
        # torch.nn.InstanceNorm2d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # Applies Instance Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper Instance Normalization: The Missing Ingredient for Fast Stylization .
        x=self.relu(x)



        #第一次空洞率为1的卷积
        x1=self.Refle2(x)
        x1=self.conv2(x1)
        x1=self.inst2(x1)

        #第二次空洞率为2的卷积
        x2 = self.Refle2(x)
        x2 = self.conv2(x2)
        x2 = self.inst2(x2)


        #第三次空洞率为4的卷积
        x3 = self.Refle2(x)
        x3 = self.conv2(x3)
        x3 = self.inst2(x3)
        #空洞率为4的卷积之后的大小：torch.Size([1, 256, 64, 64])


        #三个空洞卷积融合
        x=torch.cat((x1,x2,x3),1)
        x=self.conv5(x)

        x=torch.cat((x,z),1)
        x=self.out(x)
        x= self.inst_out(x)
        x= self.relu_out(x)

        #经过卷积 从9通道到3通道


        return x + y

class lrBLock_l3(nn.Module):

    def __init__(self, channelDepth, windowSize=3):

        super(lrBLock_l3, self).__init__()
        #padding = math.floor(windowSize/2)

        self.res_l3 = ResidualBlock(channelDepth)
        self.res_l2 = ResidualBlock(channelDepth)
        self.res_l1 = ResidualBlock(channelDepth)

    def forward(self, x):

        x_down2 = F.interpolate(x,scale_factor = 0.5,mode='bilinear',align_corners=True,recompute_scale_factor=True) #128
        x_down4 = F.interpolate(x_down2,scale_factor = 0.5,mode='bilinear',align_corners=True,recompute_scale_factor=True) #64

        x_reup2 = F.interpolate(x_down4,scale_factor = 2,mode='bilinear',align_corners=True) #128
        x_reup = F.interpolate(x_down2,scale_factor = 2,mode='bilinear',align_corners=True) #256

        Laplace_2 = x_down2 - x_reup2
        Laplace_1 = x - x_reup

        Scale1 = self.res_l1(x_down4)
        Scale2 = self.res_l2(Laplace_2)
        Scale3 = self.res_l3(Laplace_1)

        output1 = Scale1
        output2 = F.interpolate(Scale1,scale_factor = 2,mode='bilinear',align_corners=True) + Scale2
        output3 = F.interpolate(output2,scale_factor = 2,mode='bilinear',align_corners=True) + Scale3

        return output3


class lrBLock_l2(nn.Module):

    def __init__(self, channelDepth, windowSize=3):

        super(lrBLock_l2, self).__init__()
        padding = math.floor(windowSize/2)

        self.res_l2 = ResidualBlock(channelDepth)
        self.res_l1 = ResidualBlock(channelDepth)

    def forward(self, x):

        x_down2 = F.interpolate(x,scale_factor = 0.5,mode='bilinear',align_corners=True,recompute_scale_factor=True) #128

        x_reup = F.interpolate(x_down2,scale_factor = 2,mode='bilinear',align_corners=True) #256

        Laplace_1 = x - x_reup
        #print('x_down2 = {}'.format(x_down2.shape))

        Scale1 = self.res_l1(x_down2)
        Scale2 = self.res_l2(Laplace_1)

        output1 = Scale1
        output2 = F.interpolate(Scale1,scale_factor = 2,mode='bilinear',align_corners=True) + Scale2

        return output2
class LPB_3(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=12,in_features = 64,out_features =128):
        super(LPB_3, self).__init__()
        # model = [nn.ReflectionPad2d(3),
        #          nn.Conv2d(input_nc, 64, 7),
        #          nn.InstanceNorm2d(64),
        #          nn.ReLU(inplace=True) ]
        model =[]
        self.Refle1=nn.ReflectionPad2d(3)
        self.conv1=nn.Conv2d(input_nc, 64, 7)
        self.inst1=nn.InstanceNorm2d(64)
        self.Relu1=nn.ReLU(inplace=True)

        # 下采样
        #in_features = 64
        #out_features = in_features * 2
        self.conv2=nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)# 64 128
        self.inst2=nn.InstanceNorm2d(out_features)# 128
        self.Relu2=nn.ReLU(inplace=True)
        # 进行金字塔恢复
        self.lpb1=lrBLock_l3(in_features*2)

        #in_features = out_features
        #out_features = in_features * 2

        self.conv3 = nn.Conv2d(in_features*2, out_features*2, 3, stride=2, padding=1)# 128 256
        self.inst3 = nn.InstanceNorm2d(out_features*2)#256
        self.Relu3 = nn.ReLU(inplace=True)
        #进行金字塔恢复
        self.lpb2 = lrBLock_l3(in_features * 4)

        #in_features = out_features 256
        #out_features = in_features * 2 =512
        # 残差块
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features*4)]

        # 上采样
        #out_features = in_features // 2=128

        self.conv4=nn.ConvTranspose2d(in_features*4, out_features, 3, stride=2, padding=1, output_padding=1)# 256 128
        self.inst4=nn.InstanceNorm2d(out_features)# 128
        self.Relu4=nn.ReLU(inplace=True)
        # 进行金字塔恢复
        self.lpb3 = lrBLock_l3(in_features * 2)

        # in_features = out_features
        # out_features = in_features // 2

        self.conv5 = nn.ConvTranspose2d(in_features*2, out_features//2, 3, stride=2, padding=1, output_padding=1) #128 64
        self.inst5 = nn.InstanceNorm2d(out_features//2)# 64
        self.Relu5 = nn.ReLU(inplace=True)
        # 进行金字塔恢复
        self.lpb4 = lrBLock_l3(in_features )

        # in_features = out_features
        # out_features = in_features // 2

        #输出层
        self.Refle2=nn.ReflectionPad2d(3)
        self.conv6=nn.Conv2d(64, output_nc, 7)
        self.Relu1=nn.Tanh()
        self.model = nn.Sequential(*model)

        #self.conv5 = nn.Conv2d(768, 256, 3, padding=1)
        self.conv_cat1 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv_cat2 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv_cat3 = nn.Conv2d(128, 64, 3, padding=1)

    def forward(self, x):
        #初始化卷积块
        x0=self.Refle1(x)
        x1=self.conv1(x0)
        x2=self.inst1(x1)
        x3=self.Relu1(x2)
        #print('x3 shape :{}'.format(x3.shape))

        # 下采样
        #x4=self.conv2(x3)
        x5=self.conv2(x3)
        #print('x5 shape :{}'.format(x5.shape))
        x6=self.inst2(x5)
        x7=self.Relu2(x6)
        #print('relu2 result shape :{}'.format(x7.shape))

        x8=self.conv3(x7)
        #print('x8 shape :{}'.format(x8.shape))
        x9=self.inst3(x8)
        x10=self.Relu3(x9)

        #残差块
        x11=self.model(x10)
        #print('x11 shape :{}'.format(x11.shape))



        #拼接x10和x11 [1, 256, 64, 64]
        cat1=torch.cat([x10, x11], dim=1)
        #print('cat1 shape :{}'.format(cat1.shape))
        x11_1=self.conv_cat1(cat1)

        #上采样

        x12=self.conv4(x11_1)
        #print('x12 shape :{}'.format(x12.shape))
        x13=self.inst4(x12)
        x14=self.Relu4(x13)

        #拼接 x7和x12 [1, 128, 128, 128]  其实是x14
        cat2 = torch.cat([x7, x14], dim=1)
        #print('cat2 shape :{}'.format(cat2.shape))
        x14_1=self.conv_cat2(cat2)



        x15=self.conv5(x14_1)
        #print('x15 shape :{}'.format(x15.shape))
        x16=self.inst5(x15)
        x17=self.Relu5(x16)

        # 拼接x3和x15 [1, 64, 256, 256] 其实是x17
        cat3 = torch.cat([x3, x17], dim=1)
        #print('cat3 shape :{}'.format(cat3.shape))
        x17_1=self.conv_cat3(cat3)

        #输出层
        x18=self.Refle2(x17_1)
        x19=self.conv6(x18)
        x20=self.Relu1(x19)
        return x20
class LPB_2(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=12,in_features = 64,out_features =128):
        super(LPB_2, self).__init__()
        # model = [nn.ReflectionPad2d(3),
        #          nn.Conv2d(input_nc, 64, 7),
        #          nn.InstanceNorm2d(64),
        #          nn.ReLU(inplace=True) ]
        model =[]
        self.Refle1=nn.ReflectionPad2d(3)
        self.conv1=nn.Conv2d(input_nc, 64, 7)
        self.inst1=nn.InstanceNorm2d(64)
        self.Relu1=nn.ReLU(inplace=True)

        # 下采样
        #in_features = 64
        #out_features = in_features * 2
        self.conv2=nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)# 64 128
        self.inst2=nn.InstanceNorm2d(out_features)# 128
        self.Relu2=nn.ReLU(inplace=True)
        # 进行金字塔恢复
        self.lpb1=lrBLock_l2(in_features*2)

        #in_features = out_features
        #out_features = in_features * 2

        self.conv3 = nn.Conv2d(in_features*2, out_features*2, 3, stride=2, padding=1)# 128 256
        self.inst3 = nn.InstanceNorm2d(out_features*2)#256
        self.Relu3 = nn.ReLU(inplace=True)
        #进行金字塔恢复
        self.lpb2 = lrBLock_l2(in_features * 4)

        #in_features = out_features 256
        #out_features = in_features * 2 =512
        # 残差块
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features*4)]

        # 上采样
        #out_features = in_features // 2=128

        self.conv4=nn.ConvTranspose2d(in_features*4, out_features, 3, stride=2, padding=1, output_padding=1)# 256 128
        self.inst4=nn.InstanceNorm2d(out_features)# 128
        self.Relu4=nn.ReLU(inplace=True)
        # 进行金字塔恢复
        self.lpb3 = lrBLock_l2(in_features * 2)

        # in_features = out_features
        # out_features = in_features // 2

        self.conv5 = nn.ConvTranspose2d(in_features*2, out_features//2, 3, stride=2, padding=1, output_padding=1) #128 64
        self.inst5 = nn.InstanceNorm2d(out_features//2)# 64
        self.Relu5 = nn.ReLU(inplace=True)
        # 进行金字塔恢复
        self.lpb4 = lrBLock_l2(in_features )

        # in_features = out_features
        # out_features = in_features // 2

        #输出层
        self.Refle2=nn.ReflectionPad2d(3)
        self.conv6=nn.Conv2d(64, output_nc, 7)
        self.Relu1=nn.Tanh()
        self.model = nn.Sequential(*model)

        #self.conv5 = nn.Conv2d(768, 256, 3, padding=1)
        self.conv_cat1 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv_cat2 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv_cat3 = nn.Conv2d(128, 64, 3, padding=1)

    def forward(self, x):
        #初始化卷积块
        x0=self.Refle1(x)
        x1=self.conv1(x0)
        x2=self.inst1(x1)
        x3=self.Relu1(x2)
        #print('x3 shape :{}'.format(x3.shape))

        # 下采样
        #x4=self.conv2(x3)
        x5=self.conv2(x3)
        #print('x5 shape :{}'.format(x5.shape))
        x6=self.inst2(x5)
        x7=self.Relu2(x6)
        #print('relu2 result shape :{}'.format(x7.shape))

        x8=self.conv3(x7)
        #print('x8 shape :{}'.format(x8.shape))
        x9=self.inst3(x8)
        x10=self.Relu3(x9)

        #残差块
        x11=self.model(x10)
        #print('x11 shape :{}'.format(x11.shape))



        #拼接x10和x11 [1, 256, 64, 64]
        cat1=torch.cat([x10, x11], dim=1)
        #print('cat1 shape :{}'.format(cat1.shape))
        x11_1=self.conv_cat1(cat1)

        #上采样

        x12=self.conv4(x11_1)
        #print('x12 shape :{}'.format(x12.shape))
        x13=self.inst4(x12)
        x14=self.Relu4(x13)

        #拼接 x7和x12 [1, 128, 128, 128]  其实是x14
        cat2 = torch.cat([x7, x14], dim=1)
        #print('cat2 shape :{}'.format(cat2.shape))
        x14_1=self.conv_cat2(cat2)



        x15=self.conv5(x14_1)
        #print('x15 shape :{}'.format(x15.shape))
        x16=self.inst5(x15)
        x17=self.Relu5(x16)

        # 拼接x3和x15 [1, 64, 256, 256] 其实是x17
        cat3 = torch.cat([x3, x17], dim=1)
        #print('cat3 shape :{}'.format(cat3.shape))
        x17_1=self.conv_cat3(cat3)

        #输出层
        x18=self.Refle2(x17_1)
        x19=self.conv6(x18)
        x20=self.Relu1(x19)
        return x20
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=12):
        super(Generator, self).__init__()
        self.Stage1 = LPB_3()
        self.Stage2 = LPB_2()
        self.Stage3 = LPB_2()

#         # 初始卷积块
#         model = [   nn.ReflectionPad2d(3),
#                     nn.Conv2d(input_nc, 64, 7),
#                     nn.InstanceNorm2d(64),
#                     nn.ReLU(inplace=True) ]
# #ReflectionPad2d（）搭配7x7卷积，先在特征图周围以反射的方式补长度，使得卷积后特征图尺寸不变
# #InstanceNorm2d（）是相比于batchNorm更加适合图像生成，风格迁移的归一化方法，相比于batchNorm跨样本，
# #单通道统计，InstanceNorm采用单样本，单通道统计，括号中的参数代表通道数
#         # 下采样
#         # in_features = 64
#         # out_features = in_features*2
#         # for _ in range(2):
#         #     model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
#         #                 nn.InstanceNorm2d(out_features),
#         #                 nn.ReLU(inplace=True) ]
#         #     in_features = out_features
#         #     out_features = in_features*2
#         #
#         # #残差块
#         # for _ in range(n_residual_blocks):
#         #     model += [ResidualBlock(in_features)]
#         #
#         # # 上采样
#         # out_features = in_features//2
#         # for _ in range(2):
#         #     model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
#         #                 nn.InstanceNorm2d(out_features),
#         #                 nn.ReLU(inplace=True) ]
#         #     in_features = out_features
#         #     out_features = in_features//2
#
#
#         # 输出层
#         model += [  nn.ReflectionPad2d(3),
#                     nn.Conv2d(64, output_nc, 7),
#                     nn.Tanh() ] #nn.Tanh()：Applies the element-wise function:
#
#         self.model = nn.Sequential(*model)

    def forward(self, x):
        x_down2 = F.interpolate(x, scale_factor=0.5, mode='bilinear',align_corners=True,recompute_scale_factor=True)  # 128
        x_down4 = F.interpolate(x_down2, scale_factor=0.5, mode='bilinear',align_corners=True,recompute_scale_factor=True)  # 64

        x_reup2 = F.interpolate(x_down4, scale_factor=2, mode='bilinear',align_corners=True)  # 128
        x_reup = F.interpolate(x_down2, scale_factor=2, mode='bilinear',align_corners=True)  # 256
        Laplace_2 = x_down2 - x_reup2
        Laplace_1 = x - x_reup
        #print("s_down4={}".format(x_down4.shape))
        Scale1 = self.Stage1(x_down4)
        Scale2 = self.Stage2(Laplace_2)
        Scale3 = self.Stage3(Laplace_1)
        output1 = Scale1
        output2 = F.interpolate(Scale1, scale_factor=2, mode='bilinear',align_corners=True) + Scale2
        output3 = F.interpolate(output2, scale_factor=2, mode='bilinear',align_corners=True) + Scale3
        #return self.model(x)
        return output1,output2,output3,Scale2,Scale3
#=======================================================================Discriminator
#判别器部分：结构比生成器更加简单，经过5层卷积，通道数缩减为1，最后池化平均，尺寸也缩减为1x1，
#最最后reshape一下，变为（batchsize,1)
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # 卷积层
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN 分类层
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # 平均池化和展开（view）
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1),x
# net=Generator(3,3)
# #print(net)
# imgs = torch.rand(1,3, 256,256)
# jieguo=net(imgs)
# print(jieguo[2].shape)

