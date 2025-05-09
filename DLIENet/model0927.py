import torch.nn as nn
import torch.nn.functional as F
import torch
import time

# class ResidualBlock(nn.Module):
#     def __init__(self, in_features):
#         super(ResidualBlock, self).__init__()

#         conv_block = [  nn.ReflectionPad2d(1),#参数是padding，使用输入 tensor的反射来填充
#                         nn.Conv2d(in_features, in_features, 3),
#                         nn.InstanceNorm2d(in_features),
#                         #torch.nn.InstanceNorm2d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
#                         #Applies Instance Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper Instance Normalization: The Missing Ingredient for Fast Stylization .
#                         nn.ReLU(inplace=True),
#                         nn.ReflectionPad2d(2),
#                         nn.Conv2d(in_features, in_features, 3,dilation = 2),
#                         nn.InstanceNorm2d(in_features)  ]
#         #torch.nn.Sequential(*args)
#         #A sequential container.模型将按照在构造函数中传递的顺序添加到模型中。 或者，也可以传递模型的有序字典。
#         self.conv_block = nn.Sequential(*conv_block)

#     def forward(self, x):
#         return x + self.conv_block(x)


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
        self.conv5 = nn.Conv2d(768, 256, 3, padding=1)


    def forward(self, x):
        y=x
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

        #经过卷积 从9通道到3通道


        return x + y



class LPB(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=12):
        super(LPB, self).__init__()

        # 初始卷积块
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]
#ReflectionPad2d（）搭配7x7卷积，先在特征图周围以反射的方式补长度，使得卷积后特征图尺寸不变
#InstanceNorm2d（）是相比于batchNorm更加适合图像生成，风格迁移的归一化方法，相比于batchNorm跨样本，
#单通道统计，InstanceNorm采用单样本，单通道统计，括号中的参数代表通道数
        # 下采样
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        #残差块
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # 上采样
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # 输出层
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ] #nn.Tanh()：Applies the element-wise function:

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

#=======================================================================Generator
#生成器部分：网络整体上经过一个降采样然后上采样的过程，中间是一系列残差块,数目由实际情况确定，
#根据论文中所说，当输入分辨率为128x128，采用6个残差块，当输入分辨率为256x256甚至更高时，采用9个残差块
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=12):
        super(Generator, self).__init__()
        self.Stage1 = LPB()
        self.Stage2 = LPB()
        self.Stage3 = LPB()

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
        print('输入的大小：{}'.format(x.shape))
        x_down2 = F.interpolate(x, scale_factor=0.5, mode='bilinear',align_corners=True,recompute_scale_factor=True)  # 128
        x_down4 = F.interpolate(x_down2, scale_factor=0.5, mode='bilinear',align_corners=True,recompute_scale_factor=True)  # 64

        x_reup2 = F.interpolate(x_down4, scale_factor=2, mode='bilinear',align_corners=True)  # 128
        x_reup = F.interpolate(x_down2, scale_factor=2, mode='bilinear',align_corners=True)  # 256
        Laplace_2 = x_down2 - x_reup2
        Laplace_1 = x - x_reup
        Scale1 = self.Stage1(x_down4)
        Scale2 = self.Stage2(Laplace_2)
        Scale3 = self.Stage3(Laplace_1)
        output1 = Scale1
        output2 = F.interpolate(Scale1, scale_factor=2, mode='bilinear',align_corners=True) + Scale2
        output3 = F.interpolate(output2, scale_factor=2, mode='bilinear',align_corners=True) + Scale3
        print('output1的大小：{}'.format(output1.shape))
        print('output2的大小：{}'.format(output2.shape))
        print('output3的大小：{}'.format(output3.shape))
        print('Scale2的大小：{}'.format(Scale2.shape))
        print('Scale3的大小：{}'.format(Scale3.shape))
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


# # 训练开始时间
# start_time = time.time()
# net=LPB(3,3)
# net2=Generator(3,3)
# # print(net)
# imgs = torch.rand(1, 3, 256, 256)
# #print(net2)
# ceshi_jieguo=net2(imgs)
# print('返回值的长度:{}'.format(len(ceshi_jieguo)))
# print('经过生成器之后的图片大小：{}'.format(ceshi_jieguo[2].shape))
# end_time = time.time()
# print('结束时间:{}'.format(end_time-start_time))