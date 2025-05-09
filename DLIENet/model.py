import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),#参数是padding，使用输入 tensor的反射来填充
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        #torch.nn.InstanceNorm2d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
                        #Applies Instance Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper Instance Normalization: The Missing Ingredient for Fast Stylization .
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]
        #torch.nn.Sequential(*args)
        #A sequential container.模型将按照在构造函数中传递的顺序添加到模型中。 或者，也可以传递模型的有序字典。
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
#=======================================================================Generator
#生成器部分：网络整体上经过一个降采样然后上采样的过程，中间是一系列残差块,数目由实际情况确定，
#根据论文中所说，当输入分辨率为128x128，采用6个残差块，当输入分辨率为256x256甚至更高时，采用9个残差块
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=12):
        super(Generator, self).__init__()

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
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


 #ceshi1= Generator(3)

