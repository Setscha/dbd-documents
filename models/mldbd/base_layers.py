import torch.nn as nn
import torch


class Output_dbd(nn.Module):
    def __init__(self):
        super(Output_dbd, self).__init__()
        self.conv_out_base_3 = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

    def forward(self, fd4f):
        x = self.conv_out_base_3(fd4f)
        return x


class Output_edge(nn.Module):
    def __init__(self):
        super(Output_edge, self).__init__()
        self.conv_out_base_3 = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

    def forward(self, fe4):
        x = self.conv_out_base_3(fe4)
        return x


class Output_class(nn.Module):
    def __init__(self):
        super(Output_class, self).__init__()
        self.avgpool = nn.AvgPool2d((20,20))
        self.conv_fc_1 = BaseConv(512, 256, 1, 1, activation=None, use_bn=False)
        self.conv_fc_2 = BaseConv(256, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)
        # TODO: Without activation function like originally, the values can be negative, which does not make sense for probabilities



    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv_fc_1(x)
        x = self.conv_fc_2(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        class1 = x
        # print(class1.size())

        return class1


class Base1(nn.Module):
    """
    This class is the Encoder (E_D) + F_l + F_g^s + Affinity Attention Module (AAM)
    """
    # It seems like these components cannot be separated because they all use the same maxpool layer, meaning that each
    # maxpool used for downscaling share the same weights and biases (is this wanted?)
    def __init__(self):
        super(Base1, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d((20, 20))
        self.conv1_1_2 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1_2 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1_2 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1_2 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_1_2 = BaseConv(1024, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv8_1_2 = BaseConv(1024, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv8_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv8_3_2 = BaseConv(512, 1, 3, 1, activation=nn.Sigmoid(), use_bn=False)
        self.conv9_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv9_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv9_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv10_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv10_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv10_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv11_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv11_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv11_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv12_1_2 = BaseConv(1024, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv12_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv12_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv13_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv13_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv13_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, x):
        x = self.conv1_1_2(x)
        x = self.conv1_2_2(x)
        x = self.maxpool(x)
        x = self.conv2_1_2(x)
        x = self.conv2_2_2(x)
        s1 = x
        x = self.maxpool(x)
        x = self.conv3_1_2(x)
        x = self.conv3_2_2(x)
        x = self.conv3_3_2(x)
        s2 = x

        x = self.maxpool(x)
        x = self.conv4_1_2(x)
        x = self.conv4_2_2(x)
        x = self.conv4_3_2(x)
        s3 = x
        x = self.maxpool(x)
        x = self.conv5_1_2(x)
        x = self.conv5_2_2(x)
        x = self.conv5_3_2(x)
        fl = x

        x = s3
        x = self.maxpool(x)
        x = self.conv6_1_2(x)
        x = self.conv6_2_2(x)
        x = self.conv6_3_2(x)
        fgs = x

        channel1 = torch.cat([fgs, fl], 1)
        channel1 = self.conv12_1_2(channel1)
        channel1 = self.conv12_2_2(channel1)
        channel1 = self.conv12_3_2(channel1)
        channel1 = self.avgpool(channel1)
        fch = torch.mul(fgs, channel1)
        fch = self.conv13_1_2(fch)
        fch = self.conv13_2_2(fch)
        fch = self.conv13_3_2(fch)
        aam = torch.add(fl, fch)
        return s1, s2, s3, fl, fgs, aam


class Base21(nn.Module):
    def __init__(self):
        super(Base21, self).__init__()
        self.conv1_2 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, s4, s3):
        x = s4
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s3], 1)
        x = self.conv1_2(x)
        x = self.conv2_2(x)
        fd1 = x

        return fd1


class Base31(nn.Module):
    def __init__(self):
        super(Base31, self).__init__()
        self.conv1_2 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        # self.conv1_2 = BaseConv(512, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, s4, s3):
        # def forward(self, s4):
        x = s4
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s3], 1)
        x = self.conv1_2(x)
        x = self.conv2_2(x)
        fe1 = x

        return fe1


class Base22(nn.Module):
    def __init__(self):
        super(Base22, self).__init__()
        self.conv3_2 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, s2, fd1):
        x = fd1
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s2], 1)
        x = self.conv3_2(x)
        x = self.conv4_2(x)
        fd2 = x

        return fd2


class Base32(nn.Module):
    def __init__(self):
        super(Base32, self).__init__()
        self.conv3_2 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, s2, fe1f):
        x = fe1f
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s2], 1)
        x = self.conv3_2(x)
        x = self.conv4_2(x)
        fe2 = x

        return fe2


class Base23(nn.Module):
    def __init__(self):
        super(Base23, self).__init__()
        self.conv5_2 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_2 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, s1, fd2f):
        x = fd2f
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s1], 1)
        x = self.conv5_2(x)
        x = self.conv6_2(x)
        x = self.conv7_2(x)
        fd3 = x

        return fd3


class Base33(nn.Module):
    def __init__(self):
        super(Base33, self).__init__()
        self.conv5_2 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_2 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, s1, fe2):
        x = fe2
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s1], 1)
        x = self.conv5_2(x)
        x = self.conv6_2(x)
        x = self.conv7_2(x)
        fe3 = x

        return fe3


class Base24(nn.Module):
    def __init__(self):
        super(Base24, self).__init__()
        self.conv_out_base_1 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_base_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, fd3):
        x = fd3
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv_out_base_1(x)
        x = self.conv_out_base_2(x)
        fd4 = x

        return fd4


class Base34(nn.Module):
    def __init__(self):
        super(Base34, self).__init__()
        self.conv_out_base_1 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_base_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, fe3f):
        x = fe3f
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv_out_base_1(x)
        x = self.conv_out_base_2(x)
        fe4 = x

        return fe4


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)

        return x
