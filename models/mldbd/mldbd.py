import torch
import torch.nn as nn

from models.mldbd.base_layers import Base1, Base21, Base22, Base23, Base24, Base31, Base32, Base33, Base34, Output_dbd, \
    Output_edge, BaseConv, Output_class


class BdNet(nn.Module):
    def __init__(self):
        super(BdNet, self).__init__()
        self.encoder = Encoder()
        self.fl = Fl()

        self.decoder_dbd = DecoderDbd()
        self.decoder_boundary = DecoderBoundary()

    def forward(self, x):
        s1, s2, s3 = self.encoder(x)
        fl = self.fl(s3)

        dbd1 = self.decoder_dbd(fl, s3, s2, s1)
        edge1 = self.decoder_boundary(fl, s3, s2, s1)
        return dbd1, edge1


class CdNet(nn.Module):
    def __init__(self):
        super(CdNet, self).__init__()
        self.encoder = Encoder()
        self.fg = Fg()

        self.decoder_dbd = DecoderDbd()
        self.output_class = Output_class()

    def forward(self, x):
        s1, s2, s3 = self.encoder(x)
        fg = self.fg(s3)

        dbd1 = self.decoder_dbd(fg, s3, s2, s1)
        class1 = self.output_class(fg)
        return dbd1, class1


class Stage2Net(nn.Module):
    def __init__(self):
        super(Stage2Net, self).__init__()
        self.encoder = Encoder()
        self.fl = Fl()

        # Isometric Distilling
        self.fgs = Fgs()
        self.aam = AffinityAttentionModule()
        self.ifc = IsometricFeatureCompound()

        self.decoder_dbd = DecoderDbd()
        self.decoder_boundary = DecoderBoundary()

        # Encoders for datasets T_3 and T_4 as well as their following layer
        self.encoder_t3 = Encoder()
        self.fgf = Fg()
        self.encoder_t4 = Encoder()
        self.fgo = Fg()

        # Load checkpoints of CDNet and BDNet
        bdnet = torch.load('checkpoints/3172097_checkpoint_epoch278_BdNet.pth')
        self.load_state_dict(bdnet, strict=False)

        cdnet = torch.load('checkpoints/3172098_checkpoint_epoch278_CdNet.pth')
        for key in list(cdnet.keys()):
            cdnet[key.replace('decoder_dbd.', 'decoder_dbd_noload.').replace('encoder.', 'encoder_t3.').replace('fg.', 'fgf.')] = cdnet.pop(key)
        self.load_state_dict(cdnet, strict=False)
        for key in list(cdnet.keys()):
            cdnet[key.replace('fgf.', 'fgo.').replace('encoder_t3.', 'encoder_t4.')] = cdnet.pop(key)
        self.load_state_dict(cdnet, strict=False)

        # The layers of the CDNet are fixed
        self.encoder_t3.requires_grad_(False)
        self.fgf.requires_grad_(False)
        self.encoder_t4.requires_grad_(False)
        self.fgo.requires_grad_(False)

    def forward(self, x, t3, t4, gt):
        s1, s2, s3 = self.encoder(x)
        fl = self.fl(s3)
        fgs = self.fgs(s3)
        fi = self.aam(fl, fgs)

        _, _, enc_t3 = self.encoder_t3(t3)
        _, _, enc_t4 = self.encoder_t4(t4)

        fgf = self.fgf(enc_t3)
        fgo = self.fgf(enc_t4)

        fc = self.ifc(fgf, fgo, gt)

        dbd = self.decoder_dbd(fi, s3, s2, s1)
        boundary = self.decoder_boundary(fi, s3, s2, s1)
        return dbd, boundary, fgs, fc


class IsometricFeatureCompound(nn.Module):
    def __init__(self):
        super(IsometricFeatureCompound, self).__init__()

    def forward(self, fgf, fgo, gt):
        fgf = nn.functional.interpolate(fgf, scale_factor=16, mode='bilinear', align_corners=True)
        not_gt = 1 - gt
        fgo = nn.functional.interpolate(fgo, scale_factor=16, mode='bilinear', align_corners=True)
        fc = fgf * gt + fgo * not_gt
        return fc


class MldbdInference(nn.Module):

    def __init__(self):
        super(MldbdInference, self).__init__()
        self.encoder = Encoder()
        self.fl = Fl()
        self.fgs = Fgs()
        self.aam = AffinityAttentionModule()

        self.decoder_dbd = DecoderDbd()
        # self.decoder_boundary = DecoderBoundary()

    def forward(self, x):
        s1, s2, s3 = self.encoder(x)
        fl = self.fl(s3)
        fgs = self.fgs(s3)
        fi = self.aam(fl, fgs)

        dbd1 = self.decoder_dbd(fi, s3, s2, s1)
        # edge1 = self.decoder_boundary(fi, s3, s2, s1)
        return dbd1


class Encoder(nn.Module):
    """
    This is the encoder block used by BDNet, CDNet and BBDNet (E_D)
    @return: s1, s2, s3, scaled down versions of the input
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
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
        return s1, s2, s3


class Fl(nn.Module):
    """
    This is the fifth convolutional block used by BDNet (F_l)
    """
    def __init__(self):
        super(Fl, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv5_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv5_1_2(x)
        x = self.conv5_2_2(x)
        x = self.conv5_3_2(x)
        return x


class Fgs(nn.Module):
    """
    This is the fifth convolutional block used for knowledge distillation (F_g^s)
    """
    def __init__(self):
        super(Fgs, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv6_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv6_1_2(x)
        x = self.conv6_2_2(x)
        x = self.conv6_3_2(x)
        return x


class Fg(nn.Module):
    """
    This is the fifth convolutional block used for CDNet (F_g)
    """
    def __init__(self):
        super(Fg, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv11_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv11_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv11_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv11_1_2(x)
        x = self.conv11_2_2(x)
        x = self.conv11_3_2(x)
        return x


class AffinityAttentionModule(nn.Module):
    """
    The Affinity Attention Module (AAM)
    @return: The integrated feature F_i
    """
    def __init__(self):
        super(AffinityAttentionModule, self).__init__()
        self.avgpool = nn.AvgPool2d((20, 20))
        self.conv12_1_2 = BaseConv(1024, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv12_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv12_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv13_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv13_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv13_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, fl, fgs):
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
        return aam


class DecoderDbd(nn.Module):
    """
    This is the decoder of the dbd part in DBNet (D_D)
    """

    def __init__(self):
        super(DecoderDbd, self).__init__()
        self.base21 = Base21()
        self.base22 = Base22()
        self.base23 = Base23()
        self.base24 = Base24()
        self.output_dbd = Output_dbd()

    def forward(self, x, s3, s2, s1):
        x = self.base21(x, s3)
        x = self.base22(s2, x)
        x = self.base23(s1, x)
        x = self.base24(x)
        x = self.output_dbd(x)

        return x


class DecoderBoundary(nn.Module):
    """
    This is the decoder of the boundary part in DBNet (D_B)
    """

    def __init__(self):
        super(DecoderBoundary, self).__init__()
        self.base31 = Base31()
        self.base32 = Base32()
        self.base33 = Base33()
        self.base34 = Base34()
        self.output_edge = Output_edge()

    def forward(self, x, s3, s2, s1):
        x = self.base31(x, s3)
        x = self.base32(s2, x)
        x = self.base33(s1, x)
        x = self.base34(x)
        x = self.output_edge(x)

        return x
