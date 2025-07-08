import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from models.PGGNet.gcn import GraphConvolution
from models.PGGNet.attention import Channel_Shuffle


class GCN(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(in_channels, in_channels)
        self.gc2 = GraphConvolution(in_channels, hidden)
        self.gc3 = GraphConvolution(hidden, out_channels)
        self.relu = F.relu
        self.Fdropout = F.dropout
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.relu(self.gc1(x, adj))
        x = self.Fdropout(x, self.dropout, training=self.training)
        x = self.relu(self.gc2(x, adj))
        x = self.Fdropout(x, self.dropout, training=self.training)
        x = self.relu(self.gc3(x, adj))
        x = self.Fdropout(x, self.dropout, training=self.training)
        return x


class Dilation_conv(nn.Sequential):
    """
    空洞卷积模块的定义
    """

    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(Dilation_conv, self).__init__(*modules)

class Pooling(nn.Sequential):
    """
    pooling层
    """

    def __init__(self, in_channels, out_channels):  # [in_channel=out_channel=256]
        super(Pooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # [256*1*1]
            # 自适应平均池化层，只需要给定输出的特征图的尺寸(括号内数字)就好了
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(Pooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class Dilation_block(nn.Module):

    def __init__(self,in_channels, out_channels, rates):
        super(Dilation_block, self).__init__()
        block = []
        for rate in rates:
            block.append(Dilation_conv(in_channels, out_channels, rate))
        self.final = nn.ModuleList(block)

    def forward(self, x):
        for m in self.final:
            x = m(x)
        return x


class Dilation_module_dense(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dilation_module_dense, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1)
        self.bn_in = nn.BatchNorm2d(out_channels // 4)
        self.relu_in = nn.ReLU(inplace=True)

        self.dilation_conv1 = nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.dilation_conv2_1 = nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1, dilation=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(out_channels // 4)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.dilation_conv2_2 = nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=2, dilation=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(out_channels // 4)
        self.relu2_2 = nn.ReLU(inplace=True)

        self.dilation_conv3_1 = nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1, dilation=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(out_channels // 4)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.dilation_conv3_2 = nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=2, dilation=2, bias=False)
        self.bn3_2 = nn.BatchNorm2d(out_channels // 4)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.dilation_conv3_3 = nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=3, dilation=3, bias=False)
        self.bn3_3 = nn.BatchNorm2d(out_channels // 4)
        self.relu3_3 = nn.ReLU(inplace=True)

        self.pool = Pooling(out_channels // 4, out_channels // 4)

        self.channelshuffle = Channel_Shuffle(num_groups=out_channels)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)

        out1 = self.dilation_conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)

        out2_1 = self.dilation_conv2_1(x+out1)
        out2_1 = self.bn2_1(out2_1)
        out2_1 = self.relu2_1(out2_1)
        out2_2 = self.dilation_conv2_2(x+out2_1)
        out2_2 = self.bn2_2(out2_2)
        out2_2 = self.relu2_2(out2_2)

        out3_1 = self.dilation_conv3_1(x+out1+out2_1)
        out3_1 = self.bn3_1(out3_1)
        out3_1 = self.relu3_1(out3_1)
        out3_2 = self.dilation_conv3_2(x+out3_1+out2_2)
        out3_2 = self.bn3_2(out3_2)
        out3_2 = self.relu3_2(out3_2)
        out3_3 = self.dilation_conv3_3(x+out3_2)
        out3_3 = self.bn3_3(out3_3)
        out3_3 = self.relu3_3(out3_3)

        out4 = self.pool(x)

        out = torch.cat([out1,out2_1+out2_2,out3_1+out3_2+out3_3,out4],1)
        out = self.channelshuffle(out)

        return out


class encoder_without_pyramid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder_without_pyramid, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.down_conv(x)
        return out

class gen_adj(nn.Module):
    def __init__(self, in_channels):
        super(gen_adj, self).__init__()
        if in_channels == 1:
            self.adj_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.adj_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels // 8),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 8, in_channels // 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels // 16),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 16, 1, kernel_size=3, padding=1),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True)
            )
        self.projection = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # LBP = circular_LBP(x)
        # x = x * LBP

        x = self.adj_conv(x)
        out_adj_a = x.view(b, 1, -1)
        out_adj_b = x.view(b, -1, 1)
        out_adj = torch.bmm(out_adj_b, out_adj_a)
        out_adj = torch.unsqueeze(out_adj, 1)
        out_adj = self.projection(out_adj)
        out_adj = torch.squeeze(out_adj, 1)
        return out_adj

class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.down_conv = Dilation_module_dense(in_channels,out_channels)
        # ceil_mode参数取整的时候向上取整，该参数默认为False表示取整的时候向下取整
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        out = self.down_conv(x)
        out_pool = self.pool(out)
        return out, out_pool

class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        # 反卷积
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_copy, x, interpolate=True):
        out = self.up(x)
        if interpolate:
            # 迭代代替填充， 取得更好的结果
            out = F.interpolate(out, size=(x_copy.size(2), x_copy.size(3)),
                                mode="bilinear", align_corners=True
                                )
        else:
            # 如果填充物体积大小不同
            diffY = x_copy.size()[2] - x.size()[2]
            diffX = x_copy.size()[3] - x.size()[3]
            out = F.pad(out, (diffX // 2, diffX - diffX // 2, diffY, diffY - diffY // 2))
        # 连接
        out = torch.cat([x_copy, out], dim=1)
        out_conv = self.up_conv(out)
        return out_conv

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parametersL {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f"\nNbr of trainable parameters: {nbr_params}"


class PGG(BaseModel):
    def __init__(self, num_classes, in_channels=2, freeze_bn=False, **_):
        super(PGGNet, self).__init__()

        #unet编码
        self.down1 = encoder(in_channels, 64)
        self.down2 = encoder(64, 128)
        self.down3 = encoder(128, 256)
        self.down4 = encoder(256, 512)
        self.middle_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        #gcn编码
        self.gcn_encoder=GCN(1024,2048,1024,0.5)
        self.adj_encoder=gen_adj(1024)

        #unt解码
        self.up1 = decoder(1024, 512)
        self.up2 = decoder(512, 256)
        self.up3 = decoder(256, 128)
        self.up4 = decoder(128, 64)

        #全连接
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        ##priori knowledge
        self.p_down1 = encoder(3, 64)
        self.p_down2 = encoder(64, 128)
        self.p_down3 = encoder(128, 256)
        self.p_down4 = encoder(256, 512)
        self.p_middle_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.p_up1 = decoder(1024, 512)
        self.p_up2 = decoder(512, 256)
        self.p_up3 = decoder(256, 128)
        self.p_up4 = decoder(128, 64)

        self.p_final_conv = nn.Conv2d(64, 3, kernel_size=1)

        self._initalize_weights()
        if freeze_bn:
            self.freeze_bn()

    def _initalize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x,y):

        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        x = self.middle_conv(x)

        y1, y = self.p_down1(y)
        y2, y = self.p_down2(y)
        y3, y = self.p_down3(y)
        y4, y = self.p_down4(y)
        y = self.p_middle_conv(y)

        adj_en =self.adj_encoder(x+y)
        gcn_out=self.gcn_encoder(x+y,adj_en)

        z = self.up1(x4+y4, gcn_out)
        z = self.up2(x3+y3, z)
        z = self.up3(x2+y2, z)
        z = self.up4(x1+y1, z)
        z = self.final_conv(z)

        p_z = self.p_up1(x4+y4, gcn_out)
        p_z = self.p_up2(x3+y3, p_z)
        p_z = self.p_up3(x2+y2, p_z)
        p_z = self.p_up4(x1+y1, p_z)
        p_z = self.p_final_conv(p_z)

        return z, p_z

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def PGGNet(num_classes):
    model = PGG(num_classes=num_classes)
    return model
