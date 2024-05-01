import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import warnings

warnings.filterwarnings('ignore')


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        for n, m in self.resnet.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.resnet.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

    def forward(self, x):
        x0 = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))
        xm = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(xm)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        return [x0, x1, x2, x3, x4]


class LGAE(nn.Module):
    def __init__(self, channels=128, r=4):
        super(LGAE, self).__init__()
        inner_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inner_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inner_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lattn_feats = self.local_att(x)
        gattn_feats = self.global_att(x)
        w = self.sigmoid(lattn_feats + gattn_feats)
        return x * w


class LGAA(LGAE):
    def forward(self, x1, x2):
        addition_feats = x1 + x2
        lattn_feats = self.local_att(addition_feats)
        gattn_feats = self.global_att(addition_feats)
        w = self.sigmoid(lattn_feats + gattn_feats)
        return 2 * x1 * w + 2 * x2 * (1 - w)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv2d(256, 128, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.lgaa1 = LGAA(channels=256)
        self.lgaa2 = LGAA(channels=128)

    def forward(self, x):
        x0, x1, x2, x3, x4 = x
        x4 = self.block1(x4)
        x4 = self.lgaa1(x4, x3)
        x4 = self.block2(x4)
        x4 = self.lgaa2(x4, x2)
        return x0, x1, x4


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.block = self._make_layers(in_channels, out_channels)

    def _make_layers(self, in_channels, out_channels):
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(ResBlock(in_channels, out_channels, downsample))
        layers.append(ResBlock(out_channels, out_channels))
        layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 2, dilation=2, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class SSDecoder(nn.Module):
    def __init__(self):
        super(SSDecoder, self).__init__()
        self.block1 = DecoderBlock(128 + 64, 128)
        self.block2 = DecoderBlock(128 + 64, 128)
        self.lgae = LGAE(channels=128)

    def forward(self, x0, x1, x4):
        x = F.upsample(x4, x1.shape[2:], mode='bilinear')
        x = torch.cat([x, x1], 1)
        x1 = self.block1(x)
        x = F.upsample(x1, x0.shape[2:], mode='bilinear')
        x = torch.cat([x, x0], 1)
        x0 = self.block2(x)
        x4 = self.lgae(x4)
        return x0, x1, x4


class CRDecoder(nn.Module):
    def __init__(self):
        super(CRDecoder, self).__init__()
        self.block1 = DecoderBlock(256, 128)
        self.block2 = DecoderBlock(256, 128)
        self.block3 = DecoderBlock(256, 128)
        self.ctfa = CTFA(in_channels=128)

    def forward(self, x0, x1, x1_c, x2_c):
        change_feats = self.ctfa(x1_c, x2_c)
        x = self.block1(change_feats)
        x = F.upsample(x, x1.shape[2:], mode='bilinear')
        x1 = torch.cat([x, x1], 1)
        x = self.block2(x1)
        x = F.upsample(x, x0.shape[2:], mode='bilinear')
        x = torch.cat([x, x0], 1)
        x = self.block3(x)
        return x


class CTFA(nn.Module):
    def __init__(self, in_channels):
        super(CTFA, self).__init__()
        self.height = 2
        inner_channels = max(int(in_channels / 4), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, inner_channels, 1, padding=0, bias=False),
                                     nn.LeakyReLU(0.2))
        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(inner_channels, in_channels, kernel_size=1, stride=1, bias=False))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        abs_feats = torch.abs(x1 - x2)
        batch_size = x1.shape[0]
        channels = x1.shape[1]

        inp_feats_ = torch.cat([x1, x2], dim=1)
        inp_feats_ = inp_feats_.view(batch_size, self.height, channels, inp_feats_.shape[2], inp_feats_.shape[3])

        feats_U = torch.sum(inp_feats_, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, channels, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        attn_feats = torch.sum(inp_feats_ * attention_vectors, dim=1)

        return torch.cat([attn_feats, abs_feats], dim=1)


class SCDNet(nn.Module):
    def __init__(self, channels=3, num_classes=7):
        super(SCDNet, self).__init__()
        self.backbone = Backbone()
        self.encoder = Encoder()
        self.ss_decoder = SSDecoder()
        self.cr_decoder = CRDecoder()
        self.ss_classifier1 = nn.Sequential(DecoderBlock(256, 128),
                                            nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                            nn.Conv2d(64, num_classes, kernel_size=1))
        self.ss_classifier2 = nn.Sequential(DecoderBlock(256, 128),
                                            nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                            nn.Conv2d(64, num_classes, kernel_size=1))
        self.cr_classifier = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                           nn.Conv2d(64, 1, kernel_size=1))
        initialize_weights(self.encoder,
                           self.ss_decoder, self.cr_decoder,
                           self.ss_classifier1, self.ss_classifier2, self.cr_classifier)

    def forward(self, x1, x2):
        reshape_size = x1.size()[2:]

        x1 = self.backbone(x1)
        x2 = self.backbone(x2)

        x1_0, x1_1, x1_4 = self.encoder(x1)
        x2_0, x2_1, x2_4 = self.encoder(x2)

        x1_0, x1_1, x1_4 = self.ss_decoder(x1_0, x1_1, x1_4)
        x2_0, x2_1, x2_4 = self.ss_decoder(x2_0, x2_1, x2_4)

        change_feats = self.cr_decoder(torch.abs(x1_0 - x2_0), torch.abs(x1_1 - x2_1), x1_4, x2_4)
        change = self.cr_classifier(change_feats)

        out1 = self.ss_classifier1(torch.cat([x1_0, change_feats], 1))
        out2 = self.ss_classifier2(torch.cat([x2_0, change_feats], 1))

        return F.upsample(change, reshape_size, mode='bilinear'), \
               F.upsample(out1, reshape_size, mode='bilinear'), \
               F.upsample(out2, reshape_size, mode='bilinear')
