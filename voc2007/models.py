import torchvision.models as models
from torchvision.ops import Conv2dNormActivation
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn


class DSDL(nn.Module):
    def __init__(self, model, num_classes, alpha, in_channel=300):
        super(DSDL, self).__init__()
        self.alpha = alpha

        self.features = nn.Sequential(
            *list(model.features.children())[:-1], 
            Conv2dNormActivation(
                in_channels=512, 
                out_channels=2048, 
                kernel_size=1, 
                norm_layer=nn.BatchNorm2d, 
                activation_layer=nn.SiLU
            )
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        self.encoder = nn.Sequential(
            nn.Linear(in_channel, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, in_channel)
        )

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)

        semantic = self.encoder(inp[0])
        res_semantic = self.decoder(semantic)

        score = torch.matmul(
            torch.inverse(torch.matmul(semantic, semantic.transpose(0, 1)) + self.alpha * torch.eye(self.num_classes).cuda()),
            torch.matmul(semantic, feature.transpose(0, 1))).transpose(0, 1)

        return score, inp[0], res_semantic, feature, semantic

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.decoder.parameters(), 'lr': lr},
        ]


def load_model(num_classes, alpha, pretrained=True, in_channel=300):
    model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
    return DSDL(model, num_classes, alpha, in_channel=in_channel)
