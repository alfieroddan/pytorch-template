#normal imports 
import torchvision
from torch import nn
from typing import List, Tuple, Dict, Optional

class ResNetGrad(nn.Module):
    def __init__(self, in_channels, out_channels, pretrained):
        super(ResNetGrad, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # model
        self.res =  torchvision.models.resnet18(pretrained=pretrained)

        self.res.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), 
                          padding=(3, 3), bias=False)
        self.res.fc = nn.Linear(in_features=512, out_features=self.out_channels, bias=True)

        # gradients
        self.gradients = None

def AlexNet(in_channels, out_channels, pretrained=False):
    alex = torchvision.models.alexnet(pretrained=pretrained)
    alex.features[0] = nn.Conv2d(in_channels, 64, 
                                 kernel_size=(1,1), stride=(1,1))
    alex.classifier = nn.Sequential(
                    nn.Dropout(p=0.5, inplace=False),
                    nn.Linear(in_features=9216, out_features=4096, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5, inplace=False),
                    nn.Linear(in_features=4096, out_features=4096, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_features=4096, out_features=out_channels, bias=True),
                )
    return(alex)


def ResNet(in_channels, out_channels, pretrained=False):
    res = torchvision.models.resnet18(pretrained=pretrained)
    res.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), 
                          padding=(3, 3), bias=False)
    res.fc = nn.Linear(in_features=512, out_features=out_channels, bias=True)
    return(res)


def VGG(in_channels, out_channels, pretrained=False):
    vgg = torchvision.models.vgg16(pretrained=pretrained)
    vgg.features[0] = nn.Conv2d(in_channels, 64, 
                                 kernel_size=(3, 3), 
                                 stride=(1,1), 
                                 padding=(1, 1))
    vgg.classifier[6] = nn.Linear(in_features=4096, out_features=out_channels, bias=True)
    return(vgg)


def DenseNet(in_channels, out_channels, pretrained=False):
    dense = torchvision.models.densenet161(pretrained=pretrained)
    dense.features.conv0 = nn.Conv2d(in_channels, 96, 
                                 kernel_size=(7, 7), 
                                 stride=(2, 2), 
                                 padding=(3, 3), bias=False)
    dense.classifier = nn.Linear(in_features=2208, out_features=out_channels, bias=True)
    return(dense)

def SqueezeNet(in_channels, out_channels, pretrained=False):
    squeeze = torchvision.models.squeezenet1_0(pretrained=False)
    squeeze.features[0] = nn.Conv2d(in_channels, 96, kernel_size=(7, 7), stride=(2, 2))
    squeeze.classifier.add_module('flatten', nn.Flatten())
    squeeze.classifier.add_module('fc', nn.Linear(1000, out_channels))
    return(squeeze)

def MobileNetV2(in_channels, out_channels, pretrained=False):
    mobile = torchvision.models.mobilenet_v2(pretrained=pretrained)
    mobile.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    mobile.classifier[1] = nn.Linear(in_features=1280, out_features=out_channels, bias=True)
    return(mobile)

def EfficientNetb7(in_channels, out_channels, pretrained=False):
    eff = torchvision.models.efficientnet_b7(pretrained=pretrained)
    eff.features[0][0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    eff.classifier[1] = nn.Linear(in_features=2560, out_features=out_channels, bias=True)
    return(eff)

def GoogleNet(in_channels, out_channels, pretrained=False):
    google = torchvision.models.googlenet(pretrained=pretrained)
    google.conv1.conv = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    google.fc = nn.Linear(in_features=1024, out_features=out_channels, bias=True)
    return(google)

def model(config):
    f = globals().get(config['model']['name'])
    return f(**config['model']['params'])


if __name__ == '__main__':
    print(globals())
