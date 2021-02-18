from torchvision import models
from torchvision.models import utils
import torch.nn as nn
import torch

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class AlexnetModel(models.AlexNet):
    def __init__(self, num_classes=1000):
        super(AlexnetModel, self).__init__(num_classes)

    def forward(self, x, fourierColors):
        x = self.features(x)
        x = self.avgpool(x)
        fc = torch.flatten(fourierColors, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = torch.cat((x, fc), dim=1)
        x = self.classifier2(x)
        return x


def alexnetmodel(num_classes=4):
    model = AlexnetModel()
    state_dict = utils.load_state_dict_from_url("https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth")
    model.load_state_dict(state_dict)
    model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
    model.classifier2 = nn.Sequential(
        nn.Linear(4096 + 300, 4096 + 300),
        nn.ReLU(inplace=True),
        nn.Linear(4096 + 300, 4),
        nn.Softmax(1)
    )

    return model


from torchvision.models.vgg import make_layers


class VggModel(models.VGG):
    def __init__(self, features, num_classes=1000):
        super(VggModel, self).__init__(features, num_classes=num_classes, init_weights=False)

    def forward(self, x, fourierColors):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        fc = torch.flatten(fourierColors, 1)
        x = self.classifier(x)
        x = torch.cat((x, fc), dim=1)
        x = self.classifier2(x)
        return x


def vgg_create(pretrained=True):
    model = VggModel(make_layers(cfgs['D'], batch_norm=False))
    if pretrained:
        state_dict = utils.load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth",
                                                    progress=True)
        model.load_state_dict(state_dict)
    model.classifier = nn.Sequential(*[model.classifier[i] for i in range(3)])
    model.classifier2 = nn.Sequential(
        nn.Linear(4096 + 300, 4096 + 300),
        nn.ReLU(inplace=True),
        nn.Linear(4096 + 300, 4),
        nn.Softmax(1)
    )
    return model


class AlexnetModel_EF(models.AlexNet):
    def __init__(self, num_classes=1000):
        super(AlexnetModel_EF, self).__init__(num_classes)

    def forward(self, x, fourierColors):
        x = self.features(x)
        x = self.avgpool(x)
        fc = torch.flatten(fourierColors, 1)
        x = torch.flatten(x, 1)
        x = torch.cat((x, fc), dim=1)
        x = self.classifier(x)
        return x


def alex_early_create(pretrained=True):
    model = AlexnetModel_EF()
    state_dict = utils.load_state_dict_from_url("https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth")
    model.load_state_dict(state_dict)
    model.classifier[1] = nn.Linear(9216 + 300, 4096)
    model.classifier[6] = nn.Sequential(nn.Linear(4096, 4), nn.Softmax(1))
    return model


class VggModel_EF(models.VGG):
    def __init__(self, features, num_classes=1000):
        super(VggModel_EF, self).__init__(features, num_classes=num_classes, init_weights=False)

    def forward(self, x, fourierColors):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        fc = torch.flatten(fourierColors, 1)
        x = torch.cat((x, fc), dim=1)
        x = self.classifier(x)
        return x


def vgg_create_early(pretrained=True):
    model = VggModel_EF(make_layers(cfgs['D'], batch_norm=False))
    if pretrained:
        state_dict = utils.load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth",
                                                    progress=True)
        model.load_state_dict(state_dict)

    model.classifier[0] = nn.Linear(25088 + 300, 4096)
    model.classifier[6] = nn.Sequential(nn.Linear(4096, 4), nn.Softmax(1))
    return model


def alexnet_normal(pretrained=True):
    model = models.alexnet(pretrained=True)
    model.classifier[6] = nn.Sequential(nn.Linear(4096, 4), nn.Softmax(1))
    return model


def vgg16_normal(pretrained=True):
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Sequential(nn.Linear(4096, 4), nn.Softmax(1))
    return model
