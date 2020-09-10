import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from nets.pooling import L2N, GeM, RMAC
import numpy as np
from collections import OrderedDict


class BaseModel(nn.Module):
    def __str__(self):
        return self.__class__.__name__

    def summary(self, input_size, batch_size=-1, device="cuda"):
        try:
            return summary(self, input_size, batch_size, device)
        except:
            return self.__repr__()


class MobileNet(BaseModel):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.base = models.mobilenet_v2(pretrained=True)

    def forward(self, x):
        return self.base(x)


class MobileNet_RMAC(BaseModel):
    def __init__(self):
        super(MobileNet_RMAC, self).__init__()
        self.base = nn.Sequential(
            OrderedDict([*list(models.mobilenet_v2(pretrained=True).features.named_children())]))
        self.pool = RMAC()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x


class MobileNet_AVG(BaseModel):
    def __init__(self):
        super(MobileNet_AVG, self).__init__()
        self.base = nn.Sequential(
            OrderedDict([*list(models.mobilenet_v2(pretrained=True).features.named_children())]))
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.norm(x)
        return x


class MobileNet_GeM(BaseModel):
    def __init__(self):
        super(MobileNet_GeM, self).__init__()
        self.base = nn.Sequential(
            OrderedDict([*list(models.mobilenet_v2(pretrained=True).features.named_children())]))
        self.pool = GeM()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x


class DenseNet(BaseModel):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.base = models.densenet121(pretrained=True)

    def forward(self, x):
        return self.base(x)

    def summary(self, input_size, batch_size=-1, device="cuda"):
        try:
            return super().summary(input_size, batch_size, device)
        except:
            return nn.Module.__repr__()


class DenseNet_RMAC(BaseModel):
    def __init__(self):
        super(DenseNet_RMAC, self).__init__()

        self.base = nn.Sequential(*list(models.densenet121(pretrained=True).features.children()),
                                  nn.ReLU(inplace=True))
        self.pool = RMAC()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x


class DenseNet_AVG(BaseModel):
    def __init__(self):
        super(DenseNet_AVG, self).__init__()

        self.base = nn.Sequential(*list(models.densenet121(pretrained=True).features.children()),
                                  nn.ReLU(inplace=True))
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.norm(x)
        return x


class DenseNet_GeM(BaseModel):
    def __init__(self):
        super(DenseNet_GeM, self).__init__()
        self.base = nn.Sequential(OrderedDict([*list(models.densenet121(pretrained=True).features.named_children())] +
                                              [('relu', nn.ReLU(inplace=True))]))
        self.pool = GeM()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x


class Resnet50_RMAC(BaseModel):
    def __init__(self):
        super(Resnet50_RMAC, self).__init__()
        self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-2]))

        self.pool = RMAC()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x


class Resnet50_AVG(BaseModel):
    def __init__(self):
        super(Resnet50_AVG, self).__init__()

        self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-2]))
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.norm(x)
        return x


class TripletNet(BaseModel):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, *x, single=False):
        if single:
            return self.forward_single(x[0])
        else:
            return self.forward_triple(x[0], x[1], x[2])

    def forward_single(self, x):
        output = self.embedding_net(x)
        return output

    def forward_triple(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

    def __str__(self):
        return f'[{super(TripletNet, self).__str__()}]{self.embedding_net.__str__()}'


class Segment_Maxpooling(BaseModel):
    def __init__(self):
        super(Segment_Maxpooling, self).__init__()
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.norm = L2N()

    def forward(self, x):
        x = self.pool(x).squeeze(-1)
        x = self.norm(x)
        return x
