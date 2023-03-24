# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import random
import os

from opacus.validators.module_validator import ModuleValidator
from torchvision import models
from torchvision.models.squeezenet import SqueezeNet, squeezenet1_0


def set_random_seed(seed_value, use_cuda: bool = True) -> None:
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    os.environ["PYTHONHASHSEED"] = str(seed_value)  # Python hash buildin
    if use_cuda:
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def get_cnn_model(num_classes, model_type, pretrained, resnet_size=18, in_channels=3):
    """Creates global model architecture."""
    if model_type == "squeezenet1_0":
        model = (
            squeezenet1_0(pretrained=True)
            if pretrained
            else SqueezeNet(num_classes=num_classes)
        )
    elif "resnet" in model_type:
        model = Resnet(
            num_classes=num_classes, resnet_size=resnet_size, pretrained=pretrained
        )
        if in_channels == 1:
            # Change first convolution layer size to accommodate 1 channel
            model.backbone.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
    elif model_type == "cnn":
        model = SimpleConvNet(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_type} not found")
    return model


class Resnet(nn.Module):
    """RESNET model with BatchNorm replaced with GroupNorm

    """

    def __init__(self, num_classes, resnet_size, pretrained=False):
        super().__init__()

        # Retrieve resnet of appropriate size
        resnet = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }
        assert (
            resnet_size in resnet.keys()
        ), f"Resnet size {resnet_size} is not supported!"

        self._name = f"Resnet{resnet_size}"
        self.backbone = resnet[resnet_size]()

        if pretrained:
            self.backbone.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    models.resnet.model_urls[f"resnet{resnet_size}"],
                    progress=True,
                )
            )

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        # Replace batch norm with group norm
        self.backbone = ModuleValidator.fix(self.backbone)

    def forward(self, x):
        return self.backbone(x)

    def name(self):
        return self._name


class SimpleConvNet(nn.Module):
    r"""
    Simple CNN model following architecture from
    https://github.com/TalwalkarLab/leaf/blob/master/models/celeba/cnn.py#L19
    and https://arxiv.org/pdf/1903.03934.pdf
    """

    def __init__(self, in_channels, num_classes, dropout_rate=0):
        super(SimpleConvNet, self).__init__()
        self.out_channels = 32
        self.stride = 1
        self.padding = 2
        self.layers = []
        in_dim = in_channels
        for _ in range(4):
            self.layers.append(
                nn.Conv2d(in_dim, self.out_channels,
                          3, self.stride, self.padding)
            )
            in_dim = self.out_channels
        self.layers = nn.ModuleList(self.layers)

        self.gn_relu = nn.Sequential(
            nn.GroupNorm(self.out_channels, self.out_channels, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        num_features = (
            self.out_channels
            * (self.stride + self.padding)
            * (self.stride + self.padding)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        for conv in self.layers:
            x = self.gn_relu(conv(x))

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(self.dropout(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
