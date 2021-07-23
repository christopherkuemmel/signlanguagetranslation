# -*- coding: utf-8 -*-
"""3D ResNext 101 Kinectics implementation.

This is a modified and slightly extended version of
https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnext.py
"""

import logging
import math
import os
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

from utils.download import download_file_from_google_drive

logger = logging.getLogger(__name__)


# Source https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnext.py
class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# Source https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnext.py
class ResNeXt(nn.Module):
    def __init__(self, block, layers, sample_size, sample_duration, shortcut_type='B', cardinality=32, num_classes=400):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                                           nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnext101_3d(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def load_resnext101_3d_kinetics(image_size: int, window_size: int, cut_off_layer: int = None):
    """Load a ResNext101 model pretrained on the Kinetics dataset.

    Args:
        image_size (int): The dimension of the images.
        window_size (int): The dimension of the sliding depth window.
        cut_off_layer (int): How many layer to cut off the model.
            Number specifies how many layer should be trimmed.
            Model layer order:
                -1: `fc`      - Size(1, 400)
                -2: `avgpool` - Size(1, 2048, 1, 1, 1)
                -3: `layer_4` - Size(1, 1024, x, y, y) 
                              - 3 ResNeXtBottleneck blocks
                -4: `layer_3` - Size(1, 512, x, y, y) 
                              - 23 ResNeXtBottleneck blocks
                -5: `layer_2` - Size(1, 256, x, y, y) 
                              - 4 ResNeXtBottleneck blocks
                -6: `layer_1` - Size(1, 128, x, y, y) 
                              - 3 ResNeXtBottleneck blocks
                
                x, y depending on window_size and image_size respectivelys

            Example:
                cut_off_layer = 3 will return
                -4: `layer_3` - 23 ResNeXtBottleneck blocks
                -5: `layer_2` - 4 ResNeXtBottleneck blocks
                -6: `layer_1` - 3 ResNeXtBottleneck blocks
                effecively remove the fc, the avgpool and the last layer block

    Returns:
        torch.nn.model: The pre-trained 3D ResNext 101 model.
    """

    # * Download the pretrained resnext-101-kinetics.pth model file: https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M (27th feb 2020) and store it in the ./model folder
    # * Source: https://github.com/kenshohara/3D-ResNets-PyTorch#pre-trained-models
    # * Description: 3D ResNext101 model pretrained on the Kinetics dataset.
    base_path = './model/'
    model_name = 'resnext_101_kinetics.pth'

    if not os.path.exists(base_path):
        os.makedirs(base_path)
    if not os.path.exists(base_path + model_name):
        try:
            token = '1cULocPe5YvPGWU4tV5t6fC9rJdGLfkWe'  # kinetics
            download_file_from_google_drive(token, base_path + model_name)
        except:
            logger.warning("""Could not download and save the ResNext101_3D_kinetics model.
            You may want to download it yourself from this link: https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M
            The model should be saved within the projects root directory ./model/resnext_101_kinetics.pth""")
    else:
        logger.info('Found existing model, continue using cache.')

    # Create the ResNext model
    # sample_size and -_duration are irrelevant here since they will be overwritten by the state dict
    model = resnext101_3d(sample_size=image_size, sample_duration=window_size)
    # model = resnext101_3d(sample_size=0, sample_duration=0)
    state_dict = torch.load(base_path + model_name, map_location=torch.device('cpu'))['state_dict']

    # Remove module. from state_dict (probably added by saving state_dict with torch.nn.DataParallel)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    if cut_off_layer:
        model = nn.Sequential(*list(model.children())[:-cut_off_layer])

    return model
