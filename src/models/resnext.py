# -*- coding: utf-8 -*-
# Copyright 2019 Christopher Kümmel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torchvision.models as models


def load_resnext50_32x4d(pretrained=True, remove_last_layer=False):
    """load an existing rexnext model

    Args:
        pretrained: Whether you want the model to be pretrained. (default True)
        remove_last_layer: Whether you want to remove the output layer. (default False) 
            Output size with last layer = 1000
            Output size without last layer = 2048

    Returns: torch.nn.model
    """

    model = models.resnext50_32x4d(pretrained=pretrained)

    if remove_last_layer:
        model = nn.Sequential(*list(model.children())[:-1])

    return model
