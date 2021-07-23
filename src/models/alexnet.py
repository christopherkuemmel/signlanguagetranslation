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

def load_alexnet(pretrained=True, remove_last_layer=False):
    """load an existing alexnet model

    Args:
        pretrained: Whether you want the model to be pretrained. (default True)
        remove_last_layer: Whether you want to remove the output layer. (default False)    

    Returns: torch.nn.model
    """
    if pretrained:
        model = models.alexnet(pretrained=True)
    else:
        model = models.alexnet()
    
    if remove_last_layer:
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])

    return model
