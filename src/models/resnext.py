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
