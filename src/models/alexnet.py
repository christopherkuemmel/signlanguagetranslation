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
