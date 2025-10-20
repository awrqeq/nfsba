import torch
import torchvision.models as models

def load_victim_model(name, num_classes, pretrained=True):
    print(f"Loading victim model: {name} (pretrained={pretrained})")
    if name == 'ResNet18':
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes) # Adapt final layer
    elif name == 'VGG16':
        model = models.vgg16(pretrained=pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
    # Add other models like EfficientNet, DenseNet if needed
    # elif name == 'EfficientNetB0':
    #     model = models.efficientnet_b0(pretrained=pretrained)
    #     num_ftrs = model.classifier[1].in_features
    #     model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unknown victim model name: {name}")
    return model