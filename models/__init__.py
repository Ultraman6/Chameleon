import os
import torch
# from torch import nn
# from torch.nn import init
import models.cnn as cnn
import models.vgg as vgg
import models.resnet as resnet
import models.wide_resnet as wide_resnet
import models.densenet as desnet
import models.pyramidnet as pyramidnet
import models.deit as deit

# map between model name and function
models = {
    'cnn'                    : cnn.CNN,
    'resnet18'               : resnet.ResNet18,
    'wrn16_2'                : wide_resnet.WRN16_2,
    'deit_tiny'              : deit.deit_tiny,
    'deit_small'             : deit.deit_small,
    'deit_base'              : deit.deit_base,
}
def build_model(args, base_dir, seed, num_classes):
    model_file = os.path.join(base_dir, f'seed={seed}-model.pt')
    net = load_model(args.model, num_classes, model_file, args.reload_model)
    # for m in net.modules():
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    #         init.kaiming_normal_(m.weight, mode='fan_in')
    #         if m.bias is not None:
    #             init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.BatchNorm2d):
    #         init.constant_(m.weight, 1)
    #         init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Linear):
    #         init.normal_(m.weight, std=1e-3)
    #         if m.bias is not None:
    #             init.constant_(m.bias, 0)
    return net

def load_model(model_name, num_classes=10, model_file=None, reload=True):
    net = models[model_name](num_classes=num_classes)
    if reload and os.path.exists(model_file):
        net.load_state_dict(torch.load(model_file, weights_only=True))
        print(f"load model from {model_file}")
    else:
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        torch.save(net.state_dict(), model_file)
        print(f"save model to {model_file}")
    net.eval()
    return net
