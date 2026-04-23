import os, torch
from torchvision import datasets
from typing import Tuple
from torchvision.transforms import RandomCrop, AutoAugment, AutoAugmentPolicy, RandAugment, Normalize, ToTensor, Compose, RandomHorizontalFlip
from utils.utils import Cutout

def load(root: str = "../data", aug: str='basic', **kwargs)\
        -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    print(f"""loading cifar100... with following settings: aug={aug}""")
    data_path = os.path.join(root, "CIFAR100")

    transform_train = Compose([
        *([RandomCrop(32, padding=4),
           RandomHorizontalFlip()] if aug != 'none' else []),
        *([AutoAugment(AutoAugmentPolicy.CIFAR10)] if aug == 'auto' else []),
        *([RandAugment()] if aug == 'rand' else []),
        ToTensor(),
        *([Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))]),
        *([Cutout()] if aug == 'cutout' else []),
    ])

    transform_test = Compose([
        ToTensor(),
        *([Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))]),
    ])

    # load the dataset
    os.makedirs(root, exist_ok=True)
    trainset = datasets.CIFAR100(
        root=data_path, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(
        root=data_path, train=False, download=True, transform=transform_test)
    return trainset, testset