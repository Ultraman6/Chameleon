import os, torch
from torch.utils.data import Dataset
from torchvision import datasets
from typing import Tuple
from torchvision.transforms import RandomCrop, AutoAugment, AutoAugmentPolicy, RandAugment, Normalize, ToTensor, Compose, RandomHorizontalFlip
from utils.utils import Cutout

def load(root: str = "../data", aug: str='basic', **kwargs)\
        -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    print(f"""loading cifar10... with following settings: aug={aug}""")
    data_path = os.path.join(root, "CIFAR10")

    transform_train = Compose([
        *([RandomCrop(32, padding=4),
           RandomHorizontalFlip()] if aug != 'none' else []),
        *([AutoAugment(AutoAugmentPolicy.CIFAR10)] if aug == 'auto' else []),
        *([RandAugment()] if aug == 'rand' else []),
        ToTensor(),
        *([Normalize((0.49139968, 0.48215827, 0.44653124), (0.2023, 0.1994, 0.2010))]),
        *([Cutout()] if aug == 'cutout' else []),
    ])

    transform_test = Compose([
        ToTensor(),
        *([Normalize((0.49139968, 0.48215827, 0.44653124), (0.2023, 0.1994, 0.2010))]),
    ])

    # load the dataset
    os.makedirs(root, exist_ok=True)
    trainset = datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test)
    return trainset, testset