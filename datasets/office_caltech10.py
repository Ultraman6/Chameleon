import os
from torch.utils.data import Dataset
from typing import Tuple
from torchvision.transforms import Resize, CenterCrop, Normalize, ToTensor, Compose
from torchvision.datasets.utils import download_and_extract_archive
from utils.utils import DomainDataset

url = 'https://github.com/ChristophRaab/Office_Caltech_DA_Dataset/archive/refs/heads/main.zip'
classes = ['backpack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']
domains = ['Caltech', 'amazon', 'dslr', 'webcam']
transforms = Compose([
    Resize([256, 256]),
    CenterCrop(224),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load(root: str = "../data", leave_domain = 'Caltech', **kwargs) -> Tuple[Dataset, Dataset]:
    print(f"""loading office_caltech10 dataset with following settings: leave domain={leave_domain}""")
    root = os.path.join(root, "office_caltech10")
    data_path = os.path.join(root, 'Office_Caltech_DA_Dataset-main')

    file_exist = [os.path.exists(os.path.join(data_path, d)) for d in domains]
    if not all(file_exist):
        download_and_extract_archive(url, root, remove_finished=True)

    trainset = DomainDataset(data_path, [d for d in domains if d != leave_domain], classes, transforms)
    testset = DomainDataset(data_path, [leave_domain], classes, transforms)

    return trainset, testset