import os
from torch.utils.data import Dataset
from typing import Tuple
from torchvision.transforms import Resize, CenterCrop, Normalize, ToTensor, Compose
from torchvision.datasets.utils import download_and_extract_archive
from utils.utils import DomainDataset

url = 'https://github.com/MachineLearning2020/Homework3-PACS/archive/refs/heads/master.zip'
classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
domains = ['art_painting', 'cartoon', 'photo', 'sketch']
transforms = Compose([
    Resize([256, 256]),
    CenterCrop(224),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load(root: str = "../data", leave_domain = 'art_painting', **kwargs) -> Tuple[Dataset, Dataset]:
    print(f"""loading pacs dataset with following settings: leave domain={leave_domain}""")
    data_path = os.path.join(root, "PACS", 'Homework3-PACS-master','PACS')

    file_exist = [os.path.exists(os.path.join(data_path, d)) for d in domains]
    if not all(file_exist):
        download_and_extract_archive(url, data_path, remove_finished=True)

    trainset = DomainDataset(data_path, [d for d in domains if d != leave_domain], classes, transforms)
    testset = DomainDataset(data_path, [leave_domain], classes, transforms)

    return trainset, testset