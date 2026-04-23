import os
from torch.utils.data import Dataset
from typing import Tuple
from torchvision.transforms import Resize, CenterCrop, Normalize, ToTensor, Compose
from torchvision.datasets.utils import download_and_extract_archive, download_url, extract_archive
from utils.utils import DomainDataset

url = 'https://github.com/MachineLearning2020/Homework3-PACS/archive/refs/heads/master.zip'
classes = ['bird', 'feather', 'headphones', 'ice_cream', 'teapot', 'tiger', 'whale', 'windmill', 'wine_glass', 'zebra']
domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
url_temp = {
    "{}.zip": "http://csr.bu.edu/ftp/visda/2019/multi-source/{}.zip",
    "{}_train.txt": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/{}_train.txt",
    "{}_test.txt": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/{}_test.txt"
}
transforms = Compose([
    Resize([256, 256]),
    CenterCrop(224),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load(root: str = "../data", leave_domain='art_painting', **kwargs) -> Tuple[Dataset, Dataset]:
    print(f"""loading domainnet dataset with following settings: leave domain={leave_domain}""")
    data_path = os.path.join(root, "DomainNet")

    for d in domains:
        if not os.path.exists(os.path.join(data_path, "{}_{}.txt".format(d, 'train'))):
            download_url("http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/{}_{}.txt".format(d, 'train'), root=data_path)
        if not os.path.exists(os.path.join(data_path, d)):
            if os.path.exists(os.path.join(data_path, "{}.zip".format(d))):
                try:
                    extract_archive(os.path.join(data_path, "{}.zip".format(d)), data_path, remove_finished=False)
                except Exception as e:
                    print(e)
                    raise FileExistsError('There exists error in download .zipfile')
            else:
                for k, v in url_temp.items():
                    file_name = k.format(d)
                    url = v.format(d) if d in ['infograph', 'quickdraw', 'real', 'sketch'] else v.format(
                        f"groundtruth/{d}")
                    if file_name.endswith('.zip'):
                        download_and_extract_archive(url, data_path, remove_finished=True)
                    else:
                        download_url(url, data_path)
                else:
                    raise FileExistsError('File not exists. Please set download=True the download the raw data of {}'.format(d))

    trainset = DomainDataset(data_path, [d for d in domains if d != leave_domain], classes, transforms, 'train')
    testset = DomainDataset(data_path, [leave_domain], classes, transforms, 'train')

    return trainset, testset