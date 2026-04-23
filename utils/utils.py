import os, datetime, logging, random, torch

import yaml
from PIL import Image
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import Grayscale

def read_resnet18_base_shapes():
    '''Read resnet18.yaml and return a dict mapping name -> list.

    The YAML file lives next to this utils module under the same directory.
    Values of 'null' in YAML are returned as Python None.
    '''
    with open('resnet18.yaml', 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError('resnet18.yaml must contain a mapping from name to list')
    result = {}
    for name, value in data.items():
        result[str(name)] = list(value) if isinstance(value, list) else [value]
    return result

class ModelWrapper(torch.nn.Module):
    def __init__(self, model:torch.nn.Module):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x).logits   # 永远只返回 logits

class DomainDataset(Dataset):
    def __init__(self, root, domains: list, classes:list, transform=None, split:str=None):
        self.transform = transform
        self.images_path = []
        self.labels = []
        # 加载每个领域的数据
        for domain in domains:
            if split is None:
                domain_path = str(os.path.join(root, domain))
                img_lists = [os.listdir(os.path.join(domain_path, c)) for c in classes]

                for i, (img_list, c) in enumerate(zip(img_lists, classes)):
                    for img in img_list:
                        self.images_path.append(os.path.join(domain_path, c, img))
                        self.labels.append(i)
            else:
                with open(os.path.join(root, '{}_{}.txt'.format(domain, split)), 'r') as inf:
                    all_images_path = inf.readlines()
                    all_label_names = [p.split(os.path.sep)[1] for p in all_images_path]
                    label_list = tuple(sorted(list(set(all_label_names))))
                    if classes is None: classes = label_list
                    tmp_images = []
                    tmp_labels = []
                    for i, (img, lb) in enumerate(zip(all_images_path, all_label_names)):
                        if lb in classes:
                            tmp_images.append(os.path.join(root, img.strip().split(' ')[0]))
                            tmp_labels.append(lb)
                    self.images_path = tmp_images
                    self.labels = [classes.index(lb) for lb in tmp_labels]

    def __getitem__(self, item):
        img_path = self.images_path[item]
        label = self.labels[item]
        image = Image.open(img_path)

        # 确保图像为 RGB 格式
        if len(image.split()) != 3:
            image = Grayscale(num_output_channels=3)(image)

        # 应用预处理
        if self.transform is not None:
            image = self.transform(image)

        # 对图像进行归一化处理（如果需要）
        if image.dtype == torch.uint8 or image.dtype == torch.int8:
            image = image / 255.0

        return image, label

    def __len__(self):
        return len(self.images_path)


class Cutout:
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image

        left = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top + self.size)

        image[:, max(0, left): right, max(0, top): bottom] = 0
        return image


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

def _calculate_stats_from_list(rho_tensor_list):
    """Internal helper to calculate stats from a list of tensors."""
    if not rho_tensor_list:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    rho_vector = torch.cat([t for t in rho_tensor_list])
    return {
        'mean': rho_vector.mean().item(),
        'std': rho_vector.std().item(),
        'min': rho_vector.min().item(),
        'max': rho_vector.max().item(),
        'cv': (rho_vector.std() / (rho_vector.mean() + 1e-12)).item() if rho_vector.mean() > 0 else 0.0
    }


def get_datetime() -> str:
    """get the date.
    Returns:
        date (str): the date.
    """
    datetime_ = datetime.datetime.now().strftime("%m%d%H%M")
    return datetime_


def set_logger(save_path: str) -> None:
    """set the logger.
    Args:
        save_path(str): the path for saving logfile.txt
        name(str): the name of the logger
        verbose(bool): if true, will print to console.

    Returns:
        None
    """
    # set the logger
    logfile = os.path.join(save_path, "logfile.txt")
    logging.basicConfig(filename=logfile,
                        filemode="w+",
                        format='%(name)-12s: %(levelname)-8s %(message)s',
                        datefmt="%H:%M:%S",
                        level=logging.INFO)
    # define a Handler which writes DEBUG messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # tell the handler to use this format
    console.setFormatter(logging.Formatter(
        '%(name)-12s: %(levelname)-8s %(message)s'))
    # add the handler to the root logger
    logging.getLogger().addHandler(console)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k,
    and handles both traditional labels and one-hot encoded labels.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    # 如果 target 是 One-hot 编码
    if target.ndimension() == 2:  # One-hot 编码标签 (batch_size, num_classes)
        # 获取前 k 个预测的类别索引
        _, pred = output.topk(maxk, 1, True, True)  # 获取 top-k 的类别索引
        pred = pred.t()

        # 获取真实标签的类别索引
        true_labels = target.argmax(dim=1)

        # 检查真实标签是否出现在前 k 个预测类别中
        correct = pred.eq(true_labels.view(1, -1).expand_as(pred))

    else:
        # 传统标签
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

    # 计算 Top-k 精度
    res = []
    for k in topk:
        # 获取正确的前 k 个预测
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def get_logger(name:str,
               verbose:bool = True) -> logging.Logger:
    """get the logger.
    Args:
        name (str): the name of the logger
        verbose (bool): if true, will print to console.
    Returns:
        logger (logging.Logger)
    """
    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    if not verbose:
        logger.setLevel(logging.INFO)
    return logger

def set_seed(seed: int = 0) -> None:
    """set the random seed for multiple packages.
    Args:
        seed (int): the seed.

    Returns:
        None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_device(device: int) -> torch.device:
    """set GPU device.
    Args:
        device (int) the number of GPU device

    Returns:
        device (torch.device)
    """
    logger = get_logger(__name__)
    if torch.cuda.is_available():
        if device >= torch.cuda.device_count():
            logger.error("CUDA error, invalid device ordinal")
            exit(1)
    else:
        logger.error("Plz choose other machine with GPU to run the program")
        exit(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    device = torch.device("cuda:" + str(device))
    logger.info(device) 
    return device