import json, random, torch, os
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader, default_collate
from typing import Tuple
import datasets.cifar10 as cifar10
import datasets.cifar100 as cifar100
import datasets.pacs as pasc
import datasets.domainnet as domainnet
import datasets.office_caltech10 as office_caltech10
import datasets.tinyimagenet as tinyimagenet
from utils.shared_memory import MemmapManager, sharable2dataset, dataset2sharable

index_func = lambda X: [xi[-1] for xi in X]

def cutmix_collate_fn(batch, cutmix_obj):
    return cutmix_obj(*default_collate(batch))

def build_dataset(ld, nr, bs, dataset, aug, args, base_dir, seed):
    train_base_path = os.path.join(base_dir, f'{args.classes}-{seed}-train.dat',)
    test_base_path = os.path.join(base_dir, f'{args.classes}-{seed}-test.dat')
    if args.leave_domain is not None and dataset in ['pacs', 'domainnet', 'office_caltech10']:
        trainset, testset = load_dataset(dataset, aug=aug, leave_domain=ld, memmap=args.memmap)
    else:
        trainset, testset = load_dataset(dataset, aug=aug, memmap=args.memmap)

    num_classes = len(set(index_func(trainset)))
    if dataset in ['cifar10', 'cifar100'] and nr > 0:
        noise_file = os.path.join(base_dir, f'{args.classes}-{seed}-train_{args.noise_ratio}-{args.noise_mode}-noise.json')
        noise_trainset = CIFARDecorator(trainset, nr, args.noise_mode, 'all', noise_file=noise_file)
        noise_testset = CIFARDecorator(testset, nr, args.noise_mode, 'test')
        trainset = noise_trainset
        testset = noise_testset

    if args.reload_data and os.path.exists(train_base_path) and os.path.exists(test_base_path):
        print(f"Loading dataset from {train_base_path} and {test_base_path}")
        trainloader = torch.load(train_base_path, weights_only=False)
        testloader = torch.load(test_base_path, weights_only=False)
    else:
        if args.classes is not None and len(args.classes) > 0:
            print(f"Filtering dataset for classes: {args.classes}")
            # 过滤训练集
            train_indices = [i for i, (_, label) in enumerate(trainset) if label in args.classes]
            trainset = Subset(trainset, train_indices)
            # 过滤测试集
            test_indices = [i for i, (_, label) in enumerate(testset) if label in args.classes]
            testset = Subset(testset, test_indices)
            print(f"Filtered trainset size: {len(trainset)}, testset size: {len(testset)}")
        trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=args.num_workers,
                                 pin_memory=True, persistent_workers=True, prefetch_factor=10)
        testloader = DataLoader(testset, batch_size=bs, num_workers=args.num_workers,
                                pin_memory=True, persistent_workers=True, prefetch_factor=10)

    if aug == 'cutmix':
        from torchvision.transforms import v2
        from torch.utils.data import default_collate
        from functools import partial
        cutmix = v2.CutMix(num_classes=num_classes, alpha=args.mix_alpha)
        trainloader.collate_fn = partial(cutmix_collate_fn, cutmix_obj=cutmix)

    torch.save(trainloader, train_base_path)
    torch.save(testloader, test_base_path)
    print(f"Saving dataloader to {train_base_path} and {test_base_path}")
    print(f"dataloader settings: {trainloader.num_workers} {trainloader.persistent_workers} {trainloader.prefetch_factor}")
    print(f"transform_train: {trainloader.dataset.transform}")
    print(f"transform_test: {testloader.dataset.transform}")
    print(f"trainset size: {len(trainloader.dataset)}")
    print(f"testset size: {len(testloader.dataset)}")
    return trainloader, testloader, num_classes

def load_dataset(dataset: str,
                 root: str = "../../data/",
                 memmap: bool = False,
                 **kwargs) -> Tuple[Dataset, Dataset]:
    """prepare the dataset.

    Args:
        dataset (str): the dataset name.
        root (str): the root path of the dataset.

    Returns:
        trainset and testset
    """
    if dataset == "cifar10":
        trainset, testset = cifar10.load(root, **kwargs)
    elif dataset == "cifar100":
        trainset, testset = cifar100.load(root, **kwargs)
    elif dataset == 'pacs':
        trainset, testset = pacs.load(root, **kwargs)
    elif dataset == 'domainnet':
        trainset, testset = domainnet.load(root, **kwargs)
    elif dataset == 'office_caltech10':
        trainset, testset = office_caltech10.load(root, **kwargs)
    elif dataset == 'tinyimagenet':
        trainset, testset = tinyimagenet.load(root, **kwargs)
    else:
        raise NotImplementedError(f"dataset {dataset} is not implemented.")

    if memmap:
        trainset = MemmapDataset(trainset, os.path.join(trainset.root, 'mmap'), 'train')
        testset = MemmapDataset(trainset, os.path.join(testset.root, 'mmap'), 'test')

    return trainset, testset

class NoiseInjector:
    """Handles noise injection for labels"""

    def __init__(self, noise_ratio=0.4, noise_mode='sym', num_classes=10):
        self.noise_ratio = noise_ratio
        self.noise_mode = noise_mode
        self.num_classes = num_classes
        # Class transition for asymmetric noise (CIFAR10 specific)
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}

    def inject_noise(self, labels):
        """Inject noise into labels"""
        num_samples = len(labels)
        noise_labels = []

        # Randomly select samples to corrupt
        idx = list(range(num_samples))
        random.shuffle(idx)
        num_noise = int(self.noise_ratio * num_samples)
        noise_idx = idx[:num_noise]

        for i in range(num_samples):
            if i in noise_idx:
                if self.noise_mode == 'sym':
                    # Symmetric noise: randomly change to any class
                    noise_label = random.randint(0, self.num_classes - 1)
                    noise_labels.append(noise_label)
                elif self.noise_mode == 'asym':
                    # Asymmetric noise: use predefined transitions
                    if self.num_classes == 10:  # CIFAR10
                        noise_label = self.transition.get(labels[i], labels[i])
                    else:
                        # For other datasets, fallback to symmetric
                        noise_label = random.randint(0, self.num_classes - 1)
                    noise_labels.append(noise_label)
            else:
                # Keep original label
                noise_labels.append(labels[i])

        return noise_labels


class CIFARDecorator(Dataset):
    """
    A decorator class that wraps existing CIFAR datasets and adds:
    - Noise injection
    - Additional augmentations (like Cutout)
    - Support for different modes (all, labeled, unlabeled, test)
    """

    def __init__(self, base_dataset,
                 noise_ratio=0.0, noise_mode='sym', mode='all',
                 transform=None, target_transform=None, noise_file=None,
                 pred=None, probability=None):
        """
        Args:
            base_dataset: Original CIFAR10/CIFAR100 dataset from torchvision
            noise_ratio: Ratio of samples to inject noise (0.0 to 1.0)
            noise_mode: 'sym' for symmetric noise, 'asym' for asymmetric
            mode: 'all', 'labeled', 'unlabeled', or 'test'
            transform: Additional transforms to apply (overrides base_dataset.transform if provided)
            target_transform: Transform for targets (overrides base_dataset.target_transform if provided)
            noise_file: Path to save/load noise labels
            pred: Predictions for sample selection (for labeled/unlabeled mode)
            probability: Probabilities for labeled samples
        """
        self.base_dataset = base_dataset
        self.noise_ratio = noise_ratio
        self.noise_mode = noise_mode
        self.mode = mode

        # Handle transforms - use provided or fall back to base_dataset's transforms
        self.transform = transform if transform is not None else getattr(base_dataset, 'transform', None)
        self.target_transform = target_transform if target_transform is not None else getattr(base_dataset, 'target_transform', None)

        # Determine number of classes
        if hasattr(base_dataset, 'classes'):
            self.num_classes = len(base_dataset.classes)
        else:
            # Assume CIFAR10 or CIFAR100 based on dataset name
            self.num_classes = 10 if 'CIFAR10' in str(type(base_dataset)) else 100

        # Get original data and labels
        if hasattr(base_dataset, 'data'):
            self.data = base_dataset.data
        else:
            # Load all data if not directly accessible
            self.data = []
            self.original_labels = []
            for i in range(len(base_dataset)):
                img, label = base_dataset[i]
                self.data.append(np.array(img))
                self.original_labels.append(label)
            self.data = np.array(self.data)

        if hasattr(base_dataset, 'targets'):
            self.original_labels = base_dataset.targets
        elif hasattr(base_dataset, 'labels'):
            self.original_labels = base_dataset.labels

        # Handle noise injection
        if self.noise_ratio > 0 and mode != 'test':
            if noise_file and os.path.exists(noise_file):
                # Load existing noise labels
                with open(noise_file, 'r') as f:
                    self.noise_labels = json.load(f)
            else:
                # Create new noise labels
                noise_injector = NoiseInjector(noise_ratio, noise_mode, self.num_classes)
                self.noise_labels = noise_injector.inject_noise(self.original_labels)

                # Save noise labels if file path provided
                if noise_file:
                    with open(noise_file, 'w') as f:
                        json.dump(self.noise_labels, f)
        else:
            self.noise_labels = self.original_labels

        # Handle different modes
        if mode in ['labeled', 'unlabeled'] and pred is not None:
            if mode == 'labeled':
                self.indices = pred.nonzero()[0]
                self.probability = [probability[i] for i in self.indices] if probability is not None else None
            else:  # unlabeled
                self.indices = (1 - pred).nonzero()[0]
                self.probability = None

            # Filter data and labels
            self.data = self.data[self.indices]
            self.noise_labels = [self.noise_labels[i] for i in self.indices]
        else:
            self.indices = None
            self.probability = None

    def _apply_target_transform(self, target):
        """Apply target transform if available"""
        if self.target_transform is not None:
            target = self.target_transform(target)
        return target

    def _process_image(self, img):
        """Process image with transforms"""
        # Convert to PIL Image if needed
        if not isinstance(img, Image.Image):
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = Image.fromarray(img.astype('uint8'), 'RGB')
            else:
                img = Image.fromarray(img)

        # Apply image transforms
        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.noise_labels[index]

        # Process image
        img = self._process_image(img)

        # Return based on mode
        if self.mode == 'labeled' and self.probability is not None:
            # For labeled mode with probability, return augmented pairs
            img2 = self._process_image(self.data[index])
            # Apply target transform to the label
            transformed_target = self._apply_target_transform(target)
            return img, img2, transformed_target, self.probability[index]

        elif self.mode == 'unlabeled':
            # For unlabeled mode, return augmented pairs (no labels)
            img2 = self._process_image(self.data[index])
            return img, img2

        else:
            # For 'all' and 'test' modes
            transformed_target = self._apply_target_transform(target)
            return img, transformed_target

class MemmapDataset(Dataset):
    """
    将任意PyTorch数据集转换为内存映射格式并提供Dataset接口

    参数:
        dataset: 原始数据集，必须是PyTorch Dataset对象
        output_path: 存储内存映射文件的目录
        name: 数据集名称，用于生成文件名
        shard_size: 转换过程中的数据切块大小
        force_recreate: 是否强制重新创建内存映射文件
    """

    def __init__(self, dataset, output_path, name="dataset", shard_size=512, force_recreate=False):
        super().__init__()
        self.output_path = output_path
        self.name = name
        self.memmap_path = os.path.join(output_path, name)

        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)

        # 检查是否已存在内存映射文件
        meta_path = os.path.join(self.memmap_path, 'meta.json')
        if os.path.exists(meta_path) and not force_recreate:
            # 加载已有的内存映射
            self.memmap_manager = MemmapManager(self.memmap_path)
            sharable_data = self.memmap_manager.get(name)
            self._dataset = sharable2dataset(sharable_data)
        else:
            # 创建新的内存映射
            if os.path.exists(self.memmap_path):
                if force_recreate:
                    import shutil
                    shutil.rmtree(self.memmap_path)
            os.makedirs(self.memmap_path, exist_ok=True)

            # 将数据集转换为共享格式
            sharable_data = dataset2sharable(dataset, shard_size)

            # 创建内存映射管理器
            # 确定元素数量
            num_elems = len(sharable_data)
            self.memmap_manager = MemmapManager(self.memmap_path, num_elems)

            # 添加数据并保存
            self.memmap_manager.add(sharable_data, name)
            self.memmap_manager.dump()
            self.memmap_manager.save_meta()

            # 创建临时数据集对象
            sharable_data = self.memmap_manager.get(name)
            self._dataset = sharable2dataset(sharable_data)

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


if __name__ == "__main__":
    import os
    import time
    import psutil
    import numpy as np
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    # 导入之前实现的MemmapDataset类及其依赖
    # 此处省略前面实现的代码...

    # 设置参数
    batch_size = 128
    num_workers = 4
    num_epochs = 2  # 测试轮数


    def measure_performance(dataloader, name, color):
        """测量数据加载性能"""
        process = psutil.Process(os.getpid())

        # 初始化性能指标
        batch_times = []
        memory_usages = []

        # 模拟训练循环
        start_time = time.time()
        for epoch in range(num_epochs):
            print(f"[{name}] Epoch {epoch + 1}/{num_epochs}")
            epoch_start = time.time()

            for i, (images, labels) in enumerate(dataloader):
                # 记录批次加载时间
                if i > 0:  # 跳过第一个批次(预热)
                    batch_times.append(time.time() - batch_start)

                # 记录内存使用
                memory_usages.append(process.memory_info().rss / (1024 * 1024))  # MB

                # 模拟简单处理
                images = images.float()
                labels = labels.long()

                # 打印进度
                if i % 20 == 0:
                    print(f"  Batch {i}/{len(dataloader)}, "
                          f"Memory: {memory_usages[-1]:.1f} MB, "
                          f"Batch time: {batch_times[-1] if len(batch_times) > 0 else 0:.4f}s")

                batch_start = time.time()

        total_time = time.time() - start_time

        # 返回性能指标
        return {
            'name': name,
            'color': color,
            'total_time': total_time,
            'avg_batch_time': np.mean(batch_times),
            'batch_times': batch_times,
            'memory_usages': memory_usages,
        }


    # 定义转换
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # =========================
    # 方法1: 原始数据加载
    # =========================
    def test_original_dataloader():
        print("\n===== 测试原始数据加载方式 =====")

        # 加载原始CIFAR10数据集
        train_dataset = torchvision.datasets.CIFAR10(
            root='/mnt/d/github/data/CIFAR10',
            train=True,
            download=True,
            transform=transform
        )

        # 创建原始数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        # 测试性能
        return measure_performance(train_loader, "原始加载", "blue")


    # =========================
    # 方法2: MemmapDataset加载
    # =========================
    def test_mmap_dataloader():
        print("\n===== 测试MemmapDataset加载方式 =====")

        # 加载原始CIFAR10数据集
        train_dataset = torchvision.datasets.CIFAR10(
            root='/mnt/d/github/data/CIFAR10',
            train=True,
            download=True,
            transform=transform
        )

        # 转换为MemmapDataset
        mmap_dataset = MemmapDataset(
            dataset=train_dataset,
            output_path='/mnt/d/github/data/CIFAR10/mmap',
            name='train',
            force_recreate=False  # 首次运行设为True，后续运行可设为False复用已有文件
        )

        # 创建MemmapDataset数据加载器
        mmap_loader = DataLoader(
            mmap_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        # 测试性能
        return measure_performance(mmap_loader, "MemmapDataset", "red")


    # 运行测试
    results_original = test_original_dataloader()
    results_mmap = test_mmap_dataloader()


    # =========================
    # 结果分析与可视化
    # =========================
    def plot_results(results_original, results_mmap):
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 批次加载时间比较
        ax1.set_title('批次加载时间对比')
        ax1.plot(results_original['batch_times'],
                 label=f"{results_original['name']} (平均: {results_original['avg_batch_time']:.4f}s)",
                 color=results_original['color'], alpha=0.7)
        ax1.plot(results_mmap['batch_times'],
                 label=f"{results_mmap['name']} (平均: {results_mmap['avg_batch_time']:.4f}s)",
                 color=results_mmap['color'], alpha=0.7)
        ax1.set_xlabel('批次')
        ax1.set_ylabel('加载时间 (秒)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 内存使用比较
        ax2.set_title('内存使用对比')
        ax2.plot(results_original['memory_usages'], label=results_original['name'],
                 color=results_original['color'], alpha=0.7)
        ax2.plot(results_mmap['memory_usages'], label=results_mmap['name'],
                 color=results_mmap['color'], alpha=0.7)
        ax2.set_xlabel('批次')
        ax2.set_ylabel('内存使用 (MB)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 总结统计
        plt.figtext(0.5, 0.01,
                    f"总运行时间对比: \n"
                    f"{results_original['name']}: {results_original['total_time']:.2f}秒 | "
                    f"{results_mmap['name']}: {results_mmap['total_time']:.2f}秒 | "
                    f"加速比: {results_original['total_time'] / results_mmap['total_time']:.2f}x",
                    ha='center', fontsize=12, bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('cifar10_dataloader_comparison.png')
        plt.show()

        # 打印详细结果
        print("\n===== 性能对比结果 =====")
        print(f"原始加载总时间: {results_original['total_time']:.2f}秒")
        print(f"MemmapDataset总时间: {results_mmap['total_time']:.2f}秒")
        print(f"加速比: {results_original['total_time'] / results_mmap['total_time']:.2f}x")
        print(f"原始加载平均批次时间: {results_original['avg_batch_time']:.4f}秒")
        print(f"MemmapDataset平均批次时间: {results_mmap['avg_batch_time']:.4f}秒")
        print(f"原始加载最大内存使用: {max(results_original['memory_usages']):.1f} MB")
        print(f"MemmapDataset最大内存使用: {max(results_mmap['memory_usages']):.1f} MB")
        print(
            f"内存节省: {(max(results_original['memory_usages']) - max(results_mmap['memory_usages'])) / max(results_original['memory_usages']) * 100:.1f}%")


    # 分析并绘制结果
    plot_results(results_original, results_mmap)

