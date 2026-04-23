from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.hessian import Hessian

def evaluate_flatness(args, device, dataloader, net, criterion):
    """Evaluate the flatness of the model at a given epoch."""
    hessian_comp = Hessian(net, criterion, device, dataloader=dataloader)
    eigvals, _ = hessian_comp.eigenvalues(top_n=args.neigs)
    traces = hessian_comp.trace()
    # density = hessian_comp.density()

    return {'eigvals': eigvals, 'trace': np.mean(traces)}

def grad_norm(device,
              model: torch.nn.Module,
              loader: DataLoader,
              criterion: Callable,
              lp: int = 2) -> float:
    """
    计算模型在整个 dataset 上的梯度范数均值。

    Args:
        device: 设备，如 'cuda' 或 'cpu'
        model: 待评估的模型
        criterion: 损失函数
        optimizer: 优化器，用于 zero_grad()
        dataset: 用于评估的 Dataset
        batch_size: DataLoader 的批量大小
        lp: 使用的范数阶数

    Returns:
        float: 所有 batch 上梯度范数的平均值
    """
    model = model.to(device)
    model.eval()
    total_norm = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        grads = [
            torch.norm(p.grad.detach(), p=float(lp))
            for p in model.parameters()
            if p.grad is not None and p.requires_grad
        ]
        batch_norm = torch.norm(torch.stack(grads), p=float(lp)).item()
        total_norm.append(batch_norm)

    return float(np.mean(total_norm))

def lanczos(device: torch.device,
            matrix_vector: Callable,
            dim: int,
            neigs: int):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products).

    Args:
        device: GPU or CPU
        matrix_vector: the matrix-vector product
        dim: the dimension of the matrix
        neigs: the number of eigenvalues to compute

    Returns:
        the eigenvalues and eigenvectors
    """
    def mv(vec: np.ndarray): # vec: numpy array
        gpu_vec = torch.tensor(vec, dtype=torch.float).to(device)
        return matrix_vector(gpu_vec).detach().cpu() # which should be a torch tensor on CPU

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()

def compute_hvp(device: torch.device,
                model: nn.Module,
                dataset: Dataset,
                loss_fn: nn.Module,
                vector: torch.Tensor,
                physical_batch_size) -> torch.Tensor:
    """Compute a Hessian-vector product.

    Args:
        device: GPU or CPU
        model: the model
        dataset: the dataset
        loss_fn: the loss function
        vector: the vector
        physical_batch_size: the physical batch size

    Returns:
        the Hessian-vector product
    """
    p = len(parameters_to_vector(model.parameters()))
    n = len(dataset)

    hvp = torch.zeros(p, dtype=torch.float, device=device)
    vector = vector.to(device)

    dataloader = DataLoader(dataset, batch_size=physical_batch_size, shuffle=False)
    for (X, y) in tqdm(dataloader):
        # move to GPU
        X, y = X.to(device), y.to(device)
        # compute the Hessian-vector product
        loss = loss_fn(model(X), y) / n
        grads = torch.autograd.grad(loss, inputs=model.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, model.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads)
    return hvp

def H_eigval(device: torch.device,
             model: nn.Module,
             dataset: Dataset,
             loss_fn: nn.Module,
             neigs: int = 6,
             physical_batch_size: int = 1000) -> torch.Tensor:
    """Compute the leading Hessian eigenvalues.

    Args:
        device: GPU or CPU
        model: the model
        dataset: the dataset
        loss_fn: the loss function
        neigs: the number of eigenvalues to compute
        physical_batch_size: the physical batch size

    Returns:
        the eigenvalues
    """
    hvp_delta = lambda delta: compute_hvp(device, model, dataset, loss_fn,
        delta, physical_batch_size=physical_batch_size)
    nparams = len(parameters_to_vector((model.parameters())))
    evals, evecs = lanczos(device, hvp_delta, nparams, neigs=neigs)
    return evals

def H_trace(device: torch.device,
            model: nn.Module,
            dataset: Dataset,
            loss_fn: nn.Module,
            physical_batch_size: int=1000,
            n_probes: int = 50) -> float:
    """
    使用 Hutchinson 方法估计 Hessian 的迹（trace）。

    Args:
        device: GPU or CPU
        model: 已加载的模型
        dataset: 用于计算 HVP 的数据集
        loss_fn: 损失函数
        physical_batch_size: mini-batch 大小
        n_probes: 随机探测次数

    Returns:
        对 Tr(H) 的估计值
    """
    # 维度
    dim = len(parameters_to_vector(model.parameters()))
    # Hessian–vector product 接口
    hvp_fn = lambda v: compute_hvp(
        device, model, dataset, loss_fn, v, physical_batch_size
    )

    trace_est = 0.0
    for _ in range(n_probes):
        # 生成 Rademacher 随机向量 v ∈ {+1, −1}^dim
        v = torch.randint(0, 2, (dim,), dtype=torch.float32, device=device)
        v[v == 0] = -1
        # 计算 v^T H v
        hv = hvp_fn(v)
        trace_est += torch.dot(v, hv).item()

    return trace_est / n_probes

def H_density(device: torch.device,
              model: nn.Module,
              dataset: Dataset,
              loss_fn: nn.Module,
              physical_batch_size: int,
              lanczos_steps: int = 100,
              n_runs: int = 5):
    """
    使用随机 Lanczos 正交二次法（SLQ）估计 Hessian 的谱密度（spectral density）。

    Args:
        device: GPU or CPU
        model: 已加载并置于 eval() 模式的模型
        dataset: 用于 HVP 计算的 Dataset
        loss_fn: 损失函数
        physical_batch_size: mini-batch 大小
        lanczos_steps: 每次 Lanczos 迭代步数（SLQ 中的子空间维度）
        n_runs: SLQ 随机重复次数

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]：
        对于每一次 run，返回一对 (eigenvalues, weights)：
          - eigenvalues: shape=(lanczos_steps,) 对应 Tₗ 矩阵的特征值
          - weights:    shape=(lanczos_steps,) 对应初始 Rademacher 向量在这些特征向量上的权重平方
    """
    # 参数向量化后的维度
    dim = len(parameters_to_vector(model.parameters()))
    # 封装 HVP 函数
    hvp_fn = lambda v: compute_hvp(
        device, model, dataset, loss_fn, v, physical_batch_size
    )

    densities = []
    for _ in range(n_runs):
        # 1. 生成 Rademacher 随机向量 v ∈ {+1, −1}^dim 并归一化
        v = torch.randint(0, 2, (dim,), dtype=torch.float32, device=device)
        v[v == 0] = -1
        v /= (v.norm() + 1e-8)

        # 2. 调用 Lanczos 算法，得到 T_l 的特征分解
        evals, evecs = lanczos(device, hvp_fn, dim, neigs=lanczos_steps)
        # evals: Tensor(lanczos_steps)；evecs: Tensor(lanczos_steps, lanczos_steps)

        # 3. 计算权重：初始向量 v 在 Lanczos 特征向量上的投影平方
        #    由于我们无法直接获得 v 在 evecs 空间的投影，这里近似用 evecs 的第一行（对应初始基向量）：
        weights = (evecs[0] ** 2)

        densities.append((evals.cpu().numpy(), weights.cpu().numpy()))

    return densities