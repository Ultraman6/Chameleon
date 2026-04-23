import random
import numpy as np
import torch, time

from utils.utils import accuracy, enable_running_stats, disable_running_stats


class ERM:
    def __init__(self, model, criterion, base_optimizer, **kwargs):
        # 训练相关
        self.model = model
        self.criterion = criterion
        self.device = next(model.parameters()).device
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.param_to_name = {id(p): name for name, p in model.named_parameters() if p.requires_grad}
        self.optimizer = base_optimizer(self.params, **kwargs)

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self, inputs, labels, **kwargs):
        st = time.time()
        result = self._step(inputs, labels, **kwargs)
        result['batch_size'] = inputs.size(0)
        result['time'] = time.time() - st
        return result

    def _step(self, inputs, labels, **kwargs):
        """
        Performs a single step of the ERM optimization process.
        This method computes the model outputs, calculates the loss,
        and performs a backward pass to update the model parameters.
        """
        enable_running_stats(self.model)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        disable_running_stats(self.model)

        acc = accuracy(outputs, labels, topk=(1,))[0].item()
        return {'loss': loss.item(), 'acc': acc}

    def grad_norm(self):
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(self.device)
                for p in self.params
            ]),
            p=2)
        return norm + 1e-12

    def load(self, file_path, optim_kwargs=None, sche_kwargs=None, scheduler=None):
        """加载优化器状态"""
        state_dict = torch.load(file_path, weights_only=False)
        # 恢复随机状态
        torch.set_rng_state(state_dict['rng_state'])
        np.random.set_state(state_dict['np_rng_state'])
        random.setstate(state_dict['py_rng_state'])
        # 恢复基础优化器状态
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

        # 更新优化器参数
        if optim_kwargs is not None:
            for param_group in self.optimizer.param_groups:
                for key, value in optim_kwargs.items():
                    if key != 'lr' and key in param_group:
                        param_group[key] = value
        # 更新调度器参数
        if sche_kwargs is not None:
            for key, value in sche_kwargs.items():
                if hasattr(scheduler, key):
                    setattr(scheduler, key, value)
                if key == 'last_epoch':
                    for param_group in self.optimizer.param_groups:
                        param_group['initial_lr'] = optim_kwargs['lr']

    def save(self, file_path, scheduler=None, state_dict=None):
        """返回优化器状态"""
        if state_dict is None:
            state_dict = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'np_rng_state': np.random.get_state(),
                'py_rng_state': random.getstate()
            }
            if scheduler:
                state_dict['scheduler'] = scheduler.state_dict()
        torch.save(state_dict, file_path)

    def get(self, scheduler=None):
        """返回优化器状态"""
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'py_rng_state': random.getstate()
        }
        if scheduler:
            state_dict['scheduler'] = scheduler.state_dict()
        return state_dict

    def grad_norm_by_layer(self):
        """
        Returns the gradient norm for each parameter in the model.
        """
        grad_norms = {}
        for p in self.params:
            if p.requires_grad and p.grad is not None:
                grad_norms[self.param_to_name.get(id(p), str(id(p)))] = p.grad.norm(p=2).item()
            else:
                grad_norms[self.param_to_name.get(id(p), str(id(p)))] = 0.0
        return grad_norms

    def grad(self):
        """
        Returns the gradient for each parameter in the model.
        """
        grads = []
        for p in self.params:
            if p.requires_grad and p.grad is not None:
                grads.append(p.grad.clone().flatten())
            else:
                grads.append(torch.zeros_like(p).flatten())
        return torch.cat(grads)

    def set_grad(self, grad):
        """
        从拼接的一维梯度张量设置每个参数的梯度
        """
        if not isinstance(grad, torch.Tensor):
            raise ValueError("grad_tensor must be a torch.Tensor")

        # 计算总参数数量
        total_params = sum(p.numel() for p in self.params)
        if grad.numel() != total_params:
            raise ValueError(f"grad_tensor size {grad.numel()} doesn't match total parameters {total_params}")

        # 分割并设置梯度
        start_idx = 0
        for p in self.params:
            param_size = p.numel()
            if p.requires_grad:
                # 提取对应参数的梯度并reshape
                p.grad = grad[start_idx:start_idx + param_size].view_as(p)
            else:
                p.grad = None
            start_idx += param_size

    def grad_by_layer(self):
        """
        Returns the gradient for each parameter in the model.
        """
        grads = {}
        for p in self.params:
            k = self.param_to_name[id(p)]
            if p.requires_grad and p.grad is not None:
                grads[k] = p.grad.clone().flatten()
            else:
                grads[k] = torch.zeros_like(p).flatten()
        return grads