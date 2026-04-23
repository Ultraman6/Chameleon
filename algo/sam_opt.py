import random
import math
from collections import defaultdict
import numpy as np
import torch
from algo.erm import ERM
from utils.utils import disable_running_stats, enable_running_stats, accuracy


class SAM_OPT(ERM):
    def __init__(
            self, model, criterion, base_optimizer,
            rho=0.05, opt_step=20, start_sam=1, end_sam=-1,
            **kwargs
    ):
        super(SAM_OPT, self).__init__(model, criterion, base_optimizer, **kwargs)
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.state = defaultdict(dict)
        self.start_sam = start_sam
        self.opt_step = opt_step
        self.end_sam = end_sam if end_sam != -1 else math.inf
        self.rho = rho

    def _step(self, inputs, labels, epoch=None):
        if epoch < self.start_sam or epoch > self.end_sam or epoch is None:
            return super()._step(inputs, labels)
        else:
            return self.sam_step(inputs, labels)

    def sam_step(self, inputs, labels):
        max_loss = 0.0
        best_state_dict = self.model.state_dict()
        enable_running_stats(self.model)
        outputs = self.model(inputs)
        origin_loss = self.criterion(outputs, labels)
        origin_acc = accuracy(outputs, labels, topk=(1,))[0].item()
        self.model.zero_grad()
        origin_loss.backward()
        self.first_step(zero_grad=True)
        # disable_running_stats(self.model)
        for _ in range(self.opt_step):
            loss = self.criterion(self.model(inputs), labels)
            l = loss.item()
            if l > max_loss:
                max_loss = l
                best_state_dict = self.model.state_dict()
            loss.backward()
            self.find_step(zero_grad=True)
        self.model.load_state_dict(best_state_dict)
        disable_running_stats(self.model)
        outputs = self.model(inputs)
        perturb_loss = self.criterion(outputs, labels)
        perturb_acc = accuracy(outputs, labels, topk=(1,))[0].item()
        perturb_loss.backward()
        self.second_step(zero_grad=True)

        result = {
            'perturb_loss': perturb_loss.item(), 'perturb_acc': perturb_acc,
            'origin_loss': origin_loss.item(), 'origin_acc': origin_acc,
        }
        return result

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        g_0_norm = self.grad_norm()
        scale = self.rho / g_0_norm
        for p in self.params:
            e_w = p.grad * scale.to(p)
            self.state[p]['old_p'] = p.data.clone()
            p.data.add_(e_w)
        if zero_grad:
            self.model.zero_grad()

    @torch.no_grad()
    def find_step(self, zero_grad=False):
        g_0_norm = self.grad_norm()
        scale = self.rho / g_0_norm
        for p in self.params:
            e_w = p.grad * scale.to(p)
            p.data.copy_(self.state[p]['old_p'])
            p.data.add_(e_w)
        if zero_grad:
            self.model.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for p in self.params:
            p.data.copy_(self.state[p]['old_p'])
        self.optimizer.step()
        if zero_grad:
            self.model.zero_grad()

    def state_norm(self, key: str):
        norm = torch.norm(
            torch.stack([
                self.state[p][key].norm(p=2).to(self.device)
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
        if scheduler is not None:
            scheduler.load_state_dict(state_dict['scheduler'])
        # 为每个参数重建状态
        if 'state' in state_dict:
            state = state_dict['state']
            for i, (param, param_state) in enumerate(zip(self.params, state.values())):
                for key, value in param_state.items():
                    self.state[param][key] = value.to(param.device)

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
                'state': self.state,
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
            'state': self.state,
            'rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'py_rng_state': random.getstate()
        }
        if scheduler:
            state_dict['scheduler'] = scheduler.state_dict()
        return state_dict