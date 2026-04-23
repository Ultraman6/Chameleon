import torch
from .salp import SALP


class FSALP(SALP):
    def __init__(
        self, model, criterion, base_optimizer,
              rho=0.05, rho_min=0.01, rho_max=1.0, rho_lr=1.0,
              start_sam=1, end_sam=-1,
              sigma=1, lmbda=0.9,
              **kwargs
    ):
        super().__init__(model, criterion, base_optimizer,
          rho, rho_min, rho_max, rho_lr, start_sam, end_sam,
          **kwargs
        )
        self.sigma, self.lmbda = sigma, lmbda

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for p in self.params:
            if p.grad is None: continue
            grad = p.grad.clone()
            if not "momentum" in self.state[p]:
                self.state[p]["momentum"] = grad
            else:
                p.grad -= self.state[p]["momentum"] * self.sigma
                self.state[p]["momentum"] = self.state[p]["momentum"] * self.lmbda + grad * (1 - self.lmbda)

        self.g_0_norm = self.grad_norm()
        for p in self.params:
            if p.grad is None: continue
            e_w = p.grad * (self.state[p]["rho"] / self.g_0_norm)
            self.state[p]["old_p"] = p.data.clone()  # 保存原始点
            self.state[p]['g_0'] = p.grad.data.clone()
            p.add_(e_w)

        if zero_grad:
            self.model.zero_grad()
