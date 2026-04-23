import torch
from .salp import SALP


class VASSLP(SALP):
    def __init__(
        self, model, criterion, base_optimizer,
              rho=0.05, rho_min=0.01, rho_max=1.0, rho_lr=1.0,
              start_sam=1, end_sam=-1,
              theta=0.1,
              **kwargs
    ):
        super().__init__(model, criterion, base_optimizer,
          rho, rho_min, rho_max, rho_lr, start_sam, end_sam,
          **kwargs
        )
        self.theta = theta
        self.g_0_norm = 0.0

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for p in self.params:
            if 'g_0' not in self.state[p]:
                self.state[p]['g_0'] = p.grad.clone().detach()
            else:
                self.state[p]['g_0'].mul_(1 - self.theta)
                self.state[p]['g_0'].add_(p.grad, alpha=self.theta)

        self.g_0_norm = self.state_norm('g_0')
        for p in self.params:
            e_w = self.state[p]['g_0'] * (self.state[p]["rho"] / self.g_0_norm)
            self.state[p]['old_p'] = p.data.clone()
            p.data.add_(e_w)

        if zero_grad:
            self.model.zero_grad()
