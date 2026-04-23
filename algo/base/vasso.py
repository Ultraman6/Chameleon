import torch
from algo.sam import SAM


class VASSO(SAM):
    def __init__(
        self, model, criterion, base_optimizer,
              rho=0.05, theta=0.1, start_sam=1, end_sam=-1,
              **kwargs
    ):
        super().__init__(model, criterion, base_optimizer,
          rho, start_sam, end_sam,
          **kwargs
        )
        self.theta = theta
        self.g_0_norm = 0.0

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for p in self.params:
            if 'ema' not in self.state[p]:
                self.state[p]['ema'] = p.grad.clone().detach()
            else:
                self.state[p]['ema'].mul_(1 - self.theta)
                self.state[p]['ema'].add_(p.grad, alpha=self.theta)

        ema_norm = self.state_norm('ema')
        scale = self.rho / ema_norm

        for p in self.params:
            e_w = self.state[p]['ema'] * scale.to(p)
            self.state[p]['old_p'] = p.data.clone()
            p.data.add_(e_w)

        if zero_grad:
            self.model.zero_grad()
