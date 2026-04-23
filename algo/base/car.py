import torch
from algo.sam import SAM

class CAR(SAM):
    def __init__(
        self, model, criterion, base_optimizer,
        rho=0.1, start_sam=1, end_sam=-1,
        alpha=0.2, beta=0.1,
        **kwargs
    ):
        super().__init__(model, criterion, base_optimizer,
          rho, start_sam, end_sam,
          **kwargs
        )
        self.rho_ = rho
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        self.g_0_norm = self.grad_norm()
        scale = self.rho / self.g_0_norm
        for p in self.params:
            self.state[p]['p_0'] = p.data.clone()
            self.state[p]['g_0'] = p.grad.data.clone()
            p.data.add_(p.grad * scale.to(p))
        if zero_grad:
            self.optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        scale = self.rho_ / self.g_0_norm
        for p in self.params:
            g_1 = p.grad.data.clone()
            g_0 = self.state[p]['g_0']
            p.data.copy_(self.state[p]['p_0'])
            p.grad.data.add_(g_1 - g_0, alpha=self.alpha)
            p.data.sub_(g_0 * scale.to(p), alpha=self.beta)
        self.optimizer.step()
        if zero_grad:
            self.optimizer.zero_grad()
