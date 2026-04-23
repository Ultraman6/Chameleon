import torch
from algo.sam import SAM

class LiteSAM(SAM):
    def __init__(
        self, model, criterion, base_optimizer,
              rho=0.05, start_sam=1, end_sam=-1,
              sigma=1, lmbda=0.9,
              **kwargs
    ):
        super().__init__(model, criterion, base_optimizer,
          rho, start_sam, end_sam,
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
        g_0_norm = self.grad_norm()
        scale = self.rho / g_0_norm
        for p in self.params:
            if p.grad is None: continue
            e_w = p.grad * scale.to(p)
            self.state[p]["old_p"] = p.data.clone()  # 保存原始点
            p.add_(e_w)

        if zero_grad:
            self.model.zero_grad()
