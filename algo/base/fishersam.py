import torch
from algo.sam import SAM

class FisherSAM(SAM):
    def __init__(
        self, model, criterion, base_optimizer,
              rho=0.05, start_sam=1, end_sam=-1,
              alpha=0.01,
              **kwargs
    ):
        super().__init__(model, criterion, base_optimizer, rho, start_sam, end_sam, **kwargs)
        self.alpha = alpha
        for p in self.params:
            self.state[p]['fisher_diag'] = torch.zeros_like(p)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        g_0_norm = self.grad_norm()
        scale = self.rho / g_0_norm
        for p in self.params:
            if p.grad is None: continue
            # 获取该参数的 Fisher 信息矩阵对角线估计
            fisher_diag = self.state[p]['fisher_diag']
            fisher_diag.mul_(1 - self.alpha).add_(p.grad ** 2, alpha=self.alpha)
            # 计算 Fisher 加权的梯度 F^(-1/2) * grad，其中 F 是 Fisher 信息矩阵
            fisher_weighted_grad = p.grad / (torch.sqrt(fisher_diag) + 1e-12)
            # 重新归一化梯度以保持与原始 SAM 相同的扰动幅度
            fisher_grad_norm = torch.norm(fisher_weighted_grad)
            if fisher_grad_norm > 0:
                scale /= fisher_grad_norm
            e_w_0 = fisher_weighted_grad * scale
            self.state[p]['old_p'] = p.data.clone()  # 保存原始点
            p.data.add_(e_w_0)

        if zero_grad:
            self.model.zero_grad()