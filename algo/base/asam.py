import torch
from algo.sam import SAM


class ASAM(SAM):

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        g_0_norm = self.grad_norm()
        scale = self.rho / g_0_norm
        for p in self.params:
            e_w = torch.pow(p, 2) * p.grad.data * scale.to(p)
            self.state[p]['old_p'] = p.data.clone()
            p.data.add_(e_w)

        if zero_grad:
            self.model.zero_grad()
