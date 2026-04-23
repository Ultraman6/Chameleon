import torch, math
from algo.sam import SAM
from utils.utils import read_resnet18_base_shapes
class MU2P(SAM):
    def __init__(
        self, model, criterion, base_optimizer,
              rho=0.05, base=0.5, start_sam=1, end_sam=-1,
              **kwargs
    ):
        super().__init__(model, criterion, base_optimizer,
          rho, start_sam, end_sam,
          **kwargs
        )
        base_shapes = read_resnet18_base_shapes()
        for p in self.params:
            ndim = len(p.shape)
            n = self.param_to_name.get(id(p), None)
            base_shape = base_shapes[n]
            if ndim == 1: # vector-like
                dim = base_shape[0] if base_shape[0] else p.shape[0]
                factor = p.shape[0] / (base * dim) # width
            elif ndim == 2: # matrix-like
                dim1 = base_shape[0] if base_shape[0] else p.shape[0]
                dim2 = base_shape[1] if base_shape[1] else p.shape[1]
                factor = (p.shape[0] / p.shape[1]) / (base * dim1 / dim2)  # fan_out/fan_in
            else:
                factor = 1
            self.state[p]['factor'] = factor
            self.state[p]['rho'] = rho * factor
            self.state[p]['norm_scale'] = math.sqrt(factor)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        norm, mu_norm = self.grad_norm()
        for p in self.params:
            e_w = p.grad.data * (self.state[p]["rho"] / mu_norm)
            self.state[p]['mu'] = self.state[p]['factor'] * (norm / mu_norm)
            self.state[p]['old_p'] = p.data.clone()
            p.data.add_(e_w)
        if zero_grad:
            self.model.zero_grad()

    def grad_norm(self):
        norms, mu_norms = [], []
        for p in self.params:
            norms.append(p.grad.norm(p=2).to(self.device))
            mu_norms.append((self.state[p]['norm_scale'] * p.grad).norm(p=2).to(self.device))
        norm = torch.norm(
            torch.stack(norms),
            p=2)
        mu_norm = torch.norm(
            torch.stack(mu_norms),
            p=2)
        return norm + 1e-12, mu_norm + 1e-12

    def get_mu_by_layers(self):
        results = {}
        for p in self.params:
            layer_name = self.param_to_name.get(id(p), None)
            results[layer_name] = self.state[p]['mu']
        return results
