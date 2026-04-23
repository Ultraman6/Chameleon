import torch
from algo.sam import SAM
from utils.utils import disable_running_stats, enable_running_stats, _calculate_stats_from_list, accuracy


def analyse_ratio(grad_sam, grad_sgd):
    element_ratio = grad_sam / grad_sgd
    total = len(element_ratio)
    acc_ratio = (element_ratio >= 1).sum().item() / total
    red_ratio = ((element_ratio >= 0) & (element_ratio < 1)).sum().item() / total
    rev_ratio = (element_ratio < 0).sum().item() / total
    return acc_ratio, red_ratio, rev_ratio

class SALP(SAM):
    def __init__(
            self, model, criterion, base_optimizer,
            rho=0.05,
            rho_min=0.01, rho_max=1.0, rho_lr=1.0,
            start_sam=1, end_sam=-1,
            **kwargs
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        super(SALP, self).__init__(model, criterion, base_optimizer, rho, start_sam, end_sam, **kwargs)
        self.rho_lr = rho_lr
        self.rho_min, self.rho_max = rho_min, rho_max
        self.g_0_norm = 0.0
        self.g_1_loss = 0.0
        self.g_0_loss = 0.0
        for p in self.params:
            self.state[p]['rho'] = torch.full_like(p, rho)

    def sam_step(self, inputs, labels, **kwargs):
        enable_running_stats(self.model)
        outputs = self.model(inputs)
        origin_loss = self.criterion(outputs, labels)
        origin_acc = accuracy(outputs, labels, topk=(1,))[0].item()
        self.g_0_loss = origin_loss.detach()
        self.model.zero_grad()
        origin_loss.backward()
        self.first_step(zero_grad=True)

        disable_running_stats(self.model)
        outputs = self.model(inputs)
        perturb_loss = self.criterion(outputs, labels)
        perturb_acc = accuracy(outputs, labels, topk=(1,))[0].item()
        perturb_loss.backward()
        self.g_1_loss = perturb_loss.detach()
        self.second_step(zero_grad=True)
        result = {
            'perturb_loss': perturb_loss.item(), 'perturb_acc': perturb_acc,
            'origin_loss': origin_loss.item(), 'origin_acc': origin_acc,
        }
        return result

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        self.g_0_norm = self.grad_norm()
        for p in self.params:
            e_w = p.grad.data * (self.state[p]["rho"] / self.g_0_norm)
            self.state[p]['g_0'] = p.grad.data.clone()
            self.state[p]['old_p'] = p.data.clone()  # 保存原始点
            p.data.add_(e_w)
        if zero_grad:
            self.model.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False, inputs=None, targets=None):
        for p in self.params:
            p.data = self.state[p]['old_p']  # 撤销第一次扰动
            self.update_rho(p)

        self.optimizer.step()
        if zero_grad:
            self.model.zero_grad()

    def update_rho(self, p):
        g_1 = p.grad.data
        g_0 = self.state[p]['g_0']
        h_0 = g_1 - g_0
        rho = self.state[p]['rho']
        # max f(u)/||∇f(x)||
        rho_g = (g_1 * g_0 / self.g_0_norm ** 2
                 - h_0 * g_0 * (self.g_1_loss - self.g_0_loss) / self.g_0_norm ** 3)
        rho.add_(rho_g, alpha=self.rho_lr).clamp_(self.rho_min, self.rho_max)

    def get_rho_stats(self):
        """
        Calculates and returns statistics for all `rho` tensors in the optimizer.

        This method gathers all per-parameter `rho` tensors, concatenates them,
        and computes the mean, standard deviation, min, and max values.

        Returns:
            dict: A dictionary containing the statistics with keys
                  'mean', 'std', 'min', 'max'. Returns a dict of zeros if no
                  `rho` tensors are found.
        """
        all_rhos = []
        for p in self.params:
            all_rhos.append(self.state[p]['rho'].flatten())

        if not all_rhos:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

        return _calculate_stats_from_list(all_rhos)

    def get_rho_stats_by_layer(self):
        results = {'mean': {}, 'std': {}, 'min': {}, 'max': {}, 'cv': {}}
        rhos = []
        for p in self.params:
            rho_p = self.state[p]['rho']
            rhos.append(rho_p)
            param_stats = _calculate_stats_from_list([rho_p.flatten()])
            layer_name = self.param_to_name.get(id(p), None)
            for k, v in param_stats.items():
                results[k][layer_name] = v
        return results

    def get_rho(self, flatten=False):
        """
        Returns the current rho tensor for each parameter in the model.
        """
        rhos = []
        for p in self.params:
            if flatten:
                rhos.append(self.state[p]['rho'].clone().flatten())
            else:
                rhos.append(self.state[p]['rho'].clone())
        return rhos