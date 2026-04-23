import math

import torch
from algo.sam import SAM
from utils.utils import enable_running_stats, accuracy, disable_running_stats


class SSAMD(SAM):
    def __init__(
        self, model, criterion, base_optimizer,
              rho=0.05, start_sam=1, end_sam=-1,
              sparsity=0.1, drop_rate=0.5, T_start=0, T_end=100,
              drop_strategy='gradient', growth_strategy='random',
              **kwargs
    ):
        super().__init__(model, criterion, base_optimizer, rho, start_sam, end_sam, **kwargs)
        self.num_batches = 1e-7
        self.sparsity = sparsity
        self.init_mask()
        self.g_0_norm, self.g_1_loss, self.g_0_loss = 0.0, 0.0, 0.0
        self.drop_rate = drop_rate
        self.T_start, self.T_end = T_start, T_end
        self.drop_strategy = drop_strategy
        self.growth_strategy = growth_strategy

    @torch.no_grad()
    def DeathRate_Scheduler(self, epoch):
        dr = (self.drop_rate) * (1 + math.cos(math.pi * (float(epoch - self.T_start) / (self.T_end - self.T_start)))) / 2
        return dr

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
        for p in self.params:
            self.state[p]['rho_g'] = self.rho_g(p)
            # self.state[p]['rho_g'].add_(self.rho_g(p), alpha=1.0 / self.num_batches / self.lr)
        self.second_step(zero_grad=True)
        result = {
            'perturb_loss': perturb_loss.item(), 'perturb_acc': perturb_acc,
            'origin_loss': origin_loss.item(), 'origin_acc': origin_acc,
        }
        return result

    def rho_g(self, p):
        g_1 = p.grad.data
        g_0 = self.state[p]['g_0']
        h_0 = g_1 - g_0
        return g_1 * g_0 / self.g_0_norm ** 2 - h_0 * g_0 * (self.g_1_loss - self.g_0_loss) / self.g_0_norm ** 3

    @torch.no_grad()
    def init_mask(self):
        for p in self.params:
            self.state[p]['mask'] = torch.full_like(p, 1.0, requires_grad=False).to(p)
            self.state[p]['rho_g'] = torch.zeros_like(p).to(p)
            self.state[p]['rho_g_'] = torch.zeros_like(p).to(p)

    @torch.no_grad()
    def update_mask(self, epoch):
        death_scores = []
        growth_scores = []
        for p in self.params:
            death_score = self.get_score(p, self.drop_strategy)
            death_scores.append((death_score + 1e-7) * self.state[p]['mask'].cpu().data)

            growth_score = self.get_score(p, self.growth_strategy)
            growth_scores.append((growth_score + 1e-7) * (1 - self.state[p]['mask'].cpu().data))
        '''
            Death 
        '''
        death_scores = torch.cat([torch.flatten(x) for x in death_scores])
        death_rate = self.DeathRate_Scheduler(epoch=epoch)
        death_num = int(min((len(death_scores) - len(death_scores) * self.sparsity) * death_rate,
                            len(death_scores) * self.sparsity))
        d_value, d_index = torch.topk(death_scores,
                                      int((len(death_scores) - len(death_scores) * self.sparsity) * (1 - death_rate)))

        death_mask_list = torch.zeros_like(death_scores)
        death_mask_list.scatter_(0, d_index, torch.ones_like(d_value))
        '''
            Growth
        '''
        growth_scores = torch.cat([torch.flatten(x) for x in growth_scores])
        growth_num = death_num
        g_value, g_index = torch.topk(growth_scores, growth_num)

        growth_mask_list = torch.zeros_like(growth_scores)
        growth_mask_list.scatter_(0, g_index, torch.ones_like(g_value))

        '''
            Mask
        '''
        start_index = 0
        for p in self.params:
            death_mask = death_mask_list[start_index: start_index + p.numel()].reshape(p.shape)
            growth_mask = growth_mask_list[start_index: start_index + p.numel()].reshape(p.shape)
            self.state[p]['mask'] = death_mask + growth_mask
            self.state[p]['mask'] = self.state[p]['mask'].to(p)
            self.state[p]['mask'].require_grad = False
            start_index = start_index + p.numel()
            assert self.state[p]['mask'].max() <= 1.0 and self.state[p]['mask'].min() >= 0.0

        assert start_index == len(death_mask_list)

    def get_score(self, p, score_mode=None):
        if score_mode == 'weight':
            return torch.abs(p.clone()).cpu().data
        elif score_mode == 'gradient':
            return torch.abs(p.grad.clone()).cpu().data
        elif score_mode == 'random':
            return torch.rand(size=p.shape).cpu().data
        elif score_mode == 'curvature':
            return torch.abs(self.state['rho_g'].clone()).cpu().data
        else:
            raise KeyError

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        self.g_0_norm = self.grad_norm()
        scale = self.rho / self.g_0_norm
        for p in self.params:
            if p.grad is None: continue
            e_w = p.grad * scale
            e_w.data = e_w.data * self.state[p]['mask']  # mask the epsilon
            self.state[p]['old_p'] = p.data.clone()
            self.state[p]['g_0'] = p.grad.data.clone()
            p.add_(e_w)  # climb to the local maximum "w + e(w)"
        if zero_grad:
            self.model.zero_grad()

    @torch.no_grad()
    def mask_info(self):
        live_num = 0
        total_num = 0
        for p in self.params:
            live_num += self.state[p]['mask'].sum().item()
            total_num += self.state[p]['mask'].numel()
        return float(live_num) / total_num

    def get_mask_stats_by_layer(self):
        sparsity = {}
        for p in self.params:
            live_num = self.state[p]['mask'].sum().item()
            total_num = self.state[p]['mask'].numel()
            layer_name = self.param_to_name.get(id(p), None)
            sparsity[layer_name] = float(live_num) / total_num
        return sparsity