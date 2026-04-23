import numpy as np
import torch
from algo.sam import SAM
from utils.utils import enable_running_stats, accuracy, disable_running_stats

class SSAM(SAM):
    def __init__(
        self, model, criterion, base_optimizer,
              rho=0.1, start_sam=1, end_sam=-1,
              sparsity=0.5, score_mode='curt',
              drop_rate = 0.5, gen_mode='rand',
              **kwargs
    ):
        super().__init__(model, criterion, base_optimizer, rho, start_sam, end_sam, **kwargs)
        self.sparsity = sparsity
        self.score_mode = score_mode
        self.drop_rate = drop_rate
        self.gen_mode = gen_mode
        self.dr = self.drop_rate
        self.init_mask()
        self.g_0_norm, self.g_1_loss = 0.0, 0.0

    def sam_step(self, inputs, labels, **kwargs):
        enable_running_stats(self.model)
        outputs = self.model(inputs)
        origin_loss = self.criterion(outputs, labels)
        origin_acc = accuracy(outputs, labels, topk=(1,))[0].item()
        self.model.zero_grad()
        origin_loss.backward()
        self.first_step(zero_grad=True)

        disable_running_stats(self.model)
        outputs = self.model(inputs)
        perturb_loss = self.criterion(outputs, labels)
        perturb_acc = accuracy(outputs, labels, topk=(1,))[0].item()
        perturb_loss.backward()
        self.g_1_loss = perturb_loss.detach()
        self.update_mask()
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
        return g_1 * g_0 / self.g_0_norm ** 2 - h_0 * g_0 * self.g_1_loss / self.g_0_norm ** 3

    @torch.no_grad()
    def init_mask(self):
        for p in self.params:
            self.state[p]['mask'] = torch.full_like(p, 1.0, requires_grad=False).to(p)
            # self.state[p]['rho_g'] = torch.zeros_like(p).to(p)

    @torch.no_grad()
    def update_mask(self):
        fisher_value_dict = {}
        # cal fisher value
        for p in self.params:
            fisher_value_dict[id(p)] = self.get_score(p).to(self.device)
        # topk fisher value
        fisher_value_list = torch.cat([torch.flatten(x) for x in fisher_value_dict.values()])
        keep_num = int(len(fisher_value_list) * (1 - self.sparsity))
        _value, _index = torch.topk(fisher_value_list, keep_num)
        mask_list = torch.zeros_like(fisher_value_list)
        mask_list.scatter_(0, _index, torch.ones_like(_value))
        start_index = 0
        for p in self.params:
            self.state[p]['mask'] = mask_list[start_index: start_index + p.numel()].reshape(p.shape)
            self.state[p]['mask'].to(p)
            self.state[p]['mask'].require_grad = False
            start_index = start_index + p.numel()
            assert self.state[p]['mask'].max() <= 1.0 and self.state[p]['mask'].min() >= 0.0
        assert start_index == len(mask_list)

    @torch.no_grad()
    def update_mask(self):
        """
        Fisher-based SSAM-F 带 swap 的掩码更新：
        - dr: 当前丢弃率 (0~1, 例如经余弦退火后的 self.dr)
        - sparsity: 总稀疏率 s
        目标：保持 (1-s)*N 个激活参数恒定，每次替换掉其中 dr*(1-s)*N 个。
        """
        fisher_value_dict = {}
        # 计算 Fisher 重要性
        for p in self.params:
            fisher_value_dict[id(p)] = self.get_score(p)  # 一般为 (grad**2)
        fisher_value_list = torch.cat([torch.flatten(x) for x in fisher_value_dict.values()])
        N = len(fisher_value_list)
        keep_target = int(N * (1 - self.sparsity))  # 总激活目标
        # 当前掩码展开
        old_mask = torch.cat([
            torch.flatten(self.state[p]['mask']) for p in self.params
        ])
        # Fisher 分数（仅活跃参数）*旧掩码
        active_scores = fisher_value_list * old_mask
        # ========== 计算丢弃与生长数量 ==========
        drop_k = int(round(self.dr * keep_target))
        drop_k = min(drop_k, keep_target)  # 边界保护
        keep_k = keep_target - drop_k
        # ========== Drop 阶段 ==========
        # 在当前激活的参数中按 Fisher 值降序保留 keep_k 个
        if keep_k > 0:
            keep_values, keep_idx = torch.topk(active_scores, keep_k)
        else:
            keep_idx = torch.tensor([], dtype=torch.long)
        new_mask = torch.zeros_like(fisher_value_list)
        if len(keep_idx) > 0:
            new_mask.scatter_(0, keep_idx, 1.0)
        # ========== Growth 阶段 ==========
        # 在当前未激活的参数中随机选 drop_k 个加入掩码
        # 若要改成按 Fisher 值选 Top-k，可改成 torch.topk(inactive_scores, drop_k)
        if drop_k > 0: # old_mask or new_mask
            inactive_idx = (1 - old_mask).nonzero(as_tuple=False).view(-1)
            if inactive_idx.numel() > 0:
                if self.gen_mode == 'rand':
                    weights = torch.ones(inactive_idx.numel(), device=inactive_idx.device)
                    sample_idx = torch.multinomial(weights, num_samples=drop_k, replacement=False)
                    grow_idx = inactive_idx[sample_idx]
                    # perm = torch.randperm(inactive_idx.numel())
                    # grow_idx = inactive_idx[perm[:drop_k]]
                elif self.gen_mode == 'curt':
                    inactive_scores = fisher_value_list * (1 - old_mask)
                    _, grow_idx = torch.topk(inactive_scores, drop_k)
                else:
                    raise KeyError
                new_mask.scatter_(0, grow_idx, 1.0)
        # ========== 重组 per-parameter 掩码 ==========
        start = 0
        for p in self.params:
            numel = p.numel()
            m = new_mask[start:start + numel].reshape(p.shape).to(p)
            self.state[p]['mask'] = m
            self.state[p]['mask'].requires_grad = False
            start += numel
            assert self.state[p]['mask'].max() <= 1.0 and self.state[p]['mask'].min() >= 0.0
        assert start == len(new_mask)

    def get_score(self, p):
        if self.score_mode == 'para':
            return torch.abs(p.clone()).data
        elif self.score_mode == 'grad':
            return torch.abs(p.grad.clone()).data
        elif self.score_mode == 'rand':
            return torch.rand(size=p.shape).data
        elif self.score_mode == 'curt':
            rho_g = self.rho_g(p)
            return torch.abs(rho_g).data
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