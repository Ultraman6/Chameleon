from torch import optim
from algo.erm import ERM
from algo.sam import SAM
from algo.sam_opt import SAM_OPT
from algo.base import *
from algo.learn import *
from itertools import product as p

# 算法配置全局常量：定义每个算法的参数配置
ALGO_CONFIGS = {
    'sam': (SAM, [], []),
    'sam_opt': (SAM_OPT, ['opt_step'], ['opt_step']),
    'car': (CAR, [], []), # 'alpha', 'beta'
    'asam': (ASAM, [], []),
    'ssam': (SSAM, ['sparsity', 'score_mode',  'drop_rate', 'gen_mode'], ['sparsity', 'score_mode',  'drop_rate', 'gen_mode']),
    'fishersam': (FisherSAM, ['fishersam_alpha'], ['alpha']),
    'fsam': (FSAM, ['fsam_sigma', 'fsam_lmbda'], ['sigma', 'lmbda']),
    'vasso': (VASSO, ['vasso_theta'], ['theta']),
    'mu2p': (MU2P, ['base'], ['base']),
    # learn methods
    'salp': (SALP, ['rho_min', 'rho_max', 'rho_lr'],
             ['rho_min', 'rho_max', 'rho_lr']),
    'fsalp': (FSALP, ['fsam_sigma', 'fsam_lmbda', 'rho_min', 'rho_max', 'rho_lr'],
              ['sigma', 'lmbda', 'rho_min', 'rho_max', 'rho_lr']),
    'vasslp': (VASSLP, ['vasso_theta', 'rho_min', 'rho_max', 'rho_lr'], ['theta', 'rho_min', 'rho_max', 'rho_lr']),
}

def build_algo(args, algo):
    """
    构建优化算法

    Args:
        model: 神经网络模型
        base_optimizer: 基础优化器类（如SGD、Adam）
        criterion: 损失函数
        args: 命令行参数
        algo: 算法名称
        **kwargs: 其他优化器参数

    Returns:
        optimizers: 优化器列表
        algo_kwargs_list: 算法参数列表
    """
    algo_kwargs_list = []

    if algo == 'erm':
        optim = ERM
        algo_kwargs_list.append({})
    else:
        # 检查算法是否支持
        if algo not in ALGO_CONFIGS:
            raise ValueError(f"Unsupported algorithm: {algo}. "
                             f"Supported algorithms are: erm, {', '.join(ALGO_CONFIGS.keys())}")
        config = ALGO_CONFIGS[algo]
        optim = config[0]
        param_source_names = config[1]
        param_names = config[2]

        # 通过getattr动态获取参数值
        param_value_lists = []
        for source_name in param_source_names:
            param_values = getattr(args, source_name)
            # 确保参数值是列表形式，如果不是则转换为列表
            if not isinstance(param_values, (list, tuple)):
                param_values = [param_values]
            param_value_lists.append(param_values)

        # 处理需要rho参数的算法
        for rho in args.rho:
            base_algo_kwargs = {'rho': rho,'start_sam': args.start_sam,'end_sam': args.end_sam}
            # 如果没有额外参数，直接创建优化器
            if not param_names:
                algo_kwargs = base_algo_kwargs.copy()
                algo_kwargs_list.append(algo_kwargs)
            else:
                # 有额外参数，需要进行参数组合
                param_combinations = p(*param_value_lists)
                for param_values in param_combinations:
                    algo_kwargs = base_algo_kwargs.copy()
                    # 构建参数字典
                    param_dict = dict(zip(param_names, param_values))
                    algo_kwargs.update(param_dict)
                    algo_kwargs_list.append(algo_kwargs)

    return optim, algo_kwargs_list

def build_optimizer(args):
    print(f'Optimizer: {args.optimizer}')
    if args.optimizer == 'sgd':
        base_optimizer = optim.SGD
        optim_kwargs = {'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay, 'nesterov': args.nesterov}
    elif args.optimizer == 'adamw':
        base_optimizer = optim.AdamW
        optim_kwargs = {'lr': args.lr, 'betas': args.betas, 'eps': args.eps, 'weight_decay': args.weight_decay}
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')
    print(f'Optimizer: {optim_kwargs}')
    return base_optimizer, optim_kwargs

def build_scheduler(args):
    if args.scheduler == 'constant':
        sche_kwargs = {'factor': args.lr_decay, 'total_iters': args.total_iters, 'last_epoch': args.last_epoch}
        scheduler = optim.lr_scheduler.ConstantLR
    elif args.scheduler == 'multistep':
        sche_kwargs = {'milestones': args.milestones, 'gamma': args.lr_decay, 'last_epoch': args.last_epoch}
        scheduler = optim.lr_scheduler.MultiStepLR
    elif args.scheduler == 'step':
        sche_kwargs = {'step_size': args.step_size, 'gamma': args.lr_decay, 'last_epoch': args.last_epoch}
        scheduler = optim.lr_scheduler.StepLR
    elif args.scheduler == 'cosine':
        sche_kwargs = {'T_max': args.total_iters, 'eta_min': args.eta_min, 'last_epoch': args.last_epoch}
        scheduler = optim.lr_scheduler.CosineAnnealingLR
    elif args.scheduler == 'poly':
        sche_kwargs = {'total_iters': args.total_iters, 'power': args.power, 'last_epoch': args.last_epoch}
        scheduler = optim.lr_scheduler.PolynomialLR
    else:
        raise ValueError(f'Unknown scheduler: {args.scheduler}')
    print(f'Scheduler: {args.scheduler}', sche_kwargs)
    return scheduler, sche_kwargs