import itertools
import math
import os
import time
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from algo import build_algo, build_optimizer, build_scheduler
from datasets import build_dataset
from models import build_model
from utils.utils import set_seed, get_datetime, accuracy
from utils.logger import MetricsTracker

# Training
def train(trainloader, device, optimizer, epoch) -> dict:
    result = {}
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        res = optimizer.step(inputs, targets, epoch=epoch)
        for k, v in res.items():
            if k not in result:
                result[k] = 0.0
            if k in ['time', 'batch_size']:
                result[k] += v
            else:
                result[k] += v * res['batch_size']
    return {k: v / result['batch_size'] if k != 'time' else v for k, v in result.items() if k != 'batch_size'}

def test(testloader, net, criterion, device) -> tuple:
    net.eval()
    losses, acc_top1, acc_top5, total = 0, 0.0, 0.0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total += batch_size
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            losses += loss.item() * batch_size
            acc = accuracy(outputs, targets, topk=(1, 5))
            acc_top1 += acc[0].item() * batch_size
            acc_top5 += acc[1].item() * batch_size

    return losses / total, acc_top1 / total, acc_top5 / total

def base_save_folder(args, dataset, data_kwargs, optim_kwargs, sche_kwargs):
    data_info = '_'.join([dataset, *(f'{k}={v}' for k, v in data_kwargs.items())])
    optim_info = '_'.join([args.optimizer, *(f'{k}={v}' for k, v in optim_kwargs.items())])
    sche_info = '_'.join([args.scheduler, *(f'{k}={v}' for k, v in sche_kwargs.items())])
    return os.path.join(f'{args.model}-{data_info}', f'{optim_info}-{sche_info}')

def algo_save_folder(base_folder, algo_kwargs):
    algo_info = '_'.join([*(f'{k}={v}' for k, v in algo_kwargs.items())])
    return str(os.path.join(base_folder, algo_info))

def find_last_checkpoint(model_save_dir, epoch):
    # 返回轮次及其路径
    files = [f for f in os.listdir(model_save_dir) if f.startswith('checkpoint_') and f.endswith('.pt')]
    if not files:
        return 1, None
    epochs = [int(f.split('_')[1].split('.')[0]) for f in files]
    closest_epoch = min(epochs, key=lambda x: abs(x - epoch))
    # 获取模型路径
    closest_file = str(os.path.join(model_save_dir, f'checkpoint_{closest_epoch}.pt'))
    return closest_epoch, closest_file

def statistic_result(algo, df, results):
    results[algo]['train_loss'].append(df['train_loss'].min())
    results[algo]['train_loss'].append(df['test_loss'].min())
    results[algo]['train_acc'].append(df['train_acc'].max())
    results[algo]['test_acc'].append(df['test_acc'].max())

rho_metrics = ['rho_mean', 'rho_std', 'rho_max', 'rho_min', 'rho_cv']
best_names = ['train_loss', 'train_acc', 'test_loss', 'test_acc_top1', 'test_acc_top5']
last_names = ['test_acc_top1', 'test_acc_top5']

def save_rho_statistic(tracker, optimizer, epoch):
    rho_layer_stats = optimizer.get_rho_stats_by_layer()
    for metric, value in rho_layer_stats.items():
        tracker.track({k: v for k, v in value.items()}, f"rho_{metric}", {'epoch': epoch})

def save_mask_statistic(tracker, optimizer, epoch):
    mask_layer_stats = optimizer.get_mask_stats_by_layer()
    tracker.track({k: v for k, v in mask_layer_stats.items()}, 'mask', {'epoch': epoch})

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', default='records', type=str)
    parser.add_argument('--save_freq', default=20, type=int, help='save every save_epochs')
    parser.add_argument('--num_workers', default=16, type=int, help='number of workers for data loader')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', default=[0], nargs='+', type=int)
    # erm: 2 0 SAM: 0 2
    parser.add_argument('--memmap', type=bool, default=False)
    parser.add_argument('--reload_data', type=bool, default=True)
    parser.add_argument('--reload_model', type=bool, default=True)
    parser.add_argument('--reload_algo', type=str, default='')

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--pre_epoch', default=0, type=int)
    parser.add_argument('--eval_epoch', default=160, type=int)
    parser.add_argument('--shut_epoch', default=-1, type=int)
    parser.add_argument('--batch_size', default=[128], type=int, nargs='+')

    parser.add_argument('--optimizer', default='sgd', help='optimizer: sgd | adamw')
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--nesterov', default=False, type=bool)

    parser.add_argument('--scheduler', default='cosine', choices=['constant', 'step', 'multistep', 'cosine', 'poly'])
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='learning rate decay rate')
    parser.add_argument('--step_size', default=200, type=int)
    parser.add_argument('--last_epoch', default=-1, type=int)
    parser.add_argument('--total_iters', default=200, type=int)
    parser.add_argument('--milestones', nargs='+', default=[150, 225, 275], type=int)
    parser.add_argument('--eta_min', default=0.0, type=float)
    parser.add_argument('--power', default=1.0, type=float)

    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--dataset', default=['cifar10'], nargs='+')
    parser.add_argument('--classes', nargs='+', default=[])
    # pacs: 'art_painting', 'cartoon', 'photo', 'sketch'
    # office_caltech10: 'Caltech', 'amazon', 'dslr', 'webcam'
    # domainnet: 'clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'
    parser.add_argument('--leave_domain', default=[], type=str, nargs='+')
    parser.add_argument('--aug', default=['cutout'], choices=['basic', 'cutout', 'cutmix'], nargs='+')
    parser.add_argument('--mix_alpha', default=1.0, type=float, help='alpha for mixup and cutmix')
    parser.add_argument('--noise_ratio', default=[0.0], type=float, nargs='+')
    parser.add_argument('--noise_mode', default='sym', type=str, choices=['sym', 'asym'])
    parser.add_argument('--label_smoothing', default=0.1, type=float)
    # sam
    parser.add_argument("--algo", default=['car'], type=str, nargs='+')
    parser.add_argument("--rho", default=[0.1], type=float, nargs='+')
    parser.add_argument("--start_sam", default=1, type=int)
    parser.add_argument("--end_sam", default=-1, type=int)
    parser.add_argument("--opt_step", default=5, type=int)
    # salp
    parser.add_argument("--rho_min", default=-math.inf , type=float, nargs='+')
    parser.add_argument("--rho_max", default=math.inf, type=float, nargs='+')
    parser.add_argument("--rho_lr", default=[1.0], type=float, nargs='+')
    # fishersam
    parser.add_argument("--fishersam_alpha", default=0.01, type=float, nargs='+')
    # fsam
    parser.add_argument("--fsam_sigma", default=1.0, type=float, nargs='+')
    parser.add_argument("--fsam_lmbda", default=[0.7], type=float, nargs='+', help='0.6[resnet18 vgg16 wrn16-2] 0.9[wrn28-10 pyramidnet-110]')
    # vasso
    parser.add_argument("--vasso_theta", default=[0.4], type=float, nargs='+', help='0.4-cifar10 0.7-cifar100')
    # u2p
    parser.add_argument("--base", default=[0.5], type=float, nargs='+')
    # ssam & scar
    parser.add_argument("--sparsity", default=[0.5], type=float, nargs='+')
    parser.add_argument("--score_mode", default=['curt'], type=str, nargs='+')
    parser.add_argument("--drop_rate", default=[0.5], type=float, nargs='+')
    parser.add_argument("--gen_mode", default=['rand'], type=str, nargs='+')
    # car & scar
    parser.add_argument("--alpha", default=[0.01], type=float, nargs='+')
    parser.add_argument("--beta", default=[0.01], type=float, nargs='+')

    args = parser.parse_args()
    args.time = get_datetime()
    return args

def run(dataset, args, device, ld, nr, bs, aug):
    if dataset in ['pacs', 'office_caltech10', 'domainnet']:
        data_kwargs = {'bs': bs, 'ld': ld}
    else:
        data_kwargs = {'bs': bs, 'aug': aug, 'nr': nr}
    print('Data Augmentation: ' + str(data_kwargs))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    base_optim, optim_kwargs = build_optimizer(args)
    sche, sche_kwargs = build_scheduler(args)
    base_dir = os.path.join(args.save_root, base_save_folder(args, dataset, data_kwargs, optim_kwargs, sche_kwargs))
    algo_base_dir = os.path.join(base_dir, algo)
    os.makedirs(algo_base_dir, exist_ok=True)
    optim, algo_kwargs_list = build_algo(args, algo)

    algo_tracker = None
    if len(algo_kwargs_list) > 1:
        prefixes = algo_kwargs_list[0].keys()
        algo_record_file = str(os.path.join(algo_base_dir, f'{args.time}.xlsx'))
        algo_tracker = MetricsTracker(algo_record_file, args)
        for k in best_names:
            algo_tracker.set_sheet_prefix(f"best_{k}", ["seed", *prefixes])
        for k in last_names:
            algo_tracker.set_sheet_prefix(f"last_{k}", ["seed", *prefixes])

    for algo_kwargs in algo_kwargs_list:
        seed_tracker = None
        algo_folder = algo_save_folder(algo_base_dir, algo_kwargs)
        if len(args.seed) > 1:
            seed_record_file = str(os.path.join(algo_folder, f'{args.time}_seed={args.seed}.xlsx'))
            seed_tracker = MetricsTracker(seed_record_file, args)
            for k in best_names:
                seed_tracker.set_sheet_prefix(f"best_{k}", ["seed"])
            for k in last_names:
                seed_tracker.set_sheet_prefix(f"last_{k}", ["seed"])
        for seed in args.seed:
            set_seed(seed)
            trainloader, testloader, num_classes = build_dataset(ld, nr, bs, dataset, aug, args, base_dir, seed)
            net = build_model(args, base_dir, seed, num_classes=num_classes).to(device)
            train_loss, train_acc_top1, train_acc_top5 = test(trainloader, net, criterion, device)
            test_loss, test_acc_top1, test_acc_top5 = test(testloader, net, criterion, device)
            print(
                f'Epoch: 0 | train_loss: {train_loss: .4f} | train_acc_top1: {train_acc_top1: .4f} | '
                f'train_acc_top5: {train_acc_top5: .4f} | '
                f'test_loss: {test_loss: .4f} | test_acc_top1 {test_acc_top1: .4f} | '
                f'test_acc_top5 {test_acc_top5: .4f}'
            )
            optimizer = optim(net, criterion, base_optim, **algo_kwargs, **optim_kwargs)
            if args.pre_epoch >= 0:
                for param_group in optimizer.optimizer.param_groups:
                    param_group['initial_lr'] = args.lr
            scheduler = sche(optimizer.optimizer, **sche_kwargs)
            print(f'Algorithm: {algo} Config: {algo_kwargs}')
            algo_dir = str(os.path.join(algo_folder, f'seed={seed}'))
            os.makedirs(algo_dir, exist_ok=True)
            record_file = str(os.path.join(algo_dir, f'{args.time}.xlsx'))
            model_save_dir = str(os.path.join(algo_dir, args.time))
            tracker = MetricsTracker(record_file, args)
            tracker.set_sheet_prefix("performance", ["epoch"])
            if algo == 'salp':
                for metric in rho_metrics:
                    tracker.set_sheet_prefix(metric, ["epoch"])
            elif algo == 'mu2p':
                tracker.set_sheet_prefix('mu_by_layers', ["epoch"])
            elif algo == 'ssam':
                tracker.set_sheet_prefix('mask', ["epoch"])
            old_model_save_dir = str(os.path.join(algo_dir, args.reload_algo))
            if args.reload_algo and os.path.exists(old_model_save_dir):
                args.pre_epoch, save_checkpoint = find_last_checkpoint(old_model_save_dir, args.pre_epoch)
                if args.pre_epoch > 0:
                    print(f"Resuming from last saved epoch {args.pre_epoch} for algorithm {algo} with seed {seed}.")
                    for _ in tqdm(range(args.pre_epoch)):
                        for _, _ in trainloader:
                            pass  # 恢复数据装载器的随机状态
                    optimizer.load(save_checkpoint, scheduler=scheduler)
                    if algo == 'salp':
                        save_rho_statistic(tracker, optimizer, args.pre_epoch)
            else:
                tracker.track(
                    {"train_loss": train_loss, "train_acc_top1": train_acc_top1, "train_acc_top5": train_acc_top5,
                     'test_loss': test_loss, 'test_acc_top1': test_acc_top1, 'test_acc_top5': test_acc_top5},
                    "performance",
                    {"epoch": args.pre_epoch}
                )
            best_records = {k: None for k in best_names}
            best_epochs = {k: 0 for k in best_names}
            best_values = {k: 0.0 for k in best_names}
            last_values = {k: 0.0 for k in last_names}
            try:
                for epoch in range(args.pre_epoch + 1, args.epochs + 1):
                    if epoch == args.shut_epoch + 1:
                        break
                    st = time.time()
                    net.train()
                    result = train(trainloader, device, optimizer, epoch)
                    if algo == 'salp':
                        save_rho_statistic(tracker, optimizer, epoch)
                    elif algo == 'sala':
                        optimizer.backup_and_load_cache()
                    elif algo == 'mu2p':
                        tracker.track(optimizer.get_mu_by_layers(),
                                      'mu_by_layers', {'epoch': epoch})
                    elif algo == 'ssam':
                        cosine_term = 0.5 * (1.0 + math.cos(math.pi * max(epoch, 1) / max(args.epochs, 1)))
                        optimizer.dr = optimizer.drop_rate * cosine_term
                        save_mask_statistic(tracker, optimizer, epoch)
                    elif algo == 'car':
                        cosine_term = 0.5 * (1.0 + math.cos(math.pi * max(epoch, 1) / max(args.epochs, 1)))
                        optimizer.rho_ = optimizer.rho * cosine_term
                    scheduler.step()
                    print(f"Epoch {epoch}/{args.epochs} | Algo: {algo} | Train | " + ' | '.join(
                        [f"{k}: {v:.4f}" for k, v in result.items()]))
                    tracker.track({f'train_{k}': v for k, v in result.items()}, "performance", {"epoch": epoch})
                    if epoch > args.eval_epoch:
                        test_loss, test_acc_top1, test_acc_top5 = test(testloader, net, criterion, device)
                        tracker.track(
                            {"test_loss": test_loss, "test_acc_top1": test_acc_top1, "test_acc_top5": test_acc_top5},
                            "performance",
                            {"epoch": epoch}
                        )
                        print(f"Epoch {epoch}/{args.epochs} | Algo: {algo} | Test | "
                              f"loss: {test_loss:.4f} | "
                              f"acc_top1: {test_acc_top1: .4f} | acc_top5: {test_acc_top5: .4f}")

                        if algo == 'erm':
                            bv = {'train_loss': -result['loss'], 'train_acc': result['acc'], 'test_loss': -test_loss,
                                  'test_acc_top1': test_acc_top1, 'test_acc_top5': test_acc_top5}
                        else:
                            bv = {'train_loss': -result['origin_loss'], 'train_acc': result['origin_acc'],
                                  'test_loss': -test_loss,
                                  'test_acc_top1': test_acc_top1, 'test_acc_top5': test_acc_top5}

                        for k in best_values:
                            e, v = best_epochs[k], best_values[k]
                            v_ = bv[k]
                            if k in best_records:
                                r = best_records[k]
                                if r is None:
                                    best_records[k] = optimizer.get(scheduler=scheduler)
                                    best_epochs[k] = epoch
                                    best_values[k] = v_
                                elif v_ >= v:
                                    best_records[k] = optimizer.get(scheduler=scheduler)
                                    best_epochs[k] = epoch
                                    best_values[k] = v_
                            else:
                                if v_ >= v:
                                    best_epochs[k] = epoch
                                    best_values[k] = v_

                        if epoch > args.epochs - 10:
                            for k in last_values:
                                last_values[k] += bv[k]

                    if epoch >= args.eval_epoch and epoch % args.save_freq == 0:
                        os.makedirs(model_save_dir, exist_ok=True)
                        save_path = os.path.join(model_save_dir, f'checkpoint_{epoch}.pt')
                        optimizer.save(save_path, scheduler=scheduler)
                        print(f"Checkpoint and metrics saved for epoch {epoch}.")

                    print(f"Total training time for {epoch} epoch algo {algo}: {time.time() - st:.5f}s")
            finally:
                tracker.save()
                print(f"Metrics saved to {record_file} for {algo} with {algo_kwargs} in seed:{seed}.")
            # Save the best records
            for k in best_records:
                r, e, v = best_records[k], best_epochs[k], best_values[k]
                save_path = os.path.join(model_save_dir, f'checkpoint_{e}_best_{k}.pt')
                optimizer.save(save_path, state_dict=r)
                print(f"Best Checkpoint saved for epoch {e} with {k}={v}.")

            if seed_tracker is not None:
                for k in best_values:
                    seed_tracker.track(
                        {'value': best_values[k],
                         'epoch': best_epochs[k]},
                        f"best_{k}",
                        {"seed": seed}
                    )

                for k in last_values:
                    seed_tracker.track(
                        {'value': last_values[k]},
                        f"last_{k}",
                        {"seed": seed}
                    )

            if algo_tracker is not None:
                prefix_values = {"seed": seed, **algo_kwargs}
                for k in best_values:
                    algo_tracker.track({'value': best_values[k], 'epoch': best_epochs[k]}
                                       , f"best_{k}", prefix_values)

                for k in last_values:
                    algo_tracker.track({'value': last_values[k]}
                                       , f"last_{k}", prefix_values)

        if seed_tracker is not None:
            seed_tracker.save()

    if algo_tracker is not None:
        algo_tracker.save()

if __name__ == '__main__':
    args = get_args()
    device = args.device
    print('Current devices: ' + str(torch.cuda.current_device()))
    for algo, dataset in itertools.product(args.algo, args.dataset):
        if algo in ['pacs', 'domainnet']:
            for ld, nr, bs, aug in itertools.product(args.leave_domain, args.noise_ratio, args.batch_size, args.aug):
                run(dataset, args, device, ld, nr, bs, aug)
        else:
            for nr, bs, aug in itertools.product(args.noise_ratio, args.batch_size, args.aug):
                run(dataset, args, device, args.leave_domain, nr, bs, aug)
