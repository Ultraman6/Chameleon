import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

def evaluate_calibration(device, dataloader, net, n_bins=15):
    """
    在一个给定的数据集上评估模型的校准性能。
    """
    net.eval()

    all_logits = []
    all_labels = []
    # ⭐ tqdm 包装 dataloader
    for inputs, labels in tqdm(dataloader, desc="Collecting logits/labels"):
        inputs, labels = inputs.to(device), labels.to(device)
        logits = net(inputs)
        all_logits.append(logits)
        all_labels.append(labels)

    full_logits = torch.cat(all_logits)
    full_labels = torch.cat(all_labels)

    num_samples = len(full_labels)
    val_size = num_samples // 2

    logits_val_t, labels_val_t = full_logits[:val_size], full_labels[:val_size]
    logits_test_t, labels_test_t = full_logits[val_size:], full_labels[val_size:]

    logits_test_np = logits_test_t.cpu().detach().numpy()
    labels_test_np = labels_test_t.cpu().detach().numpy()
    probs_test_np = F.softmax(logits_test_t, dim=1).cpu().detach().numpy()

    logits_val_np = logits_val_t.cpu().detach().numpy()
    labels_val_np = labels_val_t.cpu().detach().numpy()

    ece_val, reliability_data = calculate_ece(
        probs_test_np, labels_test_np, n_bins=n_bins, return_bins=True
    )
    tce_val, optimal_temp_val = calculate_tce(
        logits_val_np, labels_val_np, logits_test_np, labels_test_np, n_bins=n_bins
    )
    class_ece_val = calculate_classwise_ece(probs_test_np, labels_test_np, n_bins=n_bins)
    ada_ece_val = calculate_adaece(probs_test_np, labels_test_np, n_bins=n_bins)

    results = {
        'ece': ece_val,
        'reliability': reliability_data,
        'tce': tce_val,
        'optimal_temp': optimal_temp_val,
        'class-ece': class_ece_val,
        'ada-ece': ada_ece_val
    }

    return results


def calculate_ece(probs, labels, n_bins=15, return_bins=False):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    ece = 0.0
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_data = {"conf": [], "acc": [], "prop": []}

    # ⭐ tqdm 包装 bin 循环
    for i in tqdm(range(n_bins), desc="Calculating ECE bins"):
        bin_low = bin_boundaries[i]
        bin_high = bin_boundaries[i + 1]
        if i < n_bins - 1:
            in_bin = (confidences >= bin_low) & (confidences < bin_high)
        else:
            in_bin = (confidences >= bin_low) & (confidences <= bin_high)

        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(conf_in_bin - acc_in_bin) * prop_in_bin

            if return_bins:
                bin_data["conf"].append(conf_in_bin)
                bin_data["acc"].append(acc_in_bin)
                bin_data["prop"].append(prop_in_bin)

    return (ece, bin_data) if return_bins else ece


def calculate_adaece(probs, labels, n_bins=15):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    n = len(confidences)
    sorted_idx = np.argsort(confidences)
    bin_size = n // n_bins

    adaece = 0.0
    # ⭐ tqdm 包装 bin 循环
    for i in tqdm(range(n_bins), desc="Calculating AdaECE bins"):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else n
        idx_bin = sorted_idx[start:end]

        if len(idx_bin) == 0:
            continue

        acc_in_bin = np.mean(accuracies[idx_bin])
        conf_in_bin = np.mean(confidences[idx_bin])
        adaece += (len(idx_bin) / n) * np.abs(conf_in_bin - acc_in_bin)

    return adaece


def calculate_tce(logits_val, labels_val, logits_test, labels_test, n_bins=15):
    logits_val_t = torch.from_numpy(logits_val).float()
    labels_val_t = torch.from_numpy(labels_val).long()
    logits_test_t = torch.from_numpy(logits_test).float()

    temperature = nn.Parameter(torch.ones(1) * 1.5)
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=100)

    def eval_loss():
        optimizer.zero_grad()
        loss = nll_criterion(logits_val_t / temperature, labels_val_t)
        loss.backward()
        return loss

    # ⭐ tqdm 不好直接加在 LBFGS，这里保持原样
    optimizer.step(eval_loss)
    optimal_temp = temperature.item()

    calibrated_logits = logits_test_t / optimal_temp
    calibrated_probs = F.softmax(calibrated_logits, dim=1).detach().numpy()

    tce = calculate_ece(calibrated_probs, labels_test, n_bins)
    return tce, optimal_temp

def calculate_classwise_ece(probs, labels, n_bins=15):
    n_classes = probs.shape[1]
    classwise_ece_list = []

    # ⭐ tqdm 包装类别循环
    for c in tqdm(range(n_classes), desc="Calculating Classwise ECE"):
        class_mask = (labels == c)
        if np.sum(class_mask) == 0:
            continue

        class_conf = probs[class_mask, c]
        class_preds = np.argmax(probs[class_mask], axis=1)
        class_acc = (class_preds == c)

        c_ece = 0.0
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

        for i in range(n_bins):
            bin_low = bin_boundaries[i]
            bin_high = bin_boundaries[i + 1]
            if i < n_bins - 1:
                in_bin = (class_conf >= bin_low) & (class_conf < bin_high)
            else:
                in_bin = (class_conf >= bin_low) & (class_conf <= bin_high)

            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                acc_in_bin = np.mean(class_acc[in_bin])
                conf_in_bin = np.mean(class_conf[in_bin])
                c_ece += np.abs(conf_in_bin - acc_in_bin) * prop_in_bin

        classwise_ece_list.append(c_ece)

    return np.mean(classwise_ece_list) if classwise_ece_list else 0.0

# --- 示例使用 ---
if __name__ == '__main__':
    # 模拟一个简单的网络和数据加载器
    class SimpleNet(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.fc = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            return self.fc(x)

    class Args:
        n_bins = 15

    # 生成一些模拟数据
    np.random.seed(0)
    num_samples = 1000
    num_classes = 10
    true_labels = np.random.randint(0, num_classes, num_samples)
    # 模拟一个过度自信的模型
    logits_unc = np.random.randn(num_samples, num_classes) * 2
    logits_unc[np.arange(num_samples), true_labels] += np.random.uniform(5, 10, num_samples)
    error_indices = np.random.choice(num_samples, size=int(num_samples * 0.15), replace=False)
    logits_unc[error_indices] = np.random.randn(len(error_indices), num_classes) * 5

    # 创建一个伪网络，其输出就是我们生成的logits
    pseudo_net = SimpleNet(1, num_classes)
    pseudo_net.fc.weight.data.zero_()
    pseudo_net.fc.bias.data = torch.from_numpy(logits_unc).float()

    # 创建一个伪数据加载器
    # 输入数据可以是任意的，因为网络的输出已经被我们固定了
    dummy_inputs = torch.arange(num_samples).float().view(-1, 1)
    dummy_labels = torch.from_numpy(true_labels)
    dataset = torch.utils.data.TensorDataset(dummy_inputs, dummy_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100)

    # 运行评估
    print("\n--- 5. 运行完整的校准评估函数 ---")
    args = Args()
    device = torch.device("cpu")


    # 为了匹配函数签名，我们伪造一个net，但其输出由我们的logits决定
    # 在实际使用中，net应该是你训练好的真实模型
    def mocked_net(inputs):
        return pseudo_net.fc.bias.data[inputs.long().squeeze()]


    calibration_results = evaluate_calibration(args, device, dataloader, mocked_net)

    print(f"评估结果:")
    print(f"  ECE: {calibration_results['ece']:.4f}")
    print(f"  TCE: {calibration_results['tce']:.4f}")
    print(f"  Optimal Temperature: {calibration_results['optimal_temp']:.4f}")
    print(f"  Classwise ECE: {calibration_results['class-ece']:.4f}")
    print(f"  Reliability Data (第一个区间): {list(calibration_results['reliability'].items())[0]}")

