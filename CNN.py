import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import TensorDataset, DataLoader, Subset
import random

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
snr_db_list = [-10, -5, 0, 5, 10]
pf_target = 0.1

# 固定随机种子，保证划分与训练可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_indices(labels, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    def split_one(indices):
        if len(indices) < 2:
            return indices, np.array([], dtype=int)
        test_count = max(1, int(round(test_ratio * len(indices))))
        if len(indices) - test_count < 1:
            test_count = len(indices) - 1
        test_idx = indices[:test_count]
        train_idx = indices[test_count:]
        return train_idx, test_idx

    train0, test0 = split_one(idx0)
    train1, test1 = split_one(idx1)
    train_indices = np.concatenate([train0, train1])
    test_indices = np.concatenate([test0, test1])
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices

# 定义 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 第一卷积层 + 池化层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # 输入通道1(2x2协方差)，输出通道32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # 增加 padding，减少尺寸缩小
        
        # 第二卷积层 + 池化层
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # 增加 padding
        
        # 全连接层（对应 2x2 输入）
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 1)  # 最后一层输出 1，表示是否有信号
    
    def forward(self, x):
        # 卷积 + 池化 + 激活函数（ReLU）
        x = self.pool1(torch.relu(self.conv1(x)))  # 卷积 + ReLU + 池化
        x = self.pool2(torch.relu(self.conv2(x)))  # 第二次卷积 + ReLU + 池化
        
        # 展平输出结果，以便进入全连接层
        x = torch.flatten(x, 1)  # 使用 flatten() 扁平化张量
        x = torch.relu(self.fc1(x))  # 第一全连接层
        x = torch.sigmoid(self.fc2(x))  # 输出概率（0 或 1）

        return x

epochs = 10
pd_list = []
pf_list = []
acc_list = []
auc_list = []
ed_pd_list = []
ed_pf_list = []
ed_acc_list = []
ed_auc_list = []
snr_used = []

for snr_db in snr_db_list:
    snr_tag = f"snr{snr_db}"
    cov_path = os.path.join(data_dir, f"cov_matrices_raw_{snr_tag}.npy")
    label_path = os.path.join(data_dir, f"cov_frame_labels_{snr_tag}.npy")
    user_cov_path = os.path.join(data_dir, f"cov_user_iq_raw_{snr_tag}.npy")
    energy_path = os.path.join(data_dir, f"user_energy_{snr_tag}.npy")
    if not os.path.exists(user_cov_path) or not os.path.exists(label_path):
        print(f"[SNR {snr_db} dB] Missing data files, skipped.")
        continue

    # 加载每用户 2x2 IQ 协方差矩阵和对应标签
    cov_user_raw = np.load(user_cov_path)
    labels = np.load(label_path)

    num_samples, m_users, _, _ = cov_user_raw.shape
    cov_user_flat = cov_user_raw.reshape(num_samples * m_users, 2, 2)
    labels_user = np.repeat(labels, m_users)
    sample_ids = np.repeat(np.arange(num_samples), m_users)

    cov_matrices_tensor = torch.tensor(cov_user_flat, dtype=torch.float32).unsqueeze(1)
    labels_tensor = torch.tensor(labels_user, dtype=torch.float32).unsqueeze(1)
    sample_ids_tensor = torch.tensor(sample_ids, dtype=torch.long)

    assert len(cov_matrices_tensor) == len(labels_tensor), "Data and labels length do not match!"

    dataset = TensorDataset(cov_matrices_tensor, labels_tensor, sample_ids_tensor)
    train_sample_idx, test_sample_idx = split_indices(labels)
    train_mask = np.isin(sample_ids, train_sample_idx)
    test_mask = np.isin(sample_ids, test_sample_idx)
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    # 按训练集统计量做标准化，避免数据泄漏
    train_mean = np.mean(cov_user_flat[train_indices], axis=0, keepdims=True)
    train_std = np.std(cov_user_flat[train_indices], axis=0, keepdims=True)
    train_std[train_std == 0] = 1.0
    cov_user_norm = (cov_user_flat - train_mean) / train_std

    cov_matrices_tensor = torch.tensor(cov_user_norm, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(cov_matrices_tensor, labels_tensor, sample_ids_tensor)

    train_data = Subset(dataset, train_indices.tolist())
    test_data = Subset(dataset, test_indices.tolist())

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    # 模型实例化
    model = SimpleCNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    for epoch in range(epochs):
        for inputs, label, _ in train_loader:
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        print(f"[SNR {snr_db} dB] Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    # 模型评估
    model.eval()
    all_outputs = []
    all_labels = []
    all_sample_ids = []

    with torch.no_grad():
        for inputs, batch_labels, batch_sample_ids in test_loader:
            outputs = model(inputs)
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())
            all_sample_ids.append(batch_sample_ids.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0).ravel()
    all_labels = np.concatenate(all_labels, axis=0).ravel()
    all_sample_ids = np.concatenate(all_sample_ids, axis=0).ravel()

    order = np.argsort(all_sample_ids)
    outputs_sorted = all_outputs[order]
    labels_sorted = all_labels[order]
    ids_sorted = all_sample_ids[order]
    unique_ids, start_idx, counts = np.unique(ids_sorted, return_index=True, return_counts=True)
    sums = np.add.reduceat(outputs_sorted, start_idx)
    fused_outputs = sums / counts
    fused_labels = labels_sorted[start_idx]
    pred_labels = (fused_outputs > 0.5).astype(int)

    tp = np.sum((pred_labels == 1) & (fused_labels == 1))
    fn = np.sum((pred_labels == 0) & (fused_labels == 1))
    fp = np.sum((pred_labels == 1) & (fused_labels == 0))
    tn = np.sum((pred_labels == 0) & (fused_labels == 0))

    pd = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    pf = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else float("nan")

    fpr, tpr, _ = roc_curve(fused_labels, fused_outputs)
    roc_auc = auc(fpr, tpr)

    pd_list.append(pd)
    pf_list.append(pf)
    acc_list.append(acc)
    auc_list.append(roc_auc)
    snr_used.append(snr_db)

    print(f"[SNR {snr_db} dB] Fusion Pd={pd:.4f}, Pf={pf:.4f}, Acc={acc:.4f}, AUC={roc_auc:.4f}")
    print(f"[SNR {snr_db} dB] Confusion Matrix: TP={tp}, FN={fn}, FP={fp}, TN={tn}")

    if not os.path.exists(energy_path):
        print(f"[SNR {snr_db} dB] Missing energy data, ED skipped.")
        ed_pd_list.append(float("nan"))
        ed_pf_list.append(float("nan"))
        ed_acc_list.append(float("nan"))
        ed_auc_list.append(float("nan"))
        continue

    user_energy = np.load(energy_path)
    energy_fused = np.mean(user_energy, axis=1)
    train_energy = energy_fused[train_sample_idx]
    test_energy = energy_fused[test_sample_idx]
    train_labels = labels[train_sample_idx]
    test_labels = labels[test_sample_idx]

    noise_train = train_energy[train_labels == 0]
    if len(noise_train) == 0:
        threshold = np.quantile(train_energy, 1 - pf_target)
    else:
        threshold = np.quantile(noise_train, 1 - pf_target)

    pred_ed = (test_energy > threshold).astype(int)
    tp_ed = np.sum((pred_ed == 1) & (test_labels == 1))
    fn_ed = np.sum((pred_ed == 0) & (test_labels == 1))
    fp_ed = np.sum((pred_ed == 1) & (test_labels == 0))
    tn_ed = np.sum((pred_ed == 0) & (test_labels == 0))

    pd_ed = tp_ed / (tp_ed + fn_ed) if (tp_ed + fn_ed) > 0 else float("nan")
    pf_ed = fp_ed / (fp_ed + tn_ed) if (fp_ed + tn_ed) > 0 else float("nan")
    acc_ed = (tp_ed + tn_ed) / (tp_ed + tn_ed + fp_ed + fn_ed) if (tp_ed + tn_ed + fp_ed + fn_ed) > 0 else float("nan")

    fpr_ed, tpr_ed, _ = roc_curve(test_labels, test_energy)
    auc_ed = auc(fpr_ed, tpr_ed)

    ed_pd_list.append(pd_ed)
    ed_pf_list.append(pf_ed)
    ed_acc_list.append(acc_ed)
    ed_auc_list.append(auc_ed)

    print(f"[SNR {snr_db} dB] ED Pd={pd_ed:.4f}, Pf={pf_ed:.4f}, Acc={acc_ed:.4f}, AUC={auc_ed:.4f}, Thr={threshold:.4f}")

if pd_list:
    plt.figure()
    plt.plot(snr_used, pd_list, marker='o', label='Fusion Pd')
    plt.plot(snr_used, pf_list, marker='o', label='Fusion Pf')
    if ed_pd_list:
        plt.plot(snr_used, ed_pd_list, marker='s', linestyle='--', label='ED Pd')
        plt.plot(snr_used, ed_pf_list, marker='s', linestyle='--', label='ED Pf')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Probability')
    plt.title('Pd/Pf vs SNR')
    plt.legend()
    plt.show()
