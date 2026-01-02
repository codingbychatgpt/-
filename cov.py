import os
import numpy as np
from multiuser_signal import generate_multiuser_samples

# 固定随机种子，保证数据生成可复现
np.random.seed(42)

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)
snr_db_list = [-10, -5, 0, 5, 10]
m_users = 10
n_points = 256
num_samples = 2000

def compute_iq_cov(x):
    z = np.stack([x.real, x.imag], axis=0)
    return (z @ z.T) / z.shape[1]

for snr_db in snr_db_list:
    samples, labels = generate_multiuser_samples(
        snr_db,
        num_samples=num_samples,
        m_users=m_users,
        n_points=n_points,
    )

    cov_matrices = []
    user_energy = []
    for x in samples:
        r = (x @ x.conj().T) / n_points
        cov_matrices.append(np.stack([r.real, r.imag], axis=0))
        user_energy.append(np.mean(np.abs(x) ** 2, axis=1))

    cov_matrices = np.array(cov_matrices)
    user_energy = np.array(user_energy)

    # 每个用户的 2x2 IQ 协方差（保留原始值，标准化留给训练脚本按训练集统计）
    cov_user = np.empty((num_samples, m_users, 2, 2), dtype=np.float64)
    for i, sample in enumerate(samples):
        for m in range(m_users):
            cov_user[i, m] = compute_iq_cov(sample[m])

    snr_tag = f"snr{snr_db}"
    np.save(os.path.join(data_dir, f"cov_matrices_raw_{snr_tag}.npy"), cov_matrices)
    np.save(os.path.join(data_dir, f"cov_frame_labels_{snr_tag}.npy"), labels)
    np.save(os.path.join(data_dir, f"cov_user_iq_raw_{snr_tag}.npy"), cov_user)
    np.save(os.path.join(data_dir, f"user_energy_{snr_tag}.npy"), user_energy)

    print(f"[SNR {snr_db} dB] Saved {len(cov_matrices)} covariance matrices with matching labels.")
    print(f"[SNR {snr_db} dB] Cov shape: {cov_matrices.shape}")
    print(f"[SNR {snr_db} dB] User IQ cov shape: {cov_user.shape}")
    print(f"[SNR {snr_db} dB] Example cov (real part):\n{cov_matrices[0, 0]}")
