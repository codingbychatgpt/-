import os
import numpy as np

def generate_signals(snr_db, n_symbols=20000):
    # 1. 生成随机比特流
    bits = np.random.randint(0, 2, n_symbols * 2)  # 每个符号2个比特
    bits_reshaped = bits.reshape(-1, 2)  # 每两个比特作为一个符号

    # 2. QPSK 映射规则：00 -> 45°, 01 -> 135°, 11 -> -135°, 10 -> -45°
    mapping = {
        (0, 0): (1, 1),   # 45°
        (0, 1): (-1, 1),  # 135°
        (1, 1): (-1, -1), # -135°
        (1, 0): (1, -1)   # -45°
    }

    # 3. 根据映射生成 I 和 Q 分量
    I = np.array([mapping[tuple(b)][0] for b in bits_reshaped])
    Q = np.array([mapping[tuple(b)][1] for b in bits_reshaped])

    # 4. 生成复基带 QPSK 符号（I + jQ），归一化功率
    qpsk_symbols = (I + 1j * Q) / np.sqrt(2)

    # 5. 加入复高斯白噪声（AWGN）
    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.mean(np.abs(qpsk_symbols) ** 2)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(qpsk_symbols)) + 1j * np.random.randn(len(qpsk_symbols))
    )

    # 有信号：QPSK + 噪声
    qpsk_signal_noisy = qpsk_symbols + noise

    # 无信号：仅噪声（与有信号样本同样长度和噪声功率）
    qpsk_signal_no_signal = np.sqrt(noise_power / 2) * (
        np.random.randn(len(qpsk_symbols)) + 1j * np.random.randn(len(qpsk_symbols))
    )

    # 合并有信号和无信号数据
    X = np.concatenate((qpsk_signal_noisy, qpsk_signal_no_signal), axis=0)
    Y = np.concatenate((np.ones(len(qpsk_signal_noisy)), np.zeros(len(qpsk_signal_no_signal))))

    # 将复数信号拆分为二维 I/Q 特征
    X_2d = np.column_stack((X.real, X.imag))

    return qpsk_signal_noisy, qpsk_signal_no_signal, X_2d, Y


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    snr_db_list = [-10, -5, 0, 5, 10]
    n_symbols = 50000

    for snr_db in snr_db_list:
        _, _, X_2d, Y = generate_signals(snr_db, n_symbols=n_symbols)
        snr_tag = f"snr{snr_db}"
        np.save(os.path.join(data_dir, f"X_data_2d_{snr_tag}.npy"), X_2d)
        np.save(os.path.join(data_dir, f"Y_labels_{snr_tag}.npy"), Y)
        print(f"[SNR {snr_db} dB] X_2d shape: {X_2d.shape}, Y shape: {Y.shape}")
