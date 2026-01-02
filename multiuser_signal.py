import numpy as np
import random

# 固定随机种子，保证数据生成可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def qpsk_symbols(n_symbols):
    bits = np.random.randint(0, 2, n_symbols * 2)
    bits_reshaped = bits.reshape(-1, 2)
    mapping = {
        (0, 0): (1, 1),
        (0, 1): (-1, 1),
        (1, 1): (-1, -1),
        (1, 0): (1, -1),
    }
    i = np.array([mapping[tuple(b)][0] for b in bits_reshaped])
    q = np.array([mapping[tuple(b)][1] for b in bits_reshaped])
    return (i + 1j * q) / np.sqrt(2)


def generate_multiuser_samples(
    snr_db,
    num_samples,
    m_users=10,
    n_points=256,
    fading="rayleigh",
):
    samples = []
    labels = []

    snr_linear = 10 ** (snr_db / 10)

    for _ in range(num_samples):
        is_signal = np.random.rand() > 0.5
        labels.append(1 if is_signal else 0)

        if is_signal:
            s = qpsk_symbols(n_points)
            signal_power = np.mean(np.abs(s) ** 2)
            noise_power = signal_power / snr_linear
        else:
            signal_power = 1.0
            noise_power = signal_power / snr_linear

        x = np.zeros((m_users, n_points), dtype=np.complex128)

        for m in range(m_users):
            if fading == "rayleigh":
                h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
            else:
                h = 1.0

            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(n_points) + 1j * np.random.randn(n_points)
            )

            if is_signal:
                x[m, :] = h * s + noise
            else:
                x[m, :] = noise

        samples.append(x)

    return np.array(samples), np.array(labels)
