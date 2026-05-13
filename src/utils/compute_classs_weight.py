from utils import logger
import numpy as np
import torch

def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Args:
        labels: Array numpy berisi label (0 atau 1)

    Returns:
        Tensor PyTorch berisi bobot untuk setiap kelas [weight_ham, weight_spam]
    """
    # Hitung jumlah sampel per kelas
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    jumlah_kelas = len(unique_labels)

    # Hitung bobot inverse-proportional
    weights = total / (jumlah_kelas * counts)

    logger.info(f"Class weights — Ham (0): {weights[0]:.4f}, Spam (1): {weights[1]:.4f}")

    return torch.tensor(weights, dtype=torch.float32)