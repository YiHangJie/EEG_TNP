import numpy as np
import torch


def EA_R_inv_sqrt(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    covs = np.einsum('nij,nkj->nik', X, X)
    R = covs.mean(axis=0)

    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.clip(eigvals, eps, None)
    return (eigvecs * eigvals**-0.5) @ eigvecs.T


def eeg_alignment(X: np.ndarray, R_inv_sqrt: np.ndarray) -> np.ndarray:
    return np.einsum('ij,jk->ik', R_inv_sqrt, X)


def mean_std_normalize(X: np.ndarray, axis: int = 1) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=axis, keepdims=True)
    std = X.std(axis=axis, keepdims=True)
    std[std == 0] = 1
    return (X - mean) / std


def finalize_eeg_sample(X: np.ndarray) -> torch.Tensor:
    X = mean_std_normalize(X, axis=1)
    X = np.asarray(X[np.newaxis, ...], dtype=np.float32)
    return torch.from_numpy(X)
