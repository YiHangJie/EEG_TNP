import os
import re
import mne
import numpy as np
from torcheeg.datasets import SEEDIVDataset, M3CVDataset, BCICIV2aDataset, TSUBenckmarkDataset
from torcheeg import transforms
from torcheeg.datasets.constants import SEED_IV_CHANNEL_LOCATION_DICT

def EA_R_inv_sqrt(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Euclidean Alignment (EA)
    -------------------------
    对每个样本矩阵 X_i (channels × time)，
    执行欧式对齐：X_i' = R^{-1/2} @ X_i，
    其中 R = mean_i (X_i @ X_i^T)。

    参数:
        X : np.ndarray, shape = (n_trials, n_channels, n_timepoints)
            EEG 样本集合
        eps : float, 可选
            防止奇异矩阵的正则项

    返回:
        X_aligned : np.ndarray, shape = (n_trials, n_channels, n_timepoints)
    """
    X = np.asarray(X, dtype=np.float64)
    n_trials, n_channels, _ = X.shape

    # 1. 计算每个试次的协方差矩阵 (C_i = X_i X_i^T)
    covs = np.einsum('nij,nkj->nik', X, X)

    # 2. 计算平均协方差矩阵 R
    R = covs.mean(axis=0)

    # 3. 求 R 的特征分解并构造 R^{-1/2}
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.clip(eigvals, eps, None)
    R_inv_sqrt = (eigvecs * eigvals**-0.5) @ eigvecs.T

    return R_inv_sqrt

def eeg_alignment(X: np.ndarray, R_inv_sqrt: np.ndarray) -> np.ndarray:

    # 4. 对单个样本执行 X_i' = R^{-1/2} @ X_i
    X_aligned = np.einsum('ij,jk->ik', R_inv_sqrt, X)

    return X_aligned

def load_seediv():
    if not os.path.exists('./cached_data/seediv_EA'):
        dataset = SEEDIVDataset(io_path=f'./cached_data/seediv',
                                root_path='/home/yhj/pythonProject/data/seediv/eeg_raw_data',
                                chunk_size=200*4,
                                offline_transform=transforms.Compose([
                                    transforms.Downsample(num_points=250*4, axis=1),
                                    transforms.Lambda(lambda x: band_filter(x, low_pass=1, high_pass=75, sampling_rate=250)),
                                    transforms.MeanStdNormalize(axis=1),
                                ]),
                                label_transform=transforms.Compose([
                                    transforms.Select('emotion'),
                                    transforms.Lambda(lambda x: x)
                                ]),
                                num_worker=16)
        data = [dataset[i][0] for i in range(len(dataset))]
        data = np.stack(data, axis=0)
        R_inv_sqrt = EA_R_inv_sqrt(data, eps=1e-10)
    dataset = SEEDIVDataset(io_path=f'./cached_data/seediv_EA',
                            root_path='/home/yhj/pythonProject/data/seediv/eeg_raw_data',
                            chunk_size=200*4,
                            offline_transform=transforms.Compose([
                                transforms.Downsample(num_points=250*4, axis=1),
                                transforms.Lambda(lambda x: band_filter(x, low_pass=1, high_pass=75, sampling_rate=250)),
                                transforms.MeanStdNormalize(axis=1),
                                transforms.Lambda(lambda x: eeg_alignment(x, R_inv_sqrt)),
                                transforms.MeanStdNormalize(axis=1),
                                transforms.To2d(),
                                transforms.ToTensor()
                            ]),
                            label_transform=transforms.Compose([
                                transforms.Select('emotion'),
                                transforms.Lambda(lambda x: x)
                            ]),
                            num_worker=16)
    # print(dataset[0])
    labels = [dataset[i][1] for i in range(len(dataset))]
    # print(set(labels))
    num_classes = len(set(labels))
    info = {
        'num_electrodes': dataset.num_channel,
        'chunk_size': 250*4,
        'num_classes': num_classes,
        'sampling_rate': 250,
    }
    return dataset, info


def load_m3cv():
    if not os.path.exists('./cached_data/m3cv_EA'):
        dataset = M3CVDataset(io_path=f'./cached_data/m3cv',
                            root_path='/home/yhj/pythonProject/data/m3cv',
                            chunk_size=250*4,
                            offline_transform=transforms.Compose([
                                transforms.Lambda(lambda x: band_filter(x, low_pass=1, high_pass=40, sampling_rate=250)),
                                transforms.MeanStdNormalize(axis=1),
                            ]),
                            label_transform=transforms.Compose([
                                transforms.Select('subject_id'),    # "sub001"
                                # transforms.Lambda(lambda x: sid2idx[x])
                                transforms.Lambda(lambda x: int(re.sub(r'^\D+', '', x)) - 1)
                            ]),
                            num_worker=16)
        data = [dataset[i][0] for i in range(len(dataset))]
        data = np.stack(data, axis=0)
        R_inv_sqrt = EA_R_inv_sqrt(data, eps=1e-10)
    dataset = M3CVDataset(io_path=f'./cached_data/m3cv_EA',
                        root_path='/home/yhj/pythonProject/data/m3cv',
                        chunk_size=250*4,
                        offline_transform=transforms.Compose([
                            transforms.Lambda(lambda x: band_filter(x, low_pass=1, high_pass=40, sampling_rate=250)),
                            transforms.MeanStdNormalize(axis=1),
                            transforms.Lambda(lambda x: eeg_alignment(x, R_inv_sqrt)),
                            transforms.MeanStdNormalize(axis=1),
                            transforms.To2d(),
                            transforms.ToTensor()
                        ]),
                        label_transform=transforms.Compose([
                            transforms.Select('subject_id'),    # "sub001"
                            # transforms.Lambda(lambda x: sid2idx[x])
                            transforms.Lambda(lambda x: int(re.sub(r'^\D+', '', x)) - 1)
                        ]),
                        num_worker=16)
    labels = [dataset[i][1] for i in range(len(dataset))]
    num_classes = len(set(labels))
    info = {
        'num_electrodes': dataset.num_channel,
        'chunk_size': 250*4,
        'num_classes': num_classes,
        'sampling_rate': 250,
    }
    return dataset, info


def load_bciciv2a():
    if not os.path.exists('./cached_data/bciciv2a_EA'):
        dataset = BCICIV2aDataset(io_path='./cached_data/bciciv2a',
                                root_path='/home/yhj/pythonProject/data/bciciv2a',
                                chunk_size=250*7,
                                offline_transform=transforms.Compose([
                                    transforms.Lambda(lambda x: band_filter(x, low_pass=1, high_pass=49, sampling_rate=250)),
                                    transforms.MeanStdNormalize(axis=1),
                                ]),
                                label_transform=transforms.Compose([
                                    transforms.Select('label'),
                                    transforms.Lambda(lambda x: x - 1)
                                ]),
                                num_worker=16)
        data = [dataset[i][0] for i in range(len(dataset))]
        data = np.stack(data, axis=0)
        R_inv_sqrt = EA_R_inv_sqrt(data, eps=1e-10)
    dataset = BCICIV2aDataset(io_path='./cached_data/bciciv2a_EA',
                            root_path='/home/yhj/pythonProject/data/bciciv2a',
                            chunk_size=250*7,
                            offline_transform=transforms.Compose([
                                transforms.Lambda(lambda x: band_filter(x, low_pass=1, high_pass=49, sampling_rate=250)),
                                transforms.MeanStdNormalize(axis=1),
                                transforms.Lambda(lambda x: eeg_alignment(x, R_inv_sqrt)),
                                transforms.MeanStdNormalize(axis=1),
                                transforms.To2d(),
                                transforms.ToTensor(),
                            ]),
                            label_transform=transforms.Compose([
                                transforms.Select('label'),
                                transforms.Lambda(lambda x: x - 1)
                            ]),
                            num_worker=16)
    # print(dataset[0], dataset[0][0].shape)
    labels = [dataset[i][1] for i in range(len(dataset))]
    # print(set(labels))
    num_classes = len(set(labels))
    info = {
        'num_electrodes': dataset.num_channel,
        'chunk_size': 250*7,
        'num_classes': num_classes,
        'sampling_rate': 250,
    }
    return dataset, info


def load_thubenchmark():
    if not os.path.exists('./cached_data/thubenchmark_EA'):
        dataset = TSUBenckmarkDataset(io_path='./cached_data/thubenchmark',
                                root_path='/home/yhj/pythonProject/data/THUBenchmark',
                                chunk_size=250*6,
                                offline_transform=transforms.Compose([
                                    transforms.Lambda(lambda x: band_filter(x, low_pass=7, high_pass=33, sampling_rate=250)),
                                    transforms.MeanStdNormalize(axis=1),
                                ]),
                                label_transform=transforms.Select('trial_id'),
                                num_worker=16)
        data = [dataset[i][0] for i in range(len(dataset))]
        data = np.stack(data, axis=0)
        R_inv_sqrt = EA_R_inv_sqrt(data, eps=1e-10)
    dataset = TSUBenckmarkDataset(io_path='./cached_data/thubenchmark_EA',
                            root_path='/home/yhj/pythonProject/data/THUBenchmark',
                            chunk_size=250*6,
                            offline_transform=transforms.Compose([
                                transforms.Lambda(lambda x: band_filter(x, low_pass=7, high_pass=33, sampling_rate=250)),
                                transforms.MeanStdNormalize(axis=1),
                                transforms.Lambda(lambda x: eeg_alignment(x, R_inv_sqrt)),
                                transforms.MeanStdNormalize(axis=1),
                                transforms.ToTensor(),
                                transforms.To2d(),
                            ]),
                            label_transform=transforms.Select('trial_id'),
                            num_worker=16)
    # dataset = TSUBenckmarkDataset(io_path='./cached_data/thubenchmark',
    #                         root_path='/home/yhj/pythonProject/data/THUBenchmark',
    #                         chunk_size=250*6,
    #                         offline_transform=transforms.Compose([
    #                             transforms.Lambda(lambda x: band_filter(x, low_pass=6, high_pass=18, sampling_rate=250)),
    #                             transforms.MeanStdNormalize(axis=1),
    #                             transforms.ToTensor(),
    #                             transforms.To2d(),
    #                         ]),
    #                         label_transform=transforms.Select('trial_id'),
    #                         num_worker=16)
    # print(dataset[0], dataset[0][0].shape)
    labels = [dataset[i][1] for i in range(len(dataset))]
    # print(set(labels))
    num_classes = len(set(labels))
    info = {
        'num_electrodes': dataset.num_channel,
        'chunk_size': 250*6,
        'num_classes': num_classes,
        'sampling_rate': 250,
    }
    return dataset, info


def band_filter(data, low_pass=1, high_pass=40, sampling_rate=250):
    raw = mne.io.RawArray(data, mne.create_info(['ch1']*data.shape[0], sampling_rate, ['eeg']*data.shape[0]))
    raw.notch_filter(freqs=[50], picks='eeg', method='iir', verbose=False)
    raw.filter(l_freq=low_pass, h_freq=high_pass, picks='eeg', method='iir', verbose=False)
    return raw.get_data()

if __name__ == '__main__':
    # load_seediv()
    dataset, info = load_m3cv()
    # dataset, info = load_bciciv2a() 
    # print(dataset[0], dataset[0][0].shape)
    # dataset = load_thubenchmark()
    # print(dataset[0], dataset[0][0].shape)

    