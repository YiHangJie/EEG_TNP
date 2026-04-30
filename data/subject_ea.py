import os
from copy import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from data.ea_utils import EA_R_inv_sqrt, eeg_alignment, finalize_eeg_sample


EA_PROTOCOL_TAG = 'train_only_subject_ea_subject_split'
RAW_PROTOCOL_TAG = 'train_only_subject_no_ea_subject_split'
DEFAULT_N_SPLITS = 5


def get_protocol_tag(use_ea: bool = True) -> str:
    return EA_PROTOCOL_TAG if use_ea else RAW_PROTOCOL_TAG


def get_default_n_splits(dataset_name: str) -> int:
    return DEFAULT_N_SPLITS


def get_split_path(dataset_name: str, seed: int) -> str:
    return f'./cached_data/{dataset_name}_{EA_PROTOCOL_TAG}_seed{seed}_split'


def _expected_split_files(split_path: str, n_splits: int):
    return [
        os.path.join(split_path, f'{split_name}_fold_{fold_id}.csv')
        for fold_id in range(n_splits)
        for split_name in ('train', 'val', 'test')
    ]


def _sorted_subject_items(info: pd.DataFrame):
    grouped = info.groupby('subject_id').indices
    items = list(grouped.items())
    items.sort(key=lambda item: str(item[0]))
    return items


def _validate_subject_sizes(items, n_splits: int):
    if n_splits < 2:
        raise ValueError(f'n_splits must be at least 2, but got {n_splits}.')

    min_group_size = 2 * n_splits
    for subject_id, indices in items:
        if len(indices) < min_group_size:
            raise ValueError(
                f'Subject {subject_id} has {len(indices)} samples, '
                f'which is fewer than the required minimum {min_group_size} for n_splits={n_splits}.'
            )


def _save_split_frame(info: pd.DataFrame, indices, path: str):
    split_frame = info.iloc[np.sort(np.asarray(indices, dtype=int))].copy().reset_index(drop=True)
    split_frame.to_csv(path, index=False)


def build_or_load_subject_splits(dataset_name: str,
                                 info: pd.DataFrame,
                                 n_splits: int | None = None,
                                 seed: int = 42,
                                 split_path: str | None = None) -> str:
    if n_splits is None:
        n_splits = get_default_n_splits(dataset_name)
    if split_path is None:
        split_path = get_split_path(dataset_name, seed)
    expected_files = _expected_split_files(split_path, n_splits)
    if all(os.path.exists(path) for path in expected_files):
        return split_path

    os.makedirs(split_path, exist_ok=True)
    info = info.reset_index(drop=True).copy()
    subject_items = _sorted_subject_items(info)
    _validate_subject_sizes(subject_items, n_splits)

    outer_subject_chunks = []
    for subject_index, (subject_id, indices) in enumerate(subject_items):
        indices = np.asarray(indices, dtype=int)
        rng = np.random.RandomState(seed + subject_index)
        shuffled_indices = rng.permutation(indices)
        outer_subject_chunks.append((subject_id, np.array_split(shuffled_indices, n_splits)))

    for fold_id in range(n_splits):
        train_indices = []
        val_indices = []
        test_indices = []

        for subject_index, (subject_id, fold_chunks) in enumerate(outer_subject_chunks):
            fold_holdout_indices = np.asarray(fold_chunks[fold_id], dtype=int)
            if len(fold_holdout_indices) < 2:
                raise ValueError(
                    f'Subject {subject_id} has only {len(fold_holdout_indices)} '
                    f'samples in fold {fold_id}, which cannot be split into validation and test subsets.'
                )

            subject_train_indices = [
                np.asarray(chunk, dtype=int)
                for chunk_index, chunk in enumerate(fold_chunks)
                if chunk_index != fold_id
            ]
            train_indices.extend(np.concatenate(subject_train_indices).tolist())

            subject_val_indices, subject_test_indices = train_test_split(
                fold_holdout_indices,
                test_size=0.5,
                shuffle=True,
                random_state=seed + fold_id * 1000 + subject_index
            )
            val_indices.extend(np.asarray(subject_val_indices, dtype=int).tolist())
            test_indices.extend(np.asarray(subject_test_indices, dtype=int).tolist())

        _save_split_frame(info, train_indices, os.path.join(split_path, f'train_fold_{fold_id}.csv'))
        _save_split_frame(info, val_indices, os.path.join(split_path, f'val_fold_{fold_id}.csv'))
        _save_split_frame(info, test_indices, os.path.join(split_path, f'test_fold_{fold_id}.csv'))

    return split_path


def build_or_load_grouped_splits(dataset_name: str,
                                 info: pd.DataFrame,
                                 group_key: str | None = None,
                                 n_splits: int | None = None,
                                 seed: int = 42,
                                 split_path: str | None = None) -> str:
    return build_or_load_subject_splits(
        dataset_name=dataset_name,
        info=info,
        n_splits=n_splits,
        seed=seed,
        split_path=split_path
    )


def _subset_dataset(dataset, split_info: pd.DataFrame):
    subset = copy(dataset)
    subset.info = split_info.reset_index(drop=True).copy()
    return subset


def _read_cached_eeg(dataset, index: int):
    info = dataset.read_info(index)
    eeg_index = str(info['clip_id'])
    eeg_record = str(info['_record_id'])
    return dataset.read_eeg(eeg_record, eeg_index)


def fit_subject_r_inv_sqrt(train_dataset, eps: float = 1e-10):
    subject_r_inv_sqrt = {}
    subject_groups = train_dataset.info.groupby('subject_id').indices

    for subject_id, indices in subject_groups.items():
        subject_samples = [_read_cached_eeg(train_dataset, int(index)) for index in indices]
        subject_r_inv_sqrt[subject_id] = EA_R_inv_sqrt(np.stack(subject_samples, axis=0), eps=eps)

    return subject_r_inv_sqrt


def _validate_subject_coverage(dataset, subject_r_inv_sqrt, split_name: str):
    missing_subjects = sorted(
        str(subject_id)
        for subject_id in set(dataset.info['subject_id'].tolist())
        if subject_id not in subject_r_inv_sqrt
    )
    if missing_subjects:
        raise ValueError(
            f'{split_name} split contains subjects without fitted EA matrices: {missing_subjects}.'
        )


class CachedEEGDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.info = dataset.info
        self.num_channel = getattr(dataset, 'num_channel', None)

    def __len__(self):
        return len(self.dataset)

    def read_info(self, index: int):
        return self.dataset.read_info(index)

    def _prepare_eeg(self, eeg: np.ndarray, info: dict) -> np.ndarray:
        return eeg

    def __getitem__(self, index: int):
        info = self.dataset.read_info(index)
        eeg = _read_cached_eeg(self.dataset, index)
        eeg = self._prepare_eeg(eeg, info)
        signal = finalize_eeg_sample(eeg)

        label = info
        if getattr(self.dataset, 'label_transform', None):
            label = self.dataset.label_transform(y=info)['y']

        return signal, label


class SubjectEAAlignedDataset(CachedEEGDataset):
    def __init__(self, dataset, subject_r_inv_sqrt):
        super().__init__(dataset)
        self.subject_r_inv_sqrt = subject_r_inv_sqrt

    def _prepare_eeg(self, eeg: np.ndarray, info: dict) -> np.ndarray:
        subject_id = info['subject_id']
        if subject_id not in self.subject_r_inv_sqrt:
            raise ValueError(f'Missing EA matrix for subject_id={subject_id}.')
        return eeg_alignment(eeg, self.subject_r_inv_sqrt[subject_id])


def prepare_subject_fold(dataset_name: str,
                         dataset,
                         info: dict,
                         fold_id: int,
                         seed: int = 42,
                         n_splits: int | None = None,
                         split_path: str | None = None,
                         use_ea: bool = True):
    if n_splits is None:
        n_splits = get_default_n_splits(dataset_name)
    split_path = build_or_load_subject_splits(
        dataset_name=dataset_name,
        info=dataset.info,
        n_splits=n_splits,
        seed=seed,
        split_path=split_path
    )

    train_info = pd.read_csv(os.path.join(split_path, f'train_fold_{fold_id}.csv'))
    val_info = pd.read_csv(os.path.join(split_path, f'val_fold_{fold_id}.csv'))
    test_info = pd.read_csv(os.path.join(split_path, f'test_fold_{fold_id}.csv'))

    train_dataset = _subset_dataset(dataset, train_info)
    val_dataset = _subset_dataset(dataset, val_info)
    test_dataset = _subset_dataset(dataset, test_info)

    if not use_ea:
        return (
            CachedEEGDataset(train_dataset),
            CachedEEGDataset(val_dataset),
            CachedEEGDataset(test_dataset),
            split_path,
        )

    subject_r_inv_sqrt = fit_subject_r_inv_sqrt(train_dataset)
    _validate_subject_coverage(val_dataset, subject_r_inv_sqrt, 'Validation')
    _validate_subject_coverage(test_dataset, subject_r_inv_sqrt, 'Test')

    return (
        SubjectEAAlignedDataset(train_dataset, subject_r_inv_sqrt),
        SubjectEAAlignedDataset(val_dataset, subject_r_inv_sqrt),
        SubjectEAAlignedDataset(test_dataset, subject_r_inv_sqrt),
        split_path,
    )


def prepare_subject_ea_fold(dataset_name: str,
                            dataset,
                            info: dict,
                            fold_id: int,
                            seed: int = 42,
                            n_splits: int | None = None,
                            split_path: str | None = None,
                            use_ea: bool = True):
    return prepare_subject_fold(
        dataset_name=dataset_name,
        dataset=dataset,
        info=info,
        fold_id=fold_id,
        seed=seed,
        n_splits=n_splits,
        split_path=split_path,
        use_ea=use_ea,
    )


def iter_subject_folds(dataset_name: str,
                       dataset,
                       info: dict,
                       seed: int = 42,
                       n_splits: int | None = None,
                       split_path: str | None = None,
                       use_ea: bool = True):
    if n_splits is None:
        n_splits = get_default_n_splits(dataset_name)
    for fold_id in range(n_splits):
        train_dataset, val_dataset, test_dataset, split_path = prepare_subject_fold(
            dataset_name=dataset_name,
            dataset=dataset,
            info=info,
            fold_id=fold_id,
            seed=seed,
            n_splits=n_splits,
            split_path=split_path,
            use_ea=use_ea,
        )
        yield fold_id, train_dataset, val_dataset, test_dataset, split_path


def iter_subject_ea_folds(dataset_name: str,
                          dataset,
                          info: dict,
                          seed: int = 42,
                          n_splits: int | None = None,
                          split_path: str | None = None,
                          use_ea: bool = True):
    yield from iter_subject_folds(
        dataset_name=dataset_name,
        dataset=dataset,
        info=info,
        seed=seed,
        n_splits=n_splits,
        split_path=split_path,
        use_ea=use_ea,
    )
