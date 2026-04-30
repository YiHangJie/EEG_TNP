import tempfile

import numpy as np
import pandas as pd
import torch

from data.ea_utils import EA_R_inv_sqrt, eeg_alignment, finalize_eeg_sample
from data.subject_ea import (
    CachedEEGDataset,
    SubjectEAAlignedDataset,
    build_or_load_subject_splits,
    build_or_load_grouped_splits,
    fit_subject_r_inv_sqrt,
    prepare_subject_fold,
    prepare_subject_ea_fold,
)


class FakeLabelTransform:
    def __call__(self, *, y):
        return {'y': y['label']}


class FakeDataset:
    def __init__(self, info: pd.DataFrame, eeg_store: dict):
        self.info = info.reset_index(drop=True).copy()
        self.eeg_store = eeg_store
        self.label_transform = FakeLabelTransform()
        self.num_channel = next(iter(eeg_store.values())).shape[0]

    def __len__(self):
        return len(self.info)

    def read_info(self, index: int):
        return self.info.iloc[index].to_dict()

    def read_eeg(self, record: str, key: str):
        return np.array(self.eeg_store[(record, key)], copy=True)


def _make_subject_info():
    rows = []
    for subject_id in ('s1', 's2'):
        for group_id in ('g1', 'g2'):
            for sample_idx in range(6):
                clip_id = f'{subject_id}_{group_id}_{sample_idx}'
                rows.append({
                    'subject_id': subject_id,
                    'trial_id': group_id,
                    'epoch_id': group_id,
                    'clip_id': clip_id,
                    '_record_id': 'record',
                    'label': 0 if subject_id == 's1' else 1,
                })
    return pd.DataFrame(rows)


def _make_subject_dataset():
    info = _make_subject_info()
    eeg_store = {}
    for row_index, row in info.iterrows():
        base = row_index + 1
        eeg_store[(row['_record_id'], row['clip_id'])] = np.array(
            [[base, base + 1, base + 2], [base + 3, base + 4, base + 5]],
            dtype=np.float32
        )
    return FakeDataset(info, eeg_store)


def test_build_or_load_subject_splits_partition_each_subject():
    info = _make_subject_info()

    with tempfile.TemporaryDirectory() as tmpdir:
        split_path = build_or_load_subject_splits(
            dataset_name='dummy',
            info=info,
            n_splits=3,
            seed=7,
            split_path=tmpdir
        )

        train_fold = pd.read_csv(f'{split_path}/train_fold_0.csv')
        val_fold = pd.read_csv(f'{split_path}/val_fold_0.csv')
        test_fold = pd.read_csv(f'{split_path}/test_fold_0.csv')

        combined = set(train_fold['clip_id']) | set(val_fold['clip_id']) | set(test_fold['clip_id'])
        assert combined == set(info['clip_id'])
        assert set(train_fold['clip_id']).isdisjoint(set(val_fold['clip_id']))
        assert set(train_fold['clip_id']).isdisjoint(set(test_fold['clip_id']))
        assert set(val_fold['clip_id']).isdisjoint(set(test_fold['clip_id']))

        for subject_id, subject_frame in info.groupby('subject_id'):
            subject_clips = set(subject_frame['clip_id'])
            assert len(subject_clips & set(train_fold['clip_id'])) == 8
            assert len(subject_clips & set(val_fold['clip_id'])) == 2
            assert len(subject_clips & set(test_fold['clip_id'])) == 2


def test_build_or_load_grouped_splits_alias_ignores_extra_group_columns():
    info = _make_subject_info()

    with tempfile.TemporaryDirectory() as tmpdir:
        split_path = build_or_load_grouped_splits(
            dataset_name='dummy',
            info=info,
            group_key='epoch_id',
            n_splits=3,
            seed=11,
            split_path=tmpdir
        )

        train_fold = pd.read_csv(f'{split_path}/train_fold_1.csv')
        val_fold = pd.read_csv(f'{split_path}/val_fold_1.csv')
        test_fold = pd.read_csv(f'{split_path}/test_fold_1.csv')
        combined = set(train_fold['clip_id']) | set(val_fold['clip_id']) | set(test_fold['clip_id'])
        assert combined == set(info['clip_id'])
        for subject_id, subject_frame in info.groupby('subject_id'):
            subject_clips = set(subject_frame['clip_id'])
            assert len(subject_clips & set(train_fold['clip_id'])) == 8
            assert len(subject_clips & set(val_fold['clip_id'])) == 2
            assert len(subject_clips & set(test_fold['clip_id'])) == 2


def test_build_or_load_subject_splits_rejects_small_subjects():
    info = pd.DataFrame([
        {'subject_id': 's1', 'trial_id': 't1', 'clip_id': f'c{index}', '_record_id': 'record', 'label': 0}
        for index in range(5)
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            build_or_load_subject_splits(
                dataset_name='dummy',
                info=info,
                n_splits=3,
                seed=3,
                split_path=tmpdir
            )
        except ValueError as exc:
            assert 'required minimum 6' in str(exc)
        else:
            raise AssertionError('Expected ValueError for undersized groups.')


def test_subject_ea_alignment_uses_train_only_subject_statistics():
    eeg_a1 = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    eeg_a2 = np.array([[2., 3., 4.], [5., 6., 7.]], dtype=np.float32)
    eeg_b1 = np.array([[10., 11., 12.], [13., 14., 15.]], dtype=np.float32)
    eeg_b2 = np.array([[11., 12., 13.], [14., 15., 16.]], dtype=np.float32)
    eeg_val = np.array([[3., 4., 5.], [6., 7., 8.]], dtype=np.float32)

    eeg_store = {
        ('record', 'a1'): eeg_a1,
        ('record', 'a2'): eeg_a2,
        ('record', 'b1'): eeg_b1,
        ('record', 'b2'): eeg_b2,
        ('record', 'v1'): eeg_val,
    }
    train_info = pd.DataFrame([
        {'subject_id': 's1', 'trial_id': 't1', 'clip_id': 'a1', '_record_id': 'record', 'label': 0},
        {'subject_id': 's1', 'trial_id': 't1', 'clip_id': 'a2', '_record_id': 'record', 'label': 0},
        {'subject_id': 's2', 'trial_id': 't2', 'clip_id': 'b1', '_record_id': 'record', 'label': 1},
        {'subject_id': 's2', 'trial_id': 't2', 'clip_id': 'b2', '_record_id': 'record', 'label': 1},
    ])
    val_info = pd.DataFrame([
        {'subject_id': 's1', 'trial_id': 't1', 'clip_id': 'v1', '_record_id': 'record', 'label': 0},
    ])

    train_dataset = FakeDataset(train_info, eeg_store)
    val_dataset = FakeDataset(val_info, eeg_store)
    subject_r_inv_sqrt = fit_subject_r_inv_sqrt(train_dataset)

    expected_r_inv_sqrt = EA_R_inv_sqrt(np.stack([eeg_a1, eeg_a2], axis=0))
    np.testing.assert_allclose(subject_r_inv_sqrt['s1'], expected_r_inv_sqrt, atol=1e-6)

    aligned_dataset = SubjectEAAlignedDataset(val_dataset, subject_r_inv_sqrt)
    signal, label = aligned_dataset[0]

    expected_signal = finalize_eeg_sample(eeg_alignment(eeg_val, expected_r_inv_sqrt))
    assert torch.allclose(signal, expected_signal)
    assert label == 0
    assert signal.shape == (1, 2, 3)


def test_cached_eeg_dataset_returns_unaligned_cached_signal():
    dataset = _make_subject_dataset()
    raw_dataset = CachedEEGDataset(dataset)

    signal, label = raw_dataset[0]
    info = raw_dataset.read_info(0)
    expected_signal = finalize_eeg_sample(dataset.read_eeg(info['_record_id'], info['clip_id']))

    assert torch.allclose(signal, expected_signal)
    assert label in (0, 1)
    assert signal.shape == (1, 2, 3)


def test_prepare_subject_ea_fold_smoke():
    dataset = _make_subject_dataset()
    info = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        train_dataset, val_dataset, test_dataset, _ = prepare_subject_ea_fold(
            dataset_name='dummy',
            dataset=dataset,
            info=info,
            fold_id=0,
            seed=5,
            n_splits=3,
            split_path=tmpdir
        )

        assert len(train_dataset) + len(val_dataset) + len(test_dataset) == len(dataset)
        train_signal, train_label = train_dataset[0]
        assert isinstance(train_signal, torch.Tensor)
        assert train_signal.ndim == 3
        assert train_signal.shape[0] == 1
        assert train_label in (0, 1)


def test_prepare_subject_fold_without_ea_returns_raw_dataset():
    dataset = _make_subject_dataset()
    info = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        train_dataset, val_dataset, test_dataset, _ = prepare_subject_fold(
            dataset_name='dummy',
            dataset=dataset,
            info=info,
            fold_id=0,
            seed=5,
            n_splits=3,
            split_path=tmpdir,
            use_ea=False
        )

        assert len(train_dataset) + len(val_dataset) + len(test_dataset) == len(dataset)
        assert isinstance(train_dataset, CachedEEGDataset)
        train_signal, train_label = train_dataset[0]
        assert isinstance(train_signal, torch.Tensor)
        assert train_signal.ndim == 3
        assert train_signal.shape[0] == 1
        assert train_label in (0, 1)
