import argparse
import os
import re
import mne
from torcheeg.datasets import SEEDIVDataset, M3CVDataset, BCICIV2aDataset, TSUBenckmarkDataset
from torcheeg import transforms


def _collect_labels(dataset):
    labels = []
    for index in range(len(dataset)):
        info = dataset.read_info(index)
        if dataset.label_transform:
            labels.append(dataset.label_transform(y=info)['y'])
        else:
            labels.append(info)
    return labels


def _build_info(dataset, chunk_size):
    labels = _collect_labels(dataset)
    return {
        'num_electrodes': dataset.num_channel,
        'chunk_size': chunk_size,
        'num_classes': len(set(labels)),
        'sampling_rate': 250,
    }

def load_seediv():
    dataset = SEEDIVDataset(io_path='./cached_data/seediv',
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
    return dataset, _build_info(dataset, chunk_size=250*4)


def load_m3cv():
    dataset = M3CVDataset(io_path='./cached_data/m3cv',
                          root_path='/home/yhj/pythonProject/data/m3cv',
                          chunk_size=250*4,
                          offline_transform=transforms.Compose([
                              transforms.Lambda(lambda x: band_filter(x, low_pass=1, high_pass=40, sampling_rate=250)),
                              transforms.MeanStdNormalize(axis=1),
                          ]),
                          label_transform=transforms.Compose([
                              transforms.Select('subject_id'),
                              transforms.Lambda(lambda x: int(re.sub(r'^\D+', '', x)) - 1)
                          ]),
                          num_worker=16)
    return dataset, _build_info(dataset, chunk_size=250*4)


def load_bciciv2a():
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
    return dataset, _build_info(dataset, chunk_size=250*7)


def load_thubenchmark():
    dataset = TSUBenckmarkDataset(io_path='./cached_data/thubenchmark',
                                  root_path='/home/yhj/pythonProject/data/THUBenchmark',
                                  chunk_size=250*6,
                                  offline_transform=transforms.Compose([
                                      transforms.Lambda(lambda x: band_filter(x, low_pass=7, high_pass=33, sampling_rate=250)),
                                      transforms.MeanStdNormalize(axis=1),
                                  ]),
                                  label_transform=transforms.Select('trial_id'),
                                  num_worker=16)
    return dataset, _build_info(dataset, chunk_size=250*6)


def band_filter(data, low_pass=1, high_pass=40, sampling_rate=250):
    raw = mne.io.RawArray(data, mne.create_info(['ch1']*data.shape[0], sampling_rate, ['eeg']*data.shape[0]))
    raw.notch_filter(freqs=[50], picks='eeg', method='iir', verbose=False)
    raw.filter(l_freq=low_pass, h_freq=high_pass, picks='eeg', method='iir', verbose=False)
    return raw.get_data()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EEG dataset loading example')
    parser.add_argument('--dataset', type=str, default='thubenchmark',
                        choices=['seediv', 'm3cv', 'bciciv2a', 'thubenchmark'],
                        help='dataset to load')
    parser.add_argument('--index', type=int, default=0,
                        help='sample index to inspect')
    args = parser.parse_args()

    dataset_dict = {
        'seediv': load_seediv,
        'm3cv': load_m3cv,
        'bciciv2a': load_bciciv2a,
        'thubenchmark': load_thubenchmark
    }

    print(f'[Step 1] Start loading dataset: {args.dataset}')
    dataset, info = dataset_dict[args.dataset]()
    print('[Step 2] Dataset loaded successfully.')
    print(f'  dataset type: {type(dataset).__name__}')
    print(f'  dataset size: {len(dataset)}')
    print(f'  info: {info}')

    sample_index = max(0, min(args.index, len(dataset) - 1))
    print(f'[Step 3] Inspect sample at index {sample_index}')
    sample_data, sample_label = dataset[sample_index]
    sample_info = dataset.read_info(sample_index)
    print(f'  sample label: {sample_label}')
    print(f'  sample subject_id: {sample_info.get("subject_id")}')
    print(f'  sample data type: {type(sample_data)}')
    print(f'  sample data shape: {sample_data.shape}')

    if hasattr(sample_data, 'dtype'):
        print(f'  sample data dtype: {sample_data.dtype}')
    if hasattr(sample_data, 'min') and hasattr(sample_data, 'max'):
        print(f'  sample data min/max: {sample_data.min()} / {sample_data.max()}')

    print('[Step 4] Inspect a few labels')
    preview_count = min(5, len(dataset))
    preview_labels = [dataset[i][1] for i in range(preview_count)]
    print(f'  first {preview_count} labels: {preview_labels}')
