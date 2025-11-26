def get_model_args(name, dataset, info):
    if name == 'eegnet':
        args = {
            "chunk_size": info['chunk_size'],
            "num_electrodes": info['num_electrodes'],
            "num_classes": info['num_classes'],
        }
    elif name == 'tsception':
        if dataset == 'thubenchmark':
            args = {
                "sampling_rate": info['sampling_rate'],
                "num_electrodes": info['num_electrodes'],
                "num_classes": info['num_classes'],
                "num_T": 30,
                "num_S": 30,
                "hid_channels": 64,
            }
        else:
            args = {
                "sampling_rate": info['sampling_rate'],
                "num_electrodes": info['num_electrodes'],
                "num_classes": info['num_classes'],
            }
    elif name == 'atcnet':
        args = {
            "chunk_size": info['chunk_size'],
            "num_electrodes": info['num_electrodes'],
            "num_classes": info['num_classes'],
        }
    elif name == 'conformer':
        args = {
            "sampling_rate": info['chunk_size'],    # sampling_rate is used as chunk_size in conformer
            "num_electrodes": info['num_electrodes'],
            "num_classes": info['num_classes'],
        }
    return args