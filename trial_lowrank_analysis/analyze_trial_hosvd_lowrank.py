#!/usr/bin/env python3
"""从 trial/time/channel/frequency 多视角分析 EEG 低秩结构。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = SCRIPT_DIR / ".cache"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MATPLOTLIB_CACHE_DIR = LOCAL_CACHE_DIR / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_DIR))

from runtime_env import configure_runtime_env

configure_runtime_env()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torcheeg.datasets.constants import (
    M3CV_CHANNEL_LOCATION_DICT,
    SEED_IV_CHANNEL_LOCATION_DICT,
)
from torcheeg.datasets.constants.motor_imagery import BCICIV2A_LOCATION_DICT
from torcheeg.datasets.constants.ssvep import TSUBENCHMARK_CHANNEL_LOCATION_DICT
from torcheeg.models import ATCNet, Conformer, EEGNet, TSCeption
from torcheeg.transforms import ToGrid, ToInterpolatedGrid

from attack.pgd import PGD
from data.load import load_bciciv2a, load_m3cv, load_seediv, load_thubenchmark
from data.subject_ea import get_protocol_tag, prepare_subject_fold
from models.model_args import get_model_args


DATASET_LOADERS = {
    "seediv": load_seediv,
    "m3cv": load_m3cv,
    "bciciv2a": load_bciciv2a,
    "thubenchmark": load_thubenchmark,
}

MODEL_CLASSES = {
    "eegnet": EEGNet,
    "tsception": TSCeption,
    "atcnet": ATCNet,
    "conformer": Conformer,
}

DATA_TYPE_ORDER = ("clean", "adv", "perturb")
DEFAULT_VIEWS = ("trial", "time", "channel", "frequency")
PAIR_VIEWS = ("time_frequency", "time_space", "frequency_space")
VIEW_ORDER = DEFAULT_VIEWS + PAIR_VIEWS
CENTER_MODES = ("raw", "trial_centered")
PLOT_COLORS = {
    "clean": "tab:blue",
    "adv": "tab:orange",
    "perturb": "tab:green",
}
DATASET_CHANNEL_LOCATION_DICTS = {
    "seediv": SEED_IV_CHANNEL_LOCATION_DICT,
    "m3cv": M3CV_CHANNEL_LOCATION_DICT,
    "bciciv2a": BCICIV2A_LOCATION_DICT,
    "thubenchmark": TSUBENCHMARK_CHANNEL_LOCATION_DICT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "生成 PGD 对抗 EEG 样本，并从单轴或组合视角比较 "
            "clean/adv/perturbation 的低秩谱。"
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="thubenchmark",
        choices=tuple(DATASET_LOADERS.keys()),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="eegnet",
        choices=tuple(MODEL_CLASSES.keys()),
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--pgd_steps", type=int, default=200)
    parser.add_argument("--pgd_alpha", type=float, default=2 / 255)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--min_trials", type=int, default=3)
    parser.add_argument(
        "--sample_num",
        type=int,
        default=512,
        help="从 test split 中随机抽取的样本数；小于等于 0 时使用全部样本。",
    )
    parser.add_argument(
        "--views",
        type=str,
        nargs="+",
        default=list(DEFAULT_VIEWS),
        choices=VIEW_ORDER,
        help="需要分析的低秩视角。",
    )
    parser.add_argument(
        "--tf_n_fft",
        type=int,
        default=128,
        help="联合 view 统一构造 N x space x time x frequency 张量时 STFT 的 n_fft；小于等于 0 时自动按时间长度选择。",
    )
    parser.add_argument(
        "--tf_hop_length",
        type=int,
        default=64,
        help="联合 view 统一构造 N x space x time x frequency 张量时 STFT 的 hop_length；小于等于 0 时使用 n_fft // 2。",
    )
    parser.add_argument(
        "--frequency_representation",
        type=str,
        default="complex",
        choices=("complex", "magnitude", "power", "real_imag"),
        help=(
            "FFT 频率视角使用的表示。complex 保留相位，magnitude/power "
            "只保留幅值或功率，real_imag 将实部和虚部作为额外通道特征。"
        ),
    )
    parser.add_argument(
        "--channel_view_space",
        type=str,
        default="interpolated_grid",
        choices=("raw", "grid", "interpolated_grid"),
        help=(
            "channel 视角使用的空间表示。raw 使用原始电极向量；"
            "grid/interpolated_grid 将通道投影到 2D 头皮网格，"
            "再分析得到的 3D 空间-时间张量。"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to trial_lowrank_analysis/outputs/<run-id>.",
    )
    parser.add_argument(
        "--use_ea",
        dest="use_ea",
        action="store_true",
        default=False,
        help="Use train-only subject EA alignment.",
    )
    parser.add_argument(
        "--no_ea",
        dest="use_ea",
        action="store_false",
        help="Use raw subject-split data without EA alignment.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def format_float_for_path(value: float) -> str:
    return f"{value:g}"


def default_output_dir(args: argparse.Namespace, protocol_tag: str) -> Path:
    eps = format_float_for_path(args.eps)
    view_tag = "-".join(args.views)
    run_id = (
        f"{args.dataset}_{args.model}_{protocol_tag}_pgd_"
        f"eps{eps}_seed{args.seed}_fold{args.fold}_views{view_tag}_"
        f"freq{args.frequency_representation}_space{args.channel_view_space}_"
        f"tf{args.tf_n_fft}_hop{args.tf_hop_length}_n{args.sample_num}"
    )
    return SCRIPT_DIR / "outputs" / run_id


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "analysis.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{rem:04.1f}s"
    hours, rem_minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(rem_minutes):02d}m{rem:04.1f}s"


def load_clean_model(
    args: argparse.Namespace,
    info: dict,
    protocol_tag: str,
    device: torch.device,
) -> torch.nn.Module:
    model = MODEL_CLASSES[args.model](**get_model_args(args.model, args.dataset, info))
    model.to(device)
    model.eval()

    checkpoint_path = (
        REPO_ROOT
        / "checkpoints"
        / (
            f"{args.dataset}_{args.model}_{protocol_tag}_clean_eps0_"
            f"{args.seed}_fold{args.fold}_best.pth"
        )
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing clean checkpoint: {checkpoint_path}. "
            "Run with matching --use_ea/--no_ea or train the clean model first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    logging.info("Loaded clean checkpoint: %s", checkpoint_path)
    return model


class SampledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.info = dataset.info.iloc[self.indices].copy().reset_index(drop=True)
        if "original_split_index" not in self.info.columns:
            self.info.insert(0, "original_split_index", self.indices)
        self.num_channel = getattr(dataset, "num_channel", None)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int):
        return self.dataset[int(self.indices[index])]

    def read_info(self, index: int):
        if hasattr(self.dataset, "read_info"):
            return self.dataset.read_info(int(self.indices[index]))
        return self.info.iloc[int(index)].to_dict()


def sample_dataset(dataset, sample_num: int, seed: int):
    if sample_num <= 0 or sample_num >= len(dataset):
        return dataset, np.arange(len(dataset), dtype=np.int64)

    rng = np.random.default_rng(seed)
    # 采样后排序只影响遍历顺序，不影响随机样本集合，便于日志和 metadata 对齐检查。
    indices = np.sort(rng.choice(len(dataset), size=sample_num, replace=False))
    return SampledDataset(dataset, indices), indices


def views_need_spatial_projection(views: tuple[str, ...]) -> bool:
    return any(
        view in {"channel", "time_frequency", "time_space", "frequency_space"}
        for view in views
    )


def collect_clean_adv_and_metadata(
    args: argparse.Namespace,
    test_dataset,
    info: dict,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    collect_start = time.perf_counter()
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    attack = PGD(
        model,
        eps=args.eps,
        alpha=args.pgd_alpha,
        steps=args.pgd_steps,
        device=device,
        n_classes=info["num_classes"],
    )

    clean_batches = []
    adv_batches = []
    label_batches = []
    metadata_frames = []
    cursor = 0

    for batch_index, (data, target) in enumerate(loader):
        batch_start = time.perf_counter()
        batch_size = data.size(0)
        batch_meta = test_dataset.info.iloc[cursor : cursor + batch_size].copy()
        batch_meta.insert(0, "split_index", np.arange(cursor, cursor + batch_size))
        metadata_frames.append(batch_meta)

        data = data.to(device)
        target = target.to(device)
        adv_data = attack(data, target)

        clean_batches.append(data.detach().cpu())
        adv_batches.append(adv_data.detach().cpu())
        label_batches.append(target.detach().cpu())

        cursor += batch_size
        batch_elapsed = time.perf_counter() - batch_start
        total_elapsed = time.perf_counter() - collect_start
        logging.info(
            "Generated PGD batch %d/%d, batch_size=%d, batch_time=%s, total_time=%s",
            batch_index + 1,
            len(loader),
            batch_size,
            format_elapsed(batch_elapsed),
            format_elapsed(total_elapsed),
        )

    clean = torch.cat(clean_batches, dim=0).contiguous()
    adv = torch.cat(adv_batches, dim=0).contiguous()
    labels = torch.cat(label_batches, dim=0).contiguous()
    metadata = pd.concat(metadata_frames, axis=0).reset_index(drop=True)

    if len(metadata) != clean.shape[0] or clean.shape != adv.shape:
        raise RuntimeError(
            "Collected data and metadata are misaligned: "
            f"metadata={len(metadata)}, clean={tuple(clean.shape)}, adv={tuple(adv.shape)}"
        )

    logging.info(
        "PGD collection finished: samples=%d, clean_shape=%s, adv_shape=%s, elapsed=%s",
        clean.shape[0],
        tuple(clean.shape),
        tuple(adv.shape),
        format_elapsed(time.perf_counter() - collect_start),
    )
    return clean, adv, labels, metadata


def to_trial_channel_time(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 4 and tensor.size(1) == 1:
        tensor = tensor[:, 0, :, :]
    elif tensor.dim() != 3:
        raise ValueError(
            "Expected EEG data with shape (trial, 1, channel, time) "
            f"or (trial, channel, time), got {tuple(tensor.shape)}"
        )
    return tensor.float().cpu().contiguous()


def build_channel_spatial_transform(dataset: str, channel_view_space: str):
    if channel_view_space == "raw":
        return None

    channel_location_dict = DATASET_CHANNEL_LOCATION_DICTS.get(dataset)
    if channel_location_dict is None:
        raise KeyError(f"Missing channel location dict for dataset={dataset}")

    if channel_view_space == "grid":
        return ToGrid(channel_location_dict)
    if channel_view_space == "interpolated_grid":
        return ToInterpolatedGrid(channel_location_dict)
    raise ValueError(f"Unknown channel_view_space: {channel_view_space}")


def grid_eeg_to_space_time(grid_eeg: np.ndarray, expected_time: int) -> torch.Tensor:
    grid_tensor = torch.as_tensor(grid_eeg).float()
    if grid_tensor.dim() != 3:
        raise ValueError(
            "Expected projected EEG grid with 3 dimensions, "
            f"got shape={tuple(grid_tensor.shape)}"
        )

    # torcheeg 的空间投影通常输出 (time, x, y)，这里统一成 (x, y, time)。
    if grid_tensor.shape[0] == expected_time:
        return grid_tensor.permute(1, 2, 0).contiguous()
    if grid_tensor.shape[-1] == expected_time:
        return grid_tensor.contiguous()
    if grid_tensor.shape[1] == expected_time:
        return grid_tensor.permute(0, 2, 1).contiguous()
    raise ValueError(
        "Cannot infer time axis from projected EEG grid shape="
        f"{tuple(grid_tensor.shape)} and expected_time={expected_time}"
    )


def to_trial_space_time(
    tensor: torch.Tensor,
    spatial_transform,
    channel_view_space: str,
    log_context: str = "",
    progress_interval: int = 64,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    if channel_view_space == "raw":
        return tensor, (int(tensor.shape[1]),)

    projection_start = time.perf_counter()
    trials = []
    expected_time = int(tensor.shape[-1])
    logging.info(
        "%sChannel spatial projection start: space=%s, n_trials=%d, "
        "n_channels=%d, n_times=%d",
        log_context,
        channel_view_space,
        tensor.shape[0],
        tensor.shape[1],
        tensor.shape[2],
    )
    for trial_tensor in tensor:
        projected = spatial_transform(eeg=trial_tensor.detach().cpu().numpy())["eeg"]
        space_time = grid_eeg_to_space_time(projected, expected_time=expected_time)
        trials.append(space_time.reshape(-1, expected_time))
        if len(trials) % progress_interval == 0 or len(trials) == tensor.shape[0]:
            logging.info(
                "%sChannel spatial projection progress: %d/%d, elapsed=%s",
                log_context,
                len(trials),
                tensor.shape[0],
                format_elapsed(time.perf_counter() - projection_start),
            )

    spatial_tensor = torch.stack(trials, dim=0).contiguous()
    spatial_shape = tuple(int(size) for size in space_time.shape[:-1])
    logging.info(
        "%sChannel spatial projection finished: spatial_shape=%s, "
        "spatial_points=%d, elapsed=%s",
        log_context,
        "x".join(str(size) for size in spatial_shape),
        spatial_tensor.shape[1],
        format_elapsed(time.perf_counter() - projection_start),
    )
    return spatial_tensor, spatial_shape


def to_frequency_tensor(tensor: torch.Tensor, representation: str) -> torch.Tensor:
    spectrum = torch.fft.rfft(tensor.float(), dim=-1, norm="ortho")
    return apply_frequency_representation(spectrum, representation)


def apply_frequency_representation(
    spectrum: torch.Tensor,
    representation: str,
) -> torch.Tensor:
    if representation == "complex":
        return spectrum
    if representation == "magnitude":
        return spectrum.abs()
    if representation == "power":
        return spectrum.abs().square()
    if representation == "real_imag":
        n_trials, n_channels, n_freqs = spectrum.shape
        return (
            torch.view_as_real(spectrum)
            .permute(0, 1, 3, 2)
            .contiguous()
            .reshape(n_trials, n_channels * 2, n_freqs)
        )
    raise ValueError(f"Unknown frequency_representation: {representation}")


def resolve_stft_params(n_times: int, tf_n_fft: int, tf_hop_length: int) -> tuple[int, int]:
    if tf_n_fft <= 0:
        n_fft = min(128, 2 ** int(np.floor(np.log2(max(n_times, 2)))))
    else:
        n_fft = min(tf_n_fft, n_times)
    n_fft = max(2, int(n_fft))

    if tf_hop_length <= 0:
        hop_length = max(1, n_fft // 2)
    else:
        hop_length = int(tf_hop_length)
    hop_length = max(1, min(hop_length, n_fft))
    return n_fft, hop_length


def to_time_frequency_tensor(
    tensor: torch.Tensor,
    representation: str,
    tf_n_fft: int,
    tf_hop_length: int,
) -> tuple[torch.Tensor, dict]:
    n_trials, n_features, n_times = tensor.shape
    n_fft, hop_length = resolve_stft_params(
        n_times=n_times,
        tf_n_fft=tf_n_fft,
        tf_hop_length=tf_hop_length,
    )
    window = torch.hann_window(n_fft, dtype=torch.float32, device=tensor.device)
    flat_tensor = tensor.float().reshape(n_trials * n_features, n_times)
    spectrum = torch.stft(
        flat_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
        onesided=True,
        normalized=True,
    )
    n_freqs, n_tf_times = spectrum.shape[-2:]
    spectrum = spectrum.reshape(n_trials, n_features, n_freqs, n_tf_times)

    if representation == "complex":
        tf_tensor = spectrum.permute(0, 1, 3, 2).contiguous()
    elif representation == "magnitude":
        tf_tensor = spectrum.abs().permute(0, 1, 3, 2).contiguous()
    elif representation == "power":
        tf_tensor = spectrum.abs().square().permute(0, 1, 3, 2).contiguous()
    elif representation == "real_imag":
        # 将实部/虚部并入 space 特征轴，联合 view 仍保持 (trial, space, time, frequency)。
        tf_tensor = (
            torch.view_as_real(spectrum)
            .permute(0, 1, 4, 3, 2)
            .contiguous()
            .reshape(n_trials, n_features * 2, n_tf_times, n_freqs)
        )
    else:
        raise ValueError(f"Unknown frequency_representation: {representation}")

    return tf_tensor, {
        "n_frequencies": int(n_freqs),
        "n_tf_times": int(n_tf_times),
        "tf_n_fft": int(n_fft),
        "tf_hop_length": int(hop_length),
    }


def get_joint_time_frequency_space_tensor(
    tensor: torch.Tensor,
    representation: str,
    spatial_transform,
    channel_view_space: str,
    tf_n_fft: int,
    tf_hop_length: int,
    log_context: str,
    cache: dict | None,
) -> tuple[torch.Tensor, dict]:
    if cache is not None and "joint_time_frequency_space" in cache:
        logging.info("%sReuse cached joint time-frequency-space tensor", log_context)
        return cache["joint_time_frequency_space"]

    spatial_tensor, spatial_shape = get_trial_space_time(
        tensor=tensor,
        spatial_transform=spatial_transform,
        channel_view_space=channel_view_space,
        log_context=log_context,
        cache=cache,
    )
    stft_start = time.perf_counter()
    logging.info(
        "%sJoint STFT start: representation=%s, spatial_tensor_shape=%s, "
        "tf_n_fft=%d, tf_hop_length=%d",
        log_context,
        representation,
        tuple(spatial_tensor.shape),
        tf_n_fft,
        tf_hop_length,
    )
    joint_tensor, tf_info = to_time_frequency_tensor(
        tensor=spatial_tensor,
        representation=representation,
        tf_n_fft=tf_n_fft,
        tf_hop_length=tf_hop_length,
    )
    info = {
        **tf_info,
        "frequency_representation": representation,
        "spatial_shape": "x".join(str(size) for size in spatial_shape),
        "spatial_points": int(joint_tensor.shape[1]),
        "channel_view_space": channel_view_space,
    }
    logging.info(
        "%sJoint STFT finished: joint_shape=%s, n_fft=%d, hop_length=%d, elapsed=%s",
        log_context,
        tuple(joint_tensor.shape),
        info["tf_n_fft"],
        info["tf_hop_length"],
        format_elapsed(time.perf_counter() - stft_start),
    )

    value = (joint_tensor, info)
    if cache is not None:
        cache["joint_time_frequency_space"] = value
    return value


def get_trial_space_time(
    tensor: torch.Tensor,
    spatial_transform,
    channel_view_space: str,
    log_context: str,
    cache: dict | None,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    if cache is not None and "trial_space_time" in cache:
        logging.info("%sReuse cached spatial projection", log_context)
        return cache["trial_space_time"]

    value = to_trial_space_time(
        tensor=tensor,
        spatial_transform=spatial_transform,
        channel_view_space=channel_view_space,
        log_context=log_context,
    )
    if cache is not None:
        cache["trial_space_time"] = value
    return value


def view_unfold(
    tensor: torch.Tensor,
    view: str,
    frequency_representation: str,
    spatial_transform,
    channel_view_space: str,
    tf_n_fft: int,
    tf_hop_length: int,
    cache: dict | None = None,
    log_context: str = "",
) -> tuple[torch.Tensor, dict]:
    if view == "trial":
        matrix = tensor.reshape(tensor.shape[0], tensor.shape[1] * tensor.shape[2])
        return matrix, {"n_frequencies": 0, "spatial_shape": ""}

    if view == "time":
        matrix = tensor.permute(2, 0, 1).contiguous().reshape(
            tensor.shape[2], tensor.shape[0] * tensor.shape[1]
        )
        return matrix, {"n_frequencies": 0, "spatial_shape": ""}

    if view == "channel":
        spatial_tensor, spatial_shape = get_trial_space_time(
            tensor=tensor,
            spatial_transform=spatial_transform,
            channel_view_space=channel_view_space,
            log_context=log_context,
            cache=cache,
        )
        matrix = spatial_tensor.permute(1, 0, 2).contiguous().reshape(
            spatial_tensor.shape[1], spatial_tensor.shape[0] * spatial_tensor.shape[2]
        )
        return matrix, {
            "n_frequencies": 0,
            "spatial_shape": "x".join(str(size) for size in spatial_shape),
            "spatial_points": int(spatial_tensor.shape[1]),
        }

    if view == "time_space":
        joint_tensor, joint_info = get_joint_time_frequency_space_tensor(
            tensor=tensor,
            representation=frequency_representation,
            spatial_transform=spatial_transform,
            channel_view_space=channel_view_space,
            tf_n_fft=tf_n_fft,
            tf_hop_length=tf_hop_length,
            log_context=log_context,
            cache=cache,
        )
        matrix = joint_tensor.permute(1, 2, 0, 3).contiguous().reshape(
            joint_tensor.shape[1] * joint_tensor.shape[2],
            joint_tensor.shape[0] * joint_tensor.shape[3],
        )
        return matrix, {
            **joint_info,
            "pair_axes": "time_space",
            "joint_tensor_shape": "x".join(str(size) for size in joint_tensor.shape),
        }

    if view == "frequency":
        fft_start = time.perf_counter()
        logging.info(
            "%sFrequency transform start: representation=%s, input_shape=%s",
            log_context,
            frequency_representation,
            tuple(tensor.shape),
        )
        frequency_tensor = to_frequency_tensor(
            tensor,
            representation=frequency_representation,
        )
        logging.info(
            "%sFrequency transform finished: frequency_shape=%s, elapsed=%s",
            log_context,
            tuple(frequency_tensor.shape),
            format_elapsed(time.perf_counter() - fft_start),
        )
        matrix = frequency_tensor.permute(2, 0, 1).contiguous().reshape(
            frequency_tensor.shape[2],
            frequency_tensor.shape[0] * frequency_tensor.shape[1],
        )
        return matrix, {
            "n_frequencies": int(frequency_tensor.shape[2]),
            "frequency_representation": frequency_representation,
            "spatial_shape": "",
        }

    if view == "time_frequency":
        joint_tensor, joint_info = get_joint_time_frequency_space_tensor(
            tensor=tensor,
            representation=frequency_representation,
            spatial_transform=spatial_transform,
            channel_view_space=channel_view_space,
            tf_n_fft=tf_n_fft,
            tf_hop_length=tf_hop_length,
            log_context=log_context,
            cache=cache,
        )
        matrix = joint_tensor.permute(2, 3, 0, 1).contiguous().reshape(
            joint_tensor.shape[2] * joint_tensor.shape[3],
            joint_tensor.shape[0] * joint_tensor.shape[1],
        )
        return matrix, {
            **joint_info,
            "pair_axes": "time_frequency",
            "joint_tensor_shape": "x".join(str(size) for size in joint_tensor.shape),
        }

    if view == "frequency_space":
        joint_tensor, joint_info = get_joint_time_frequency_space_tensor(
            tensor=tensor,
            representation=frequency_representation,
            spatial_transform=spatial_transform,
            channel_view_space=channel_view_space,
            tf_n_fft=tf_n_fft,
            tf_hop_length=tf_hop_length,
            log_context=log_context,
            cache=cache,
        )
        matrix = joint_tensor.permute(1, 3, 0, 2).contiguous().reshape(
            joint_tensor.shape[1] * joint_tensor.shape[3],
            joint_tensor.shape[0] * joint_tensor.shape[2],
        )
        return matrix, {
            **joint_info,
            "pair_axes": "frequency_space",
            "joint_tensor_shape": "x".join(str(size) for size in joint_tensor.shape),
        }

    raise ValueError(f"Unknown low-rank view: {view}")


def singular_value_energy(matrix: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.numel() == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty

    with torch.no_grad():
        if matrix.is_complex():
            matrix_for_svd = matrix.to(torch.complex64)
        else:
            matrix_for_svd = matrix.float()
        singular_values = torch.linalg.svdvals(matrix_for_svd).detach().cpu().numpy()

    energy_raw = singular_values.astype(np.float64) ** 2
    total_energy = float(energy_raw.sum())
    if total_energy <= 0:
        energy = np.zeros_like(energy_raw, dtype=np.float64)
    else:
        energy = energy_raw / total_energy
    cumulative_energy = np.cumsum(energy)
    return singular_values, energy, cumulative_energy


def rank_at_energy(cumulative_energy: np.ndarray, threshold: float) -> int:
    if cumulative_energy.size == 0 or cumulative_energy[-1] <= 0:
        return 0
    return int(np.searchsorted(cumulative_energy, threshold, side="left") + 1)


def effective_rank(energy: np.ndarray) -> float:
    positive_energy = energy[energy > 0]
    if positive_energy.size == 0:
        return 0.0
    entropy = -float(np.sum(positive_energy * np.log(positive_energy)))
    return float(np.exp(entropy))


def topk_energy(energy: np.ndarray, k: int) -> float:
    if energy.size == 0:
        return 0.0
    return float(np.sum(energy[: min(k, energy.size)]))


def safe_key(value) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_") or "empty"


def analyze_tensor_set(
    data_type: str,
    tensor: torch.Tensor,
    metadata: pd.DataFrame,
    center_mode: str,
    min_trials: int,
    views: tuple[str, ...],
    frequency_representation: str,
    spatial_transform,
    channel_view_space: str,
    tf_n_fft: int,
    tf_hop_length: int,
) -> tuple[list[dict], list[dict]]:
    if "subject_id" not in metadata.columns:
        raise KeyError("metadata must contain a subject_id column.")

    tensor_tct = to_trial_channel_time(tensor)
    rows = []
    spectra_records = []
    subject_groups = list(metadata.groupby("subject_id", sort=True))
    analyze_start = time.perf_counter()
    logging.info(
        "Low-rank analysis start: data_type=%s, center_mode=%s, tensor_shape=%s, "
        "subjects=%d, views=%s",
        data_type,
        center_mode,
        tuple(tensor_tct.shape),
        len(subject_groups),
        ",".join(views),
    )

    analyzed_subjects = 0
    for subject_index, (subject_id, group) in enumerate(subject_groups, start=1):
        indices = group.index.to_numpy(dtype=np.int64)
        if len(indices) < min_trials:
            logging.info(
                "Skip subject_id=%s for %s/%s: subject=%d/%d, "
                "n_trials=%d < min_trials=%d",
                subject_id,
                data_type,
                center_mode,
                subject_index,
                len(subject_groups),
                len(indices),
                min_trials,
            )
            continue

        subject_start = time.perf_counter()
        subject_tensor = tensor_tct[indices]
        if center_mode == "trial_centered":
            subject_tensor = subject_tensor - subject_tensor.mean(dim=0, keepdim=True)
        elif center_mode != "raw":
            raise ValueError(f"Unknown center_mode: {center_mode}")

        analyzed_subjects += 1
        logging.info(
            "Subject analysis start: data_type=%s, center_mode=%s, subject_id=%s, "
            "subject=%d/%d, n_trials=%d, n_channels=%d, n_times=%d",
            data_type,
            center_mode,
            subject_id,
            subject_index,
            len(subject_groups),
            subject_tensor.shape[0],
            subject_tensor.shape[1],
            subject_tensor.shape[2],
        )

        view_cache: dict = {}
        for view in views:
            view_start = time.perf_counter()
            log_context = (
                f"[data={data_type} center={center_mode} "
                f"subject={subject_id} view={view}] "
            )
            logging.info("%sView unfold start", log_context)
            matrix, view_info = view_unfold(
                tensor=subject_tensor,
                view=view,
                frequency_representation=frequency_representation,
                spatial_transform=spatial_transform,
                channel_view_space=channel_view_space,
                tf_n_fft=tf_n_fft,
                tf_hop_length=tf_hop_length,
                cache=view_cache,
                log_context=log_context,
            )
            unfold_elapsed = time.perf_counter() - view_start
            svd_start = time.perf_counter()
            logging.info(
                "%sSVD start: matrix_shape=%s, dtype=%s, unfold_time=%s",
                log_context,
                tuple(matrix.shape),
                matrix.dtype,
                format_elapsed(unfold_elapsed),
            )
            singular_values, energy, cumulative_energy = singular_value_energy(matrix)
            svd_elapsed = time.perf_counter() - svd_start
            rank90 = rank_at_energy(cumulative_energy, 0.90)
            rank95 = rank_at_energy(cumulative_energy, 0.95)
            rank99 = rank_at_energy(cumulative_energy, 0.99)
            eff_rank = effective_rank(energy)
            logging.info(
                "%sSVD finished: spectrum_length=%d, rank95=%d, "
                "effective_rank=%.3f, svd_time=%s, view_time=%s",
                log_context,
                len(energy),
                rank95,
                eff_rank,
                format_elapsed(svd_elapsed),
                format_elapsed(time.perf_counter() - view_start),
            )
            rows.append(
                {
                    "data_type": data_type,
                    "subject_id": subject_id,
                    "center_mode": center_mode,
                    "hosvd_mode": view,
                    "view": view,
                    "n_trials": int(subject_tensor.shape[0]),
                    "n_channels": int(subject_tensor.shape[1]),
                    "n_times": int(subject_tensor.shape[2]),
                    "n_frequencies": int(view_info.get("n_frequencies", 0)),
                    "n_tf_times": int(view_info.get("n_tf_times", 0)),
                    "tf_n_fft": int(view_info.get("tf_n_fft", 0)),
                    "tf_hop_length": int(view_info.get("tf_hop_length", 0)),
                    "channel_view_space": (
                        channel_view_space
                        if view
                        in ("channel", "time_frequency", "time_space", "frequency_space")
                        else ""
                    ),
                    "frequency_representation": (
                        frequency_representation
                        if view
                        in (
                            "frequency",
                            "time_frequency",
                            "time_space",
                            "frequency_space",
                        )
                        else ""
                    ),
                    "pair_axes": view_info.get("pair_axes", ""),
                    "joint_tensor_shape": view_info.get("joint_tensor_shape", ""),
                    "spatial_shape": view_info.get("spatial_shape", ""),
                    "spatial_points": int(
                        view_info.get(
                            "spatial_points",
                            subject_tensor.shape[1] if view == "channel" else 0,
                        )
                    ),
                    "matrix_rows": int(matrix.shape[0]),
                    "matrix_cols": int(matrix.shape[1]),
                    "rank90": rank90,
                    "rank95": rank95,
                    "rank99": rank99,
                    "effective_rank": eff_rank,
                    "top1_energy": topk_energy(energy, 1),
                    "top5_energy": topk_energy(energy, 5),
                    "spectrum_length": int(len(energy)),
                }
            )
            spectra_records.append(
                {
                    "data_type": data_type,
                    "subject_id": subject_id,
                    "center_mode": center_mode,
                    "hosvd_mode": view,
                    "view": view,
                    "matrix_shape": tuple(int(size) for size in matrix.shape),
                    "view_info": view_info,
                    "singular_values": singular_values,
                    "energy": energy,
                    "cumulative_energy": cumulative_energy,
                }
            )
        logging.info(
            "Subject analysis finished: data_type=%s, center_mode=%s, "
            "subject_id=%s, elapsed=%s",
            data_type,
            center_mode,
            subject_id,
            format_elapsed(time.perf_counter() - subject_start),
        )

    logging.info(
        "Low-rank analysis finished: data_type=%s, center_mode=%s, "
        "analyzed_subjects=%d/%d, rows=%d, elapsed=%s",
        data_type,
        center_mode,
        analyzed_subjects,
        len(subject_groups),
        len(rows),
        format_elapsed(time.perf_counter() - analyze_start),
    )
    return rows, spectra_records


def spectra_to_npz_arrays(spectra_records: list[dict]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    manifest = []

    for record in spectra_records:
        prefix = "__".join(
            [
                safe_key(record["center_mode"]),
                safe_key(record["data_type"]),
                f"subject_{safe_key(record['subject_id'])}",
                safe_key(record["hosvd_mode"]),
            ]
        )
        arrays[f"{prefix}__singular_values"] = record["singular_values"]
        arrays[f"{prefix}__energy"] = record["energy"]
        arrays[f"{prefix}__cumulative_energy"] = record["cumulative_energy"]
        manifest.append(
            {
                "prefix": prefix,
                "center_mode": record["center_mode"],
                "data_type": record["data_type"],
                "subject_id": str(record["subject_id"]),
                "hosvd_mode": record["hosvd_mode"],
                "view": record.get("view", record["hosvd_mode"]),
                "matrix_shape": list(record.get("matrix_shape", ())),
                "view_info": record.get("view_info", {}),
            }
        )

    arrays["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=False))
    return arrays


def pad_nan(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.empty((0, 0), dtype=np.float64)
    max_len = max(len(array) for array in arrays)
    padded = np.full((len(arrays), max_len), np.nan, dtype=np.float64)
    for index, array in enumerate(arrays):
        padded[index, : len(array)] = array
    return padded


def plot_cumulative_energy(
    spectra_records: list[dict],
    center_mode: str,
    views: tuple[str, ...],
    output_path: Path,
) -> None:
    plot_start = time.perf_counter()
    logging.info(
        "Plot cumulative energy start: center_mode=%s, views=%s, output=%s",
        center_mode,
        ",".join(views),
        output_path,
    )
    fig, axes = plt.subplots(
        1,
        len(views),
        figsize=(5 * len(views), 4),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    fig.suptitle(f"Cumulative singular-value energy ({center_mode})")

    for axis, view in zip(axes, views):
        for data_type in DATA_TYPE_ORDER:
            curves = [
                record["cumulative_energy"]
                for record in spectra_records
                if record["center_mode"] == center_mode
                and record["hosvd_mode"] == view
                and record["data_type"] == data_type
            ]
            if not curves:
                continue

            padded = pad_nan(curves)
            x = np.arange(1, padded.shape[1] + 1)
            mean_curve = np.nanmean(padded, axis=0)
            std_curve = np.nanstd(padded, axis=0)
            axis.plot(
                x,
                mean_curve,
                label=data_type,
                color=PLOT_COLORS[data_type],
                linewidth=2,
            )
            axis.fill_between(
                x,
                np.clip(mean_curve - std_curve, 0, 1),
                np.clip(mean_curve + std_curve, 0, 1),
                color=PLOT_COLORS[data_type],
                alpha=0.15,
                linewidth=0,
            )

        axis.set_title(f"{view} view")
        axis.set_xlabel("rank index")
        axis.grid(True, alpha=0.3)
        axis.set_ylim(0, 1.02)

    axes[0].set_ylabel("cumulative energy")
    axes[-1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logging.info(
        "Plot cumulative energy finished: output=%s, elapsed=%s",
        output_path,
        format_elapsed(time.perf_counter() - plot_start),
    )


def plot_rank95(
    summary: pd.DataFrame,
    center_mode: str,
    views: tuple[str, ...],
    output_path: Path,
) -> None:
    plot_start = time.perf_counter()
    logging.info(
        "Plot rank95 start: center_mode=%s, views=%s, output=%s",
        center_mode,
        ",".join(views),
        output_path,
    )
    fig, axes = plt.subplots(
        1,
        len(views),
        figsize=(4.5 * len(views), 4),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    fig.suptitle(f"Rank95 by subject ({center_mode})")

    rng = np.random.default_rng(0)
    for axis, view in zip(axes, views):
        filtered = summary[
            (summary["center_mode"] == center_mode) & (summary["hosvd_mode"] == view)
        ]

        values = [
            filtered.loc[filtered["data_type"] == data_type, "rank95"].to_numpy()
            for data_type in DATA_TYPE_ORDER
        ]
        non_empty_values = [value for value in values if len(value) > 0]

        if non_empty_values:
            axis.boxplot(values, labels=DATA_TYPE_ORDER, showmeans=True)
            for index, value in enumerate(values, start=1):
                if len(value) == 0:
                    continue
                jitter = rng.normal(0, 0.03, size=len(value))
                axis.scatter(
                    np.full(len(value), index) + jitter,
                    value,
                    color=PLOT_COLORS[DATA_TYPE_ORDER[index - 1]],
                    alpha=0.65,
                    s=18,
                )
        else:
            axis.text(
                0.5,
                0.5,
                "No subjects passed min_trials",
                ha="center",
                va="center",
            )

        axis.set_title(f"{view} view")
        axis.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("rank95")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logging.info(
        "Plot rank95 finished: output=%s, elapsed=%s",
        output_path,
        format_elapsed(time.perf_counter() - plot_start),
    )


def save_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    split_path: str,
    clean: torch.Tensor,
    adv: torch.Tensor,
    labels: torch.Tensor,
    metadata: pd.DataFrame,
    summary: pd.DataFrame,
    spectra_records: list[dict],
) -> None:
    save_start = time.perf_counter()
    logging.info("Saving outputs start: output_dir=%s", output_dir)
    perturb = adv - clean
    if not torch.allclose(perturb, adv - clean):
        raise RuntimeError("Internal perturbation consistency check failed.")

    metadata.to_csv(output_dir / "metadata.csv", index=False)
    summary.to_csv(output_dir / "hosvd_summary.csv", index=False)
    logging.info(
        "Saved metadata and summary: metadata_rows=%d, summary_rows=%d, elapsed=%s",
        len(metadata),
        len(summary),
        format_elapsed(time.perf_counter() - save_start),
    )

    np.savez_compressed(output_dir / "spectra.npz", **spectra_to_npz_arrays(spectra_records))
    logging.info(
        "Saved spectra.npz: spectra_records=%d, elapsed=%s",
        len(spectra_records),
        format_elapsed(time.perf_counter() - save_start),
    )

    torch.save(
        {
            "clean": clean,
            "adv": adv,
            "perturb": perturb,
            "labels": labels,
            "metadata": metadata.to_dict(orient="list"),
            "args": vars(args),
            "split_path": split_path,
        },
        output_dir / "bundle.pt",
    )
    logging.info(
        "Saved bundle.pt, elapsed=%s",
        format_elapsed(time.perf_counter() - save_start),
    )

    run_config = vars(args).copy()
    run_config["split_path"] = split_path
    run_config["repo_root"] = str(REPO_ROOT)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, ensure_ascii=False)
    logging.info("Saving outputs finished: elapsed=%s", format_elapsed(time.perf_counter() - save_start))


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    seed_everything(args.seed)
    views = tuple(dict.fromkeys(args.views))
    args.views = list(views)

    protocol_tag = get_protocol_tag(use_ea=args.use_ea)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args, protocol_tag)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    setup_logging(output_dir)

    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    logging.info("Args: %s", args)
    logging.info("Output directory: %s", output_dir)
    logging.info("Device: %s", device)
    logging.info("Data protocol: %s, use_ea=%s", protocol_tag, args.use_ea)
    logging.info(
        "Views: %s, frequency_representation=%s, channel_view_space=%s, "
        "tf_n_fft=%d, tf_hop_length=%d",
        ",".join(views),
        args.frequency_representation,
        args.channel_view_space,
        args.tf_n_fft,
        args.tf_hop_length,
    )

    dataset, info = DATASET_LOADERS[args.dataset]()
    spatial_transform = None
    if views_need_spatial_projection(views):
        spatial_transform = build_channel_spatial_transform(
            dataset=args.dataset,
            channel_view_space=args.channel_view_space,
        )
    _, _, test_dataset, split_path = prepare_subject_fold(
        dataset_name=args.dataset,
        dataset=dataset,
        info=info,
        fold_id=args.fold,
        seed=args.seed,
        use_ea=args.use_ea,
    )
    logging.info("Split path: %s", split_path)
    full_test_samples = len(test_dataset)
    test_dataset, _ = sample_dataset(
        dataset=test_dataset,
        sample_num=args.sample_num,
        seed=args.seed,
    )
    logging.info(
        "Test samples: %d, sampled samples: %d, sample_num=%d",
        full_test_samples,
        len(test_dataset),
        args.sample_num,
    )

    model = load_clean_model(args, info, protocol_tag, device)
    clean, adv, labels, metadata = collect_clean_adv_and_metadata(
        args=args,
        test_dataset=test_dataset,
        info=info,
        model=model,
        device=device,
    )
    logging.info(
        "Adversarial generation stage finished: elapsed=%s",
        format_elapsed(time.perf_counter() - run_start),
    )
    subject_counts = metadata.groupby("subject_id").size()
    eligible_subjects = int((subject_counts >= args.min_trials).sum())
    planned_svd_calls = (
        len(CENTER_MODES)
        * len(DATA_TYPE_ORDER)
        * len(views)
        * eligible_subjects
    )
    logging.info(
        "Low-rank analysis plan: subjects=%d, eligible_subjects=%d, "
        "min_trials=%d, center_modes=%d, data_types=%d, views=%d, "
        "planned_svd_calls=%d",
        len(subject_counts),
        eligible_subjects,
        args.min_trials,
        len(CENTER_MODES),
        len(DATA_TYPE_ORDER),
        len(views),
        planned_svd_calls,
    )
    perturb = adv - clean

    all_rows = []
    spectra_records = []
    tensors = {
        "clean": clean,
        "adv": adv,
        "perturb": perturb,
    }

    for center_mode in CENTER_MODES:
        for data_type in DATA_TYPE_ORDER:
            stage_start = time.perf_counter()
            logging.info(
                "Analysis stage dispatch: center_mode=%s, data_type=%s",
                center_mode,
                data_type,
            )
            rows, records = analyze_tensor_set(
                data_type=data_type,
                tensor=tensors[data_type],
                metadata=metadata,
                center_mode=center_mode,
                min_trials=args.min_trials,
                views=views,
                frequency_representation=args.frequency_representation,
                spatial_transform=spatial_transform,
                channel_view_space=args.channel_view_space,
                tf_n_fft=args.tf_n_fft,
                tf_hop_length=args.tf_hop_length,
            )
            all_rows.extend(rows)
            spectra_records.extend(records)
            logging.info(
                "Analysis stage completed: center_mode=%s, data_type=%s, "
                "rows=%d, records=%d, elapsed=%s",
                center_mode,
                data_type,
                len(rows),
                len(records),
                format_elapsed(time.perf_counter() - stage_start),
            )

    summary = pd.DataFrame(all_rows)
    if summary.empty:
        raise RuntimeError(
            "No subject had enough trials for analysis. "
            f"Try lowering --min_trials below {args.min_trials}."
        )
    summary["hosvd_mode"] = pd.Categorical(
        summary["hosvd_mode"],
        categories=list(VIEW_ORDER),
        ordered=True,
    )
    summary = (
        summary.sort_values(["center_mode", "data_type", "subject_id", "hosvd_mode"])
        .reset_index(drop=True)
    )
    summary["hosvd_mode"] = summary["hosvd_mode"].astype(str)
    summary["view"] = summary["hosvd_mode"]

    save_outputs(
        output_dir=output_dir,
        args=args,
        split_path=split_path,
        clean=clean,
        adv=adv,
        labels=labels,
        metadata=metadata,
        summary=summary,
        spectra_records=spectra_records,
    )

    plot_cumulative_energy(
        spectra_records,
        center_mode="raw",
        views=views,
        output_path=output_dir / "cum_energy_raw.png",
    )
    plot_rank95(
        summary,
        center_mode="raw",
        views=views,
        output_path=output_dir / "rank95_raw.png",
    )
    plot_cumulative_energy(
        spectra_records,
        center_mode="trial_centered",
        views=views,
        output_path=output_dir / "cum_energy_trial_centered.png",
    )
    plot_rank95(
        summary,
        center_mode="trial_centered",
        views=views,
        output_path=output_dir / "rank95_trial_centered.png",
    )

    logging.info("Saved metadata, summary, spectra, bundle, and plots to %s", output_dir)
    logging.info("Summary rows: %d", len(summary))
    logging.info("Run finished: total_elapsed=%s", format_elapsed(time.perf_counter() - run_start))


if __name__ == "__main__":
    main()
