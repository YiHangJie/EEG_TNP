import argparse
import csv
import datetime
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from runtime_env import configure_runtime_env

configure_runtime_env()

import numpy as np
import tensorly as tl
import torch
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/eegap_matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorly.decomposition import tensor_ring
from tensorly.tr_tensor import tr_to_tensor
from torcheeg.models import EEGNet
from tqdm import tqdm

from data.load import load_thubenchmark
from data.subject_ea import prepare_subject_fold
from models.model_args import get_model_args
from purify import interpolate, inv_interpolate
from TN.opt import Config
from TN.rank_growth import PTR_3d_rank_growth
from TN.utils import get_TN_args
from utils.experiment_artifacts import eeg_classification_collate, load_tensor_payload, safe_token, short_protocol_tag


DEFAULT_CONFIG = "PTR3d_8_2048_rank25_3d_interpolate.yaml"
DEFAULT_RANKS = "1,2,3,4,5,6,7,8"
DEFAULT_EPS = 0.1
DEFAULT_FOLD = 0
DEFAULT_SAMPLE_NUM = 32
DEFAULT_SEED = 42
DEFAULT_PAIR_SAMPLE_NUM = 512
DEFAULT_CONSISTENCY_VERSION = "consistency2"
DEFAULT_CONSISTENCY_RANK_TAG = "rank25-30"


def parse_rank_list(value):
    """解析逗号分隔的 rank 列表，保证 rank 为正整数且不重复。"""
    ranks = []
    for item in str(value).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        rank = int(item)
        if rank <= 0:
            raise argparse.ArgumentTypeError("All ranks must be positive integers.")
        ranks.append(rank)
    if not ranks:
        raise argparse.ArgumentTypeError("At least one rank must be provided.")
    if len(set(ranks)) != len(ranks):
        raise argparse.ArgumentTypeError("Ranks must not contain duplicates.")
    return ranks


def eps_to_path_token(eps):
    """生成现有 checkpoint/adversarial 文件名使用的 eps 字符串。"""
    return f"{eps:g}"


def eps_to_tag_token(eps):
    """生成实验 tag 使用的 eps 字符串，例如 0.03 -> 0p03。"""
    return eps_to_path_token(eps).replace(".", "p")


def build_consistency_tag(consistency_version, consistency_rank_tag, pair_sample_num, eps,
                          consistency_tag=None):
    """生成 checkpoint 和 paired cache 共享的实验 tag。"""
    if consistency_tag:
        return consistency_tag
    eps_tag = eps_to_tag_token(eps)
    return f"{consistency_version}_{consistency_rank_tag}_n{pair_sample_num}_eps{eps_tag}"


def build_pair_kind(consistency_version, pair_kind=None):
    """生成 paired cache 文件名和目录中的 kind 字段。"""
    return pair_kind or f"train_pair_{consistency_version}"


def build_adv_model_tag(consistency_version, consistency_rank_tag, pair_sample_num, eps,
                        consistency_tag=None, adv_model_tag=None):
    """生成 ad_data 文件名中表示被攻击模型来源的 tag。"""
    if adv_model_tag:
        return safe_token(adv_model_tag)
    return safe_token(build_consistency_tag(
        consistency_version=consistency_version,
        consistency_rank_tag=consistency_rank_tag,
        pair_sample_num=pair_sample_num,
        eps=eps,
        consistency_tag=consistency_tag,
    ))


def build_default_pair_path(dataset, fold, seed, eps, config, pair_sample_num,
                            consistency_version, consistency_rank_tag,
                            consistency_tag=None, pair_kind=None):
    config_tag = Path(config).stem
    eps_path = eps_to_path_token(eps)
    consistency_tag = build_consistency_tag(
        consistency_version=consistency_version,
        consistency_rank_tag=consistency_rank_tag,
        pair_sample_num=pair_sample_num,
        eps=eps,
        consistency_tag=consistency_tag,
    )
    pair_kind = build_pair_kind(consistency_version, pair_kind=pair_kind)
    file_name = (
        f"{dataset}_eegnet_no_ea_fold{fold}_seed{seed}_{pair_kind}_"
        f"autoattack_eps{eps_path}_{config_tag}_n{pair_sample_num}_tag{consistency_tag}.pth"
    )
    return str(Path("purified_data") / pair_kind / file_name)


def build_default_checkpoint_path(dataset, fold, seed, eps, pair_sample_num,
                                  consistency_version, consistency_rank_tag,
                                  consistency_tag=None, checkpoint_lr=None,
                                  checkpoint_weight_decay=None):
    eps_path = eps_to_path_token(eps)
    consistency_tag = build_consistency_tag(
        consistency_version=consistency_version,
        consistency_rank_tag=consistency_rank_tag,
        pair_sample_num=pair_sample_num,
        eps=eps,
        consistency_tag=consistency_tag,
    )
    file_name = (
        f"{dataset}_eegnet_train_only_subject_no_ea_subject_split_madry_"
        f"eps{eps_path}_{seed}_fold{fold}"
    )
    if checkpoint_lr is not None and checkpoint_weight_decay is not None:
        file_name += f"_{checkpoint_lr}_{checkpoint_weight_decay}"
    file_name += f"_{consistency_tag}_best.pth"
    return str(Path("checkpoints") / file_name)


def build_default_ad_data_path(dataset, model, fold, seed, eps, attack, at_strategy,
                               use_ea, adv_model_tag):
    """按 attack.py 的保存规则生成默认测试集对抗样本路径。"""
    protocol_short = short_protocol_tag(use_ea)
    model_tag = safe_token(adv_model_tag)
    if model_tag == safe_token(at_strategy):
        source_tag = at_strategy
    else:
        source_tag = f"{model_tag}_{at_strategy}"
    file_name = (
        f"{dataset}_{model}_{protocol_short}_{source_tag}_{attack}_"
        f"eps{eps_to_path_token(eps)}_seed{seed}_fold{fold}.pth"
    )
    return str(Path("ad_data") / file_name)


def resolve_ad_data_path(args, adv_model_tag):
    """优先精确匹配当前 checkpoint tag 对应的 ad_data；必要时用 glob 给出明确报错。"""
    if args.ad_data_path:
        path = Path(args.ad_data_path)
        if not path.exists():
            raise FileNotFoundError(f"ad_data_path does not exist: {path}")
        return str(path)

    exact_path = Path(build_default_ad_data_path(
        dataset=args.dataset,
        model=args.model,
        fold=args.fold,
        seed=args.seed,
        eps=args.eps,
        attack=args.attack,
        at_strategy=args.at_strategy,
        use_ea=args.use_ea,
        adv_model_tag=adv_model_tag,
    ))
    if exact_path.exists():
        return str(exact_path)

    protocol_short = short_protocol_tag(args.use_ea)
    pattern = (
        f"{args.dataset}_{args.model}_{protocol_short}_*_{args.at_strategy}_"
        f"{args.attack}_eps{eps_to_path_token(args.eps)}_seed{args.seed}_fold{args.fold}.pth"
    )
    candidates = sorted(Path(args.ad_data_dir).glob(pattern))
    tagged_candidates = [
        path for path in candidates
        if safe_token(adv_model_tag) in path.name
    ]
    if len(tagged_candidates) == 1:
        return str(tagged_candidates[0])
    if len(tagged_candidates) > 1:
        joined = ", ".join(str(path) for path in tagged_candidates)
        raise FileExistsError(
            f"Multiple ad_data files match adv_model_tag={adv_model_tag}: {joined}. "
            "Use --ad_data_path to choose one explicitly."
        )
    if len(candidates) == 1:
        return str(candidates[0])
    if candidates:
        joined = ", ".join(str(path) for path in candidates)
        raise FileExistsError(
            "Multiple ad_data candidates found, and none uniquely matches the current tag. "
            f"Candidates: {joined}. Use --ad_data_path to choose one explicitly."
        )
    raise FileNotFoundError(
        "No matching adversarial test data found. "
        f"Expected exact path: {exact_path}; glob pattern: {Path(args.ad_data_dir) / pattern}. "
        "Run attack.py with --save_adv for the target checkpoint, or pass --ad_data_path."
    )


def resolve_input_paths(args, config_name, adv_model_tag):
    checkpoint_path = args.checkpoint_path or build_default_checkpoint_path(
        dataset=args.dataset,
        fold=args.fold,
        seed=args.seed,
        eps=args.eps,
        pair_sample_num=args.pair_sample_num,
        consistency_version=args.consistency_version,
        consistency_rank_tag=args.consistency_rank_tag,
        consistency_tag=args.consistency_tag,
        checkpoint_lr=args.checkpoint_lr,
        checkpoint_weight_decay=args.checkpoint_weight_decay,
    )

    missing = []
    if not Path(checkpoint_path).exists():
        missing.append(f"checkpoint_path={checkpoint_path}")

    pair_path = None
    ad_data_path = None
    if args.data_source == "pair" or args.pair_path:
        pair_path = args.pair_path or build_default_pair_path(
            dataset=args.dataset,
            fold=args.fold,
            seed=args.seed,
            eps=args.eps,
            config=config_name,
            pair_sample_num=args.pair_sample_num,
            consistency_version=args.consistency_version,
            consistency_rank_tag=args.consistency_rank_tag,
            consistency_tag=args.consistency_tag,
            pair_kind=args.pair_kind,
        )
        if not Path(pair_path).exists():
            missing.append(f"pair_path={pair_path}")
    else:
        ad_data_path = resolve_ad_data_path(args, adv_model_tag)

    if missing:
        raise FileNotFoundError(
            "Resolved input file(s) do not exist for the requested eps. "
            f"eps={args.eps}, missing: {', '.join(missing)}. "
            "Use --ad_data_path/--pair_path and/or --checkpoint_path to point to existing files."
        )
    return pair_path, ad_data_path, checkpoint_path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze classifier prediction changes after TensorLy tensor-ring reconstruction."
    )
    parser.add_argument("--pair_path", type=str, default=None)
    parser.add_argument("--ad_data_path", type=str, default=None,
                        help="explicit adversarial test data .pth path; defaults to matching ad_data file.")
    parser.add_argument("--ad_data_dir", type=str, default="ad_data")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=str, default="thubenchmark", choices=["thubenchmark"])
    parser.add_argument("--model", type=str, default="eegnet", choices=["eegnet"])
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--fold", type=int, default=DEFAULT_FOLD)
    parser.add_argument("--attack", type=str, default="autoattack",
                        choices=["fgsm", "pgd", "cw", "autoattack"])
    parser.add_argument("--at_strategy", type=str, default="madry",
                        choices=["madry", "fbf", "trades", "clean"])
    parser.add_argument("--use_ea", dest="use_ea", action="store_true", default=False,
                        help="use EA-aligned test data when matching ad_data.")
    parser.add_argument("--no_ea", dest="use_ea", action="store_false",
                        help="use raw/no-EA test data when matching ad_data.")
    parser.add_argument("--data_source", type=str, default="ad_data", choices=["ad_data", "pair"],
                        help="ad_data uses test split clean samples plus saved adversarial samples; pair keeps legacy train_pair behavior.")
    parser.add_argument("--pair_sample_num", type=int, default=DEFAULT_PAIR_SAMPLE_NUM)
    parser.add_argument("--consistency_version", type=str, default=DEFAULT_CONSISTENCY_VERSION,
                        help="consistency experiment version, e.g. consistency2 or consistancy.")
    parser.add_argument("--consistency_rank_tag", type=str, default=DEFAULT_CONSISTENCY_RANK_TAG,
                        help="rank fragment used when auto-building consistency_tag.")
    parser.add_argument("--consistency_tag", type=str, default=None,
                        help="full tag used in pair/checkpoint names; overrides auto-built tag.")
    parser.add_argument("--pair_kind", type=str, default=None,
                        help="paired cache kind/directory, default train_pair_<consistency_version>.")
    parser.add_argument("--adv_model_tag", type=str, default=None,
                        help="model/source tag used in ad_data file names; defaults to consistency_tag.")
    parser.add_argument("--checkpoint_lr", type=str, default=None,
                        help="optional checkpoint lr fragment, e.g. 0.001.")
    parser.add_argument("--checkpoint_weight_decay", type=str, default=None,
                        help="optional checkpoint weight_decay fragment, e.g. 0.0001.")
    parser.add_argument("--sample_num", type=int, default=DEFAULT_SAMPLE_NUM)
    parser.add_argument("--ranks", type=parse_rank_list, default=parse_rank_list(DEFAULT_RANKS))
    parser.add_argument("--analysis_mode", type=str, default="tensorly_tr",
                        choices=["tensorly_tr", "rank_growth"],
                        help="tensorly_tr keeps the legacy TensorLy TR sweep; rank_growth runs PTR_3d_rank_growth.")
    parser.add_argument("--rank_growth_steps_per_rank", type=int, default=None,
                        help="override rank_growth_steps_per_rank for rank_growth analysis.")
    parser.add_argument("--rank_growth_js_threshold", type=float, default=None,
                        help="override rank_growth_js_threshold for rank_growth analysis.")
    parser.add_argument("--rank_growth_max_mse_to_input", type=float, default=None,
                        help="override rank_growth_max_mse_to_input for rank_growth analysis.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tr_mode", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="tensor_ring_rank_analysis/results")
    parser.add_argument("--plot_format", type=str, default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--plot_dpi", type=int, default=180)
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--plot_only", action="store_true",
                        help="only generate plots from existing output CSVs; currently supports rank_growth.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    """固定随机源，保证样本抽取和分类结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_config_name(config):
    """purify.py 期望 config 是 configs/<dataset>/ 下的文件名。"""
    return os.path.basename(config)


def load_config(dataset, config_name):
    config_path = ROOT_DIR / "configs" / dataset / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file), config_path


def load_pair_payload(path):
    payload = torch_load_cpu(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected paired rank-analysis dict payload, got {type(payload)} from {path}.")

    required_keys = ["x", "x_adv", "labels"]
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise ValueError(f"Missing keys in paired payload {path}: {missing_keys}")

    x = torch.as_tensor(payload["x"]).detach().cpu().float()
    x_adv = torch.as_tensor(payload["x_adv"]).detach().cpu().float()
    labels = torch.as_tensor(payload["labels"]).detach().cpu().long().view(-1)
    source_indices = payload.get("source_indices", list(range(labels.numel())))
    meta = payload.get("meta", {})
    meta = meta if isinstance(meta, dict) else {}

    if x.shape != x_adv.shape:
        raise ValueError(f"x and x_adv shape mismatch: {tuple(x.shape)} vs {tuple(x_adv.shape)}")
    if x.size(0) != labels.numel():
        raise ValueError(f"Sample/label count mismatch: {x.size(0)} samples vs {labels.numel()} labels")
    if len(source_indices) != labels.numel():
        raise ValueError(f"source_indices length mismatch: {len(source_indices)} vs {labels.numel()}")
    if x.dim() != 4:
        raise ValueError(f"Expected EEG batch shape (N,1,C,T), got {tuple(x.shape)}")

    return {
        "x": x,
        "x_adv": x_adv,
        "labels": labels,
        "source_indices": list(source_indices),
        "meta": meta,
    }


def load_test_ad_payload(args, dataset, info, ad_data_path):
    """加载测试 split 的 clean 样本，并与 attack.py 保存的 ad_data 按顺序配对。"""
    _, _, test_dataset, split_path = prepare_subject_fold(
        dataset_name=args.dataset,
        dataset=dataset,
        info=info,
        fold_id=args.fold,
        seed=args.seed,
        use_ea=args.use_ea,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=eeg_classification_collate,
    )

    clean_parts = []
    clean_label_parts = []
    for data, target in test_loader:
        clean_parts.append(data.detach().cpu().float())
        clean_label_parts.append(target.detach().cpu().long().view(-1))
    clean_data = torch.cat(clean_parts, dim=0)
    clean_labels = torch.cat(clean_label_parts, dim=0)

    adv_data, adv_labels, adv_meta = load_tensor_payload(ad_data_path)
    adv_source_indices = None
    if adv_data.size(0) == clean_data.size(0):
        # attack.py 保存的是完整 test_loader 顺序，clean/adversarial 可按位置直接配对。
        adv_source_indices = list(range(clean_labels.numel()))
    else:
        # 若 ad_data 是测试集随机子集，必须依赖生成时保存的 source_indices 恢复 clean 样本。
        raw_indices = adv_meta.get("source_indices") if isinstance(adv_meta, dict) else None
        if raw_indices is None:
            raise ValueError(
                "ad_data appears to be a subset of the test split, but its meta does not contain "
                "source_indices. Randomly re-selecting clean test samples cannot guarantee overlap. "
                f"clean_count={clean_data.size(0)}, adv_count={adv_data.size(0)}, ad_data_path={ad_data_path}"
            )
        adv_source_indices = [int(index) for index in raw_indices]
        if len(adv_source_indices) != adv_data.size(0):
            raise ValueError(
                "ad_data source_indices length mismatch. "
                f"indices={len(adv_source_indices)}, adv_count={adv_data.size(0)}, ad_data_path={ad_data_path}"
            )
        if min(adv_source_indices, default=0) < 0 or max(adv_source_indices, default=-1) >= clean_data.size(0):
            raise ValueError(
                "ad_data source_indices are outside the test split range. "
                f"test_count={clean_data.size(0)}, ad_data_path={ad_data_path}"
            )
        index_tensor = torch.as_tensor(adv_source_indices, dtype=torch.long)
        clean_data = clean_data[index_tensor]
        clean_labels = clean_labels[index_tensor]

    if clean_data.shape != adv_data.shape:
        raise ValueError(
            "Clean test data and ad_data shape mismatch. "
            f"clean={tuple(clean_data.shape)}, adv={tuple(adv_data.shape)}, ad_data_path={ad_data_path}"
        )
    if clean_labels.numel() != adv_labels.numel():
        raise ValueError(
            "Clean test labels and ad_data labels length mismatch. "
            f"clean={clean_labels.numel()}, adv={adv_labels.numel()}, ad_data_path={ad_data_path}"
        )
    if not torch.equal(clean_labels, adv_labels):
        mismatch = clean_labels.ne(adv_labels).nonzero(as_tuple=False).view(-1)[:10].tolist()
        raise ValueError(
            "Clean test labels differ from ad_data labels; the adversarial file may not match "
            f"the requested split/model. First mismatched positions: {mismatch}"
        )

    meta = {
        "kind": "test_ad_data_pair",
        "source_split": "test",
        "split_path": split_path,
        "use_ea": args.use_ea,
        "ad_data_path": ad_data_path,
        "ad_meta": adv_meta,
    }
    return {
        "x": clean_data,
        "x_adv": adv_data,
        "labels": clean_labels,
        # source_index 表示测试 split 内的位置；全量 ad_data 为 0..N-1，子集则来自 meta。
        "source_indices": adv_source_indices,
        "meta": meta,
    }


def select_samples(sample_count, sample_num, seed):
    if sample_num <= 0:
        raise ValueError("--sample_num must be positive.")
    if sample_num > sample_count:
        raise ValueError(f"--sample_num={sample_num} exceeds available sample count {sample_count}.")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(sample_count, generator=generator)[:sample_num]


def load_classifier(checkpoint_path, info, device):
    model = EEGNet(**get_model_args("eegnet", "thubenchmark", info)).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        checkpoint = {
            key.removeprefix("module."): value
            for key, value in checkpoint.items()
        }
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def validate_uniform_tr_ranks(pre_shape, ranks, tr_mode):
    """提前检查 TensorLy TR-SVD 的第一轮矩阵化 rank 约束。"""
    if tr_mode < 0 or tr_mode >= len(pre_shape):
        raise ValueError(f"--tr_mode must be in [0, {len(pre_shape) - 1}], got {tr_mode}.")

    ordered_shape = tuple(pre_shape[tr_mode:]) + tuple(pre_shape[:tr_mode])
    first_limit = min(ordered_shape[0], int(np.prod(ordered_shape[1:])))
    max_rank = int(np.sqrt(first_limit))
    invalid_ranks = [rank for rank in ranks if rank * rank > first_limit]
    if invalid_ranks:
        raise ValueError(
            "Invalid uniform TR ranks for TensorLy tensor_ring: "
            f"pre_shape={tuple(pre_shape)}, tr_mode={tr_mode}, first_limit={first_limit}, "
            f"max_uniform_rank={max_rank}, invalid={invalid_ranks}."
        )


def tensor_ring_reconstruct(sample, purify_args, strategy, sampling_rate, rank, tr_mode):
    """单样本 TensorLy tensor-ring 分解-重建，并还原到分类器输入格式。"""
    sample = sample.detach().cpu().float()
    pre_data = interpolate(purify_args, sample, sampling_rate).detach().cpu().float()

    # 三维 TR 有三个唯一 bond；TensorLy API 需要额外写出闭环 rank。
    rank_spec = [rank] * (pre_data.dim() + 1)
    tr_factors = tensor_ring(pre_data, rank=rank_spec, mode=tr_mode)
    reconstructed_pre = tr_to_tensor(tr_factors).detach().cpu().float()
    reconstructed = inv_interpolate(
        purify_args,
        reconstructed_pre,
        original_shape=sample.shape[-2:],
        strategy=strategy,
    ).detach().cpu().float()

    if reconstructed.shape != sample.shape:
        raise ValueError(
            f"Reconstructed shape mismatch: expected {tuple(sample.shape)}, got {tuple(reconstructed.shape)}"
        )
    return reconstructed


@torch.no_grad()
def classify_tensor_batch(model, data, batch_size, device):
    logits = []
    for start in range(0, data.size(0), batch_size):
        batch = data[start:start + batch_size].to(device)
        logits.append(model(batch).detach().cpu())
    return torch.cat(logits, dim=0)


def predict_for_source(source_type, samples, labels, ranks, model, purify_args, strategy,
                       sampling_rate, tr_mode, batch_size, device):
    rank_logits = []
    for rank in ranks:
        reconstructed_samples = []
        iterator = tqdm(
            range(samples.size(0)),
            desc=f"{source_type} rank={rank}",
            leave=False,
        )
        for sample_index in iterator:
            reconstructed = tensor_ring_reconstruct(
                samples[sample_index],
                purify_args=purify_args,
                strategy=strategy,
                sampling_rate=sampling_rate,
                rank=rank,
                tr_mode=tr_mode,
            )
            reconstructed_samples.append(reconstructed)

        reconstructed_batch = torch.stack(reconstructed_samples, dim=0)
        logits = classify_tensor_batch(model, reconstructed_batch, batch_size, device)
        rank_logits.append(logits)

    logits = torch.stack(rank_logits, dim=1)
    probs = torch.softmax(logits, dim=-1)
    probs_clamped = probs.clamp_min(1e-12)
    entropy = -(probs_clamped * probs_clamped.log()).sum(dim=-1)
    confidence, top1 = probs.max(dim=-1)
    correct = top1.eq(labels.view(-1, 1))
    return {
        "logits": logits,
        "probs": probs,
        "entropy": entropy,
        "confidence": confidence,
        "top1": top1,
        "correct": correct,
    }


def build_rank_growth_config(config, args):
    """复制 YAML 配置并应用 rank growth 分析用 override，避免修改原配置文件。"""
    config = dict(config)
    if args.rank_growth_steps_per_rank is not None:
        config["rank_growth_steps_per_rank"] = int(args.rank_growth_steps_per_rank)
    if args.rank_growth_js_threshold is not None:
        config["rank_growth_js_threshold"] = float(args.rank_growth_js_threshold)
    if args.rank_growth_max_mse_to_input is not None:
        config["rank_growth_max_mse_to_input"] = float(args.rank_growth_max_mse_to_input)
    return config


def config_to_namespace(config):
    """转换 YAML dict 为 TN 代码期望的 Config 对象。"""
    cfg = Config()
    for key, value in config.items():
        setattr(cfg, key, value)
    return cfg


def rank_growth_reconstruct_and_trace(
    sample,
    label,
    source_type,
    sample_id,
    source_index,
    model,
    purify_args,
    config,
    strategy,
    sampling_rate,
    device,
):
    """运行一次 PTR_3d_rank_growth，并返回该样本每个 rank block 的评估轨迹。"""
    sample = sample.detach().cpu().float()
    pre_data = interpolate(purify_args, sample, sampling_rate).detach().cpu().float()
    cfg = config_to_namespace(config)
    cfg.device_type = "gpu" if torch.cuda.is_available() else "cpu"
    tn_args = get_TN_args(cfg, pre_data.clone(), sampling_rate, None, cfg.device_type)
    tn = PTR_3d_rank_growth(**tn_args)

    eval_records = []

    def rank_eval_callback(recon_resized):
        # PTR_3d_rank_growth 当前不把 rank 显式传给 callback；
        # callback 调用顺序与 rank_growth_ranks 一致，因此这里用计数恢复 rank。
        rank_index = len(eval_records)
        rank = int(tn.rank_growth_ranks[rank_index])
        purified = inv_interpolate(
            purify_args,
            recon_resized,
            original_shape=sample.shape[-2:],
            strategy=strategy,
        ).detach().cpu().float()
        mse_to_input = torch.nn.functional.mse_loss(purified, sample).item()
        with torch.no_grad():
            logits = model(purified.unsqueeze(0).to(device)).detach().cpu().squeeze(0)
        eval_records.append({
            "rank": rank,
            "mse_to_input": float(mse_to_input),
            "logits": logits,
        })
        return {
            "mse_to_input": mse_to_input,
            "logits": logits,
        }

    tn.train(
        pre_data.clone().to(device),
        cfg,
        target_index=sample_id,
        logging=None,
        rank_eval_callback=rank_eval_callback,
    )

    eval_by_rank = {record["rank"]: record for record in eval_records}
    rows = []
    selected_rank = int(tn.selected_rank)
    for history_row in tn.dynamic_rank_history:
        rank = int(history_row["rank"])
        eval_row = eval_by_rank[rank]
        probs = torch.softmax(eval_row["logits"], dim=-1)
        confidence, top1 = probs.max(dim=-1)
        rows.append({
            "sample_id": sample_id,
            "source_index": source_index,
            "source_type": source_type,
            "label": int(label.item()),
            "rank": rank,
            "mse_to_input": float(eval_row["mse_to_input"]),
            "js_to_prev": history_row["js_to_prev"],
            "top1": int(top1.item()),
            "confidence": float(confidence.item()),
            "correct": int(top1.item() == int(label.item())),
            "mse_rel_delta_to_prev": history_row["mse_rel_delta_to_prev"],
            "top1_unchanged": history_row["top1_unchanged"],
            "fidelity_gate_pass": history_row["fidelity_gate_pass"],
            "rank_growth_max_mse_to_input": history_row["rank_growth_max_mse_to_input"],
            "rejected_by_mse_gate": bool(history_row["rejected_by_mse_gate"]),
            "loss": history_row["loss"],
            "selected": rank == selected_rank,
        })
    return rows, eval_records, selected_rank


def predict_rank_growth_for_source(source_type, samples, labels, source_indices, model,
                                   purify_args, config, strategy, sampling_rate, device):
    """对 clean/adv 样本运行 rank growth，返回 per-rank 历史行和原始 logits trace。"""
    history_rows = []
    traces = []
    iterator = tqdm(range(samples.size(0)), desc=f"{source_type} rank-growth", leave=False)
    for sample_id in iterator:
        rows, eval_records, selected_rank = rank_growth_reconstruct_and_trace(
            sample=samples[sample_id],
            label=labels[sample_id],
            source_type=source_type,
            sample_id=sample_id,
            source_index=source_indices[sample_id],
            model=model,
            purify_args=purify_args,
            config=config,
            strategy=strategy,
            sampling_rate=sampling_rate,
            device=device,
        )
        history_rows.extend(rows)
        traces.append({
            "sample_id": sample_id,
            "source_index": source_indices[sample_id],
            "selected_rank": selected_rank,
            "eval_records": eval_records,
        })
    return history_rows, traces


def build_rank_growth_summary_rows(history_rows):
    """按 source_type/rank 聚合 rank growth 轨迹，便于调 JS/MSE 阈值。"""
    grouped = defaultdict(list)
    for row in history_rows:
        grouped[(row["source_type"], row["rank"])].append(row)

    rows = []
    for (source_type, rank), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        selected = [row for row in items if row["selected"]]
        js_values = [float(row["js_to_prev"]) for row in items if row["js_to_prev"] is not None]
        mse_values = [float(row["mse_to_input"]) for row in items]
        confidence_values = [float(row["confidence"]) for row in items]
        rows.append({
            "source_type": source_type,
            "rank": rank,
            "evaluated_count": len(items),
            "selected_count": len(selected),
            "selected_rate": len(selected) / len(items) if items else "",
            "mse_mean": float(np.mean(mse_values)) if mse_values else "",
            "mse_std": float(np.std(mse_values)) if mse_values else "",
            "js_mean": float(np.mean(js_values)) if js_values else "",
            "js_std": float(np.std(js_values)) if js_values else "",
            "top1_change_rate": float(np.mean([
                float(row["top1_unchanged"] is False)
                for row in items
                if row["top1_unchanged"] is not None
            ])) if any(row["top1_unchanged"] is not None for row in items) else "",
            "confidence_mean": float(np.mean(confidence_values)) if confidence_values else "",
            "rejected_by_mse_gate_count": sum(int(row["rejected_by_mse_gate"]) for row in items),
        })
    return rows


def js_divergence(probs_a, probs_b):
    probs_a = probs_a.clamp_min(1e-12)
    probs_b = probs_b.clamp_min(1e-12)
    probs_a = probs_a / probs_a.sum(dim=-1, keepdim=True)
    probs_b = probs_b / probs_b.sum(dim=-1, keepdim=True)
    midpoint = 0.5 * (probs_a + probs_b)
    kl_a = (probs_a * (probs_a / midpoint).log()).sum(dim=-1)
    kl_b = (probs_b * (probs_b / midpoint).log()).sum(dim=-1)
    return 0.5 * (kl_a + kl_b)


def build_per_rank_rows(results, source_indices, labels, ranks):
    rows = []
    for source_type, result in results.items():
        for sample_id in range(labels.numel()):
            for rank_id, rank in enumerate(ranks):
                rows.append({
                    "sample_id": sample_id,
                    "source_index": source_indices[sample_id],
                    "source_type": source_type,
                    "label": int(labels[sample_id].item()),
                    "rank": rank,
                    "rank_spec": f"[{rank},{rank},{rank},{rank}]",
                    "entropy": float(result["entropy"][sample_id, rank_id].item()),
                    "top1": int(result["top1"][sample_id, rank_id].item()),
                    "confidence": float(result["confidence"][sample_id, rank_id].item()),
                    "correct": int(result["correct"][sample_id, rank_id].item()),
                })
    return rows


def build_adjacent_rows(results, source_indices, labels, ranks):
    rows = []
    adjacent_metrics = {}
    for source_type, result in results.items():
        probs_prev = result["probs"][:, :-1, :]
        probs_next = result["probs"][:, 1:, :]
        js_values = js_divergence(probs_prev, probs_next)
        top1_changed = result["top1"][:, :-1].ne(result["top1"][:, 1:])
        entropy_delta = result["entropy"][:, 1:] - result["entropy"][:, :-1]
        confidence_delta = result["confidence"][:, 1:] - result["confidence"][:, :-1]

        adjacent_metrics[source_type] = {
            "js": js_values,
            "top1_changed": top1_changed,
            "entropy_delta": entropy_delta,
            "confidence_delta": confidence_delta,
        }

        for sample_id in range(labels.numel()):
            for rank_id in range(len(ranks) - 1):
                rows.append({
                    "sample_id": sample_id,
                    "source_index": source_indices[sample_id],
                    "source_type": source_type,
                    "label": int(labels[sample_id].item()),
                    "rank_prev": ranks[rank_id],
                    "rank_next": ranks[rank_id + 1],
                    "rank_spec_prev": f"[{ranks[rank_id]},{ranks[rank_id]},{ranks[rank_id]},{ranks[rank_id]}]",
                    "rank_spec_next": f"[{ranks[rank_id + 1]},{ranks[rank_id + 1]},"
                                      f"{ranks[rank_id + 1]},{ranks[rank_id + 1]}]",
                    "js_divergence": float(js_values[sample_id, rank_id].item()),
                    "top1_prev": int(result["top1"][sample_id, rank_id].item()),
                    "top1_next": int(result["top1"][sample_id, rank_id + 1].item()),
                    "top1_changed": int(top1_changed[sample_id, rank_id].item()),
                    "confidence_prev": float(result["confidence"][sample_id, rank_id].item()),
                    "confidence_next": float(result["confidence"][sample_id, rank_id + 1].item()),
                    "confidence_delta": float(confidence_delta[sample_id, rank_id].item()),
                    "entropy_prev": float(result["entropy"][sample_id, rank_id].item()),
                    "entropy_next": float(result["entropy"][sample_id, rank_id + 1].item()),
                    "entropy_delta": float(entropy_delta[sample_id, rank_id].item()),
                })
    return rows, adjacent_metrics


def mean_std(tensor):
    tensor = tensor.float().view(-1)
    if tensor.numel() == 0:
        return None, None
    return float(tensor.mean().item()), float(tensor.std(unbiased=False).item())


def build_summary_rows(results, adjacent_metrics, ranks):
    rows = []
    for source_type, result in results.items():
        for rank_id, rank in enumerate(ranks):
            entropy_mean, entropy_std = mean_std(result["entropy"][:, rank_id])
            confidence_mean, confidence_std = mean_std(result["confidence"][:, rank_id])
            accuracy = float(result["correct"][:, rank_id].float().mean().item())
            rows.append({
                "source_type": source_type,
                "scope": "per_rank",
                "rank": rank,
                "rank_pair": "",
                "count": int(result["entropy"].size(0)),
                "accuracy": accuracy,
                "entropy_mean": entropy_mean,
                "entropy_std": entropy_std,
                "confidence_mean": confidence_mean,
                "confidence_std": confidence_std,
                "js_mean": "",
                "js_std": "",
                "top1_change_rate": "",
                "entropy_delta_mean": "",
                "confidence_delta_mean": "",
            })

        metric = adjacent_metrics[source_type]
        for rank_id in range(len(ranks) - 1):
            js_mean, js_std = mean_std(metric["js"][:, rank_id])
            entropy_delta_mean, _ = mean_std(metric["entropy_delta"][:, rank_id])
            confidence_delta_mean, _ = mean_std(metric["confidence_delta"][:, rank_id])
            rows.append({
                "source_type": source_type,
                "scope": "adjacent_rank",
                "rank": "",
                "rank_pair": f"{ranks[rank_id]}->{ranks[rank_id + 1]}",
                "count": int(metric["js"].size(0)),
                "accuracy": "",
                "entropy_mean": "",
                "entropy_std": "",
                "confidence_mean": "",
                "confidence_std": "",
                "js_mean": js_mean,
                "js_std": js_std,
                "top1_change_rate": float(metric["top1_changed"][:, rank_id].float().mean().item()),
                "entropy_delta_mean": entropy_delta_mean,
                "confidence_delta_mean": confidence_delta_mean,
            })
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def save_figure(fig, path, dpi):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_rank_metric(results, ranks, metric_key, ylabel, title, output_path, dpi, ylim=None):
    """绘制 clean/adv 在不同 rank 下的均值趋势，并用阴影表示样本间标准差。"""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.asarray(ranks, dtype=float)
    colors = {"clean": "#1f77b4", "adv": "#d62728"}

    for source_type, result in results.items():
        values = result[metric_key]
        if values.dtype == torch.bool:
            values = values.float()
        values_np = tensor_to_numpy(values.float())
        mean = values_np.mean(axis=0)
        std = values_np.std(axis=0)
        ax.plot(x, mean, marker="o", linewidth=2, label=source_type, color=colors.get(source_type))
        ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=colors.get(source_type))

    ax.set_title(title)
    ax.set_xlabel("Tensor ring rank")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ranks)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend()
    save_figure(fig, output_path, dpi)


def plot_adjacent_metric(adjacent_metrics, ranks, metric_key, ylabel, title, output_path, dpi, ylim=None):
    """绘制相邻 rank pair 的 JS/top1-change 等稳定性指标。"""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(ranks) - 1)
    x_labels = [f"{ranks[i]}->{ranks[i + 1]}" for i in range(len(ranks) - 1)]
    colors = {"clean": "#1f77b4", "adv": "#d62728"}

    for source_type, metric in adjacent_metrics.items():
        values = metric[metric_key]
        if values.dtype == torch.bool:
            values = values.float()
        values_np = tensor_to_numpy(values.float())
        mean = values_np.mean(axis=0)
        std = values_np.std(axis=0)
        ax.plot(x, mean, marker="o", linewidth=2, label=source_type, color=colors.get(source_type))
        ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=colors.get(source_type))

    ax.set_title(title)
    ax.set_xlabel("Adjacent rank pair")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend()
    save_figure(fig, output_path, dpi)


def plot_heatmap(matrix, x_labels, y_labels, title, colorbar_label, output_path, dpi,
                 cmap="viridis", value_format=None):
    """绘制样本 x rank 的热力图，用于观察单样本预测是否随 rank 突变。"""
    matrix = np.asarray(matrix)
    height = min(max(4.0, 0.28 * matrix.shape[0] + 1.8), 16.0)
    width = min(max(7.0, 0.8 * matrix.shape[1] + 2.5), 14.0)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Sample")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)

    if len(y_labels) <= 40:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=7)
    else:
        tick_count = 20
        ticks = np.linspace(0, len(y_labels) - 1, tick_count, dtype=int)
        ax.set_yticks(ticks)
        ax.set_yticklabels([y_labels[i] for i in ticks], fontsize=7)

    if value_format is not None and matrix.size <= 300:
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, value_format.format(matrix[row, col]),
                        ha="center", va="center", fontsize=6, color="white")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    save_figure(fig, output_path, dpi)


def generate_visualizations(results, adjacent_metrics, ranks, source_indices,
                            output_dir, plot_format, dpi):
    """生成结果图，避免只依赖 CSV 表格判断 rank 稳定性。"""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_rank_metric(
        results,
        ranks,
        metric_key="entropy",
        ylabel="Prediction entropy",
        title="Prediction entropy by tensor-ring rank",
        output_path=plots_dir / f"entropy_by_rank.{plot_format}",
        dpi=dpi,
    )
    plot_rank_metric(
        results,
        ranks,
        metric_key="confidence",
        ylabel="Top-1 confidence",
        title="Top-1 confidence by tensor-ring rank",
        output_path=plots_dir / f"confidence_by_rank.{plot_format}",
        dpi=dpi,
        ylim=(0.0, 1.0),
    )
    plot_rank_metric(
        results,
        ranks,
        metric_key="correct",
        ylabel="Accuracy",
        title="Accuracy by tensor-ring rank",
        output_path=plots_dir / f"accuracy_by_rank.{plot_format}",
        dpi=dpi,
        ylim=(0.0, 1.0),
    )
    plot_adjacent_metric(
        adjacent_metrics,
        ranks,
        metric_key="js",
        ylabel="JS divergence",
        title="Adjacent-rank JS divergence",
        output_path=plots_dir / f"adjacent_rank_js.{plot_format}",
        dpi=dpi,
    )
    plot_adjacent_metric(
        adjacent_metrics,
        ranks,
        metric_key="top1_changed",
        ylabel="Top-1 change rate",
        title="Adjacent-rank top-1 prediction changes",
        output_path=plots_dir / f"adjacent_rank_top1_change_rate.{plot_format}",
        dpi=dpi,
        ylim=(0.0, 1.0),
    )
    plot_adjacent_metric(
        adjacent_metrics,
        ranks,
        metric_key="entropy_delta",
        ylabel="Entropy delta",
        title="Adjacent-rank entropy delta",
        output_path=plots_dir / f"adjacent_rank_entropy_delta.{plot_format}",
        dpi=dpi,
    )

    y_labels = [str(index) for index in source_indices]
    rank_labels = [str(rank) for rank in ranks]
    pair_labels = [f"{ranks[i]}->{ranks[i + 1]}" for i in range(len(ranks) - 1)]

    for source_type, result in results.items():
        plot_heatmap(
            tensor_to_numpy(result["entropy"]),
            rank_labels,
            y_labels,
            title=f"{source_type} entropy heatmap",
            colorbar_label="Entropy",
            output_path=plots_dir / f"{source_type}_entropy_heatmap.{plot_format}",
            dpi=dpi,
            cmap="magma",
        )
        plot_heatmap(
            tensor_to_numpy(result["confidence"]),
            rank_labels,
            y_labels,
            title=f"{source_type} confidence heatmap",
            colorbar_label="Confidence",
            output_path=plots_dir / f"{source_type}_confidence_heatmap.{plot_format}",
            dpi=dpi,
            cmap="viridis",
        )
        plot_heatmap(
            tensor_to_numpy(result["top1"]),
            rank_labels,
            y_labels,
            title=f"{source_type} top-1 class heatmap",
            colorbar_label="Predicted class",
            output_path=plots_dir / f"{source_type}_top1_heatmap.{plot_format}",
            dpi=dpi,
            cmap="tab20",
            value_format="{:.0f}",
        )
        plot_heatmap(
            tensor_to_numpy(adjacent_metrics[source_type]["js"]),
            pair_labels,
            y_labels,
            title=f"{source_type} adjacent-rank JS heatmap",
            colorbar_label="JS divergence",
            output_path=plots_dir / f"{source_type}_adjacent_js_heatmap.{plot_format}",
            dpi=dpi,
            cmap="plasma",
        )


def optional_float(value):
    if value in (None, ""):
        return None
    return float(value)


def optional_bool(value):
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def group_rank_growth_history(history_rows):
    grouped = defaultdict(dict)
    ranks = set()
    source_types = set()
    sample_keys = set()
    for row in history_rows:
        source_type = row["source_type"]
        sample_id = int(row["sample_id"])
        rank = int(row["rank"])
        source_types.add(source_type)
        sample_keys.add((source_type, sample_id))
        ranks.add(rank)
        grouped[(source_type, sample_id)][rank] = row
    source_types = sorted(source_types)
    ranks = sorted(ranks)
    sample_keys = sorted(sample_keys, key=lambda item: (item[0], item[1]))
    return grouped, source_types, ranks, sample_keys


def plot_rank_growth_selected_distribution(history_rows, output_path, dpi):
    selected_rows = [row for row in history_rows if optional_bool(row.get("selected"))]
    _, source_types, ranks, _ = group_rank_growth_history(history_rows)
    counts = {
        source_type: [
            sum(1 for row in selected_rows if row["source_type"] == source_type and int(row["rank"]) == rank)
            for rank in ranks
        ]
        for source_type in source_types
    }

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(ranks))
    width = 0.8 / max(1, len(source_types))
    colors = {"clean": "#1f77b4", "adv": "#d62728"}
    for index, source_type in enumerate(source_types):
        offset = (index - (len(source_types) - 1) / 2) * width
        ax.bar(x + offset, counts[source_type], width=width, label=source_type,
               color=colors.get(source_type))
    ax.set_title("Selected rank distribution")
    ax.set_xlabel("Selected rank")
    ax.set_ylabel("Sample count")
    ax.set_xticks(x)
    ax.set_xticklabels([str(rank) for rank in ranks])
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend()
    save_figure(fig, output_path, dpi)


def plot_rank_growth_summary_metric(summary_rows, metric_key, ylabel, title, output_path, dpi):
    source_types = sorted({row["source_type"] for row in summary_rows})
    ranks = sorted({int(row["rank"]) for row in summary_rows})
    by_source_rank = {
        (row["source_type"], int(row["rank"])): row
        for row in summary_rows
    }

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = {"clean": "#1f77b4", "adv": "#d62728"}
    for source_type in source_types:
        values = []
        for rank in ranks:
            row = by_source_rank.get((source_type, rank), {})
            values.append(optional_float(row.get(metric_key)))
        values_np = np.asarray([np.nan if value is None else value for value in values], dtype=float)
        ax.plot(ranks, values_np, marker="o", linewidth=2, label=source_type,
                color=colors.get(source_type))
    ax.set_title(title)
    ax.set_xlabel("Rank")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ranks)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend()
    save_figure(fig, output_path, dpi)


def plot_rank_growth_selected_mse(history_rows, output_path, dpi):
    source_types = sorted({row["source_type"] for row in history_rows})
    values = []
    labels = []
    for source_type in source_types:
        source_values = [
            float(row["mse_to_input"])
            for row in history_rows
            if row["source_type"] == source_type and optional_bool(row.get("selected"))
        ]
        if source_values:
            values.append(source_values)
            labels.append(source_type)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set_title("Selected reconstruction MSE")
    ax.set_xlabel("Source type")
    ax.set_ylabel("MSE to input")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    save_figure(fig, output_path, dpi)


def plot_rank_growth_heatmap(history_rows, value_key, title, colorbar_label,
                             output_path, dpi, cmap="viridis"):
    grouped, source_types, ranks, _ = group_rank_growth_history(history_rows)
    for source_type in source_types:
        sample_ids = sorted({
            int(row["sample_id"])
            for row in history_rows
            if row["source_type"] == source_type
        })
        matrix = np.full((len(sample_ids), len(ranks)), np.nan, dtype=float)
        for row_index, sample_id in enumerate(sample_ids):
            rank_rows = grouped[(source_type, sample_id)]
            for col_index, rank in enumerate(ranks):
                row = rank_rows.get(rank)
                if row is None:
                    continue
                value = optional_float(row.get(value_key))
                if value is not None:
                    matrix[row_index, col_index] = value

        masked = np.ma.masked_invalid(matrix)
        fig, ax = plt.subplots(figsize=(8.5, min(max(4.5, len(sample_ids) * 0.12 + 1.8), 12.0)))
        image = ax.imshow(masked, aspect="auto", cmap=cmap)
        ax.set_title(f"{source_type} {title}")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Sample")
        ax.set_xticks(np.arange(len(ranks)))
        ax.set_xticklabels([str(rank) for rank in ranks])
        if len(sample_ids) <= 40:
            ax.set_yticks(np.arange(len(sample_ids)))
            ax.set_yticklabels([str(sample_id) for sample_id in sample_ids], fontsize=7)
        else:
            tick_count = 16
            ticks = np.linspace(0, len(sample_ids) - 1, tick_count, dtype=int)
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(sample_ids[i]) for i in ticks], fontsize=7)
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(colorbar_label)
        source_output_path = output_path.with_name(
            f"{source_type}_{output_path.name}"
        )
        save_figure(fig, source_output_path, dpi)


def generate_rank_growth_visualizations(history_rows, summary_rows, output_dir, plot_format, dpi):
    """基于 rank_growth CSV 生成调参图，不需要重新运行 TN 优化。"""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_rank_growth_selected_distribution(
        history_rows,
        plots_dir / f"selected_rank_distribution.{plot_format}",
        dpi,
    )
    plot_rank_growth_summary_metric(
        summary_rows,
        metric_key="mse_mean",
        ylabel="MSE to input",
        title="Rank-growth MSE by rank",
        output_path=plots_dir / f"mse_by_rank.{plot_format}",
        dpi=dpi,
    )
    plot_rank_growth_summary_metric(
        summary_rows,
        metric_key="js_mean",
        ylabel="JS divergence to previous rank",
        title="Rank-growth JS by rank",
        output_path=plots_dir / f"js_by_rank.{plot_format}",
        dpi=dpi,
    )
    plot_rank_growth_summary_metric(
        summary_rows,
        metric_key="rejected_by_mse_gate_count",
        ylabel="Rejected block count",
        title="MSE-gate rejections by rank",
        output_path=plots_dir / f"mse_gate_rejections_by_rank.{plot_format}",
        dpi=dpi,
    )
    plot_rank_growth_selected_mse(
        history_rows,
        plots_dir / f"selected_mse_boxplot.{plot_format}",
        dpi,
    )
    plot_rank_growth_heatmap(
        history_rows,
        value_key="mse_to_input",
        title="MSE trajectory",
        colorbar_label="MSE to input",
        output_path=plots_dir / f"mse_trajectory_heatmap.{plot_format}",
        dpi=dpi,
        cmap="magma",
    )
    plot_rank_growth_heatmap(
        history_rows,
        value_key="js_to_prev",
        title="JS trajectory",
        colorbar_label="JS divergence",
        output_path=plots_dir / f"js_trajectory_heatmap.{plot_format}",
        dpi=dpi,
        cmap="plasma",
    )


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def prepare_output_dir(output_dir, overwrite, analysis_mode):
    output_dir = Path(output_dir)
    if analysis_mode == "rank_growth":
        target_files = [
            "rank_growth_predictions.pt",
            "rank_growth_history.csv",
            "rank_growth_summary.csv",
            "meta.json",
        ]
    else:
        target_files = [
            "rank_predictions.pt",
            "per_rank_metrics.csv",
            "adjacent_rank_metrics.csv",
            "summary.csv",
            "meta.json",
        ]
    existing = [output_dir / name for name in target_files if (output_dir / name).exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist: {joined}. Use --overwrite to replace them.")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    args = parse_args()
    if args.plot_dpi <= 0:
        raise ValueError("--plot_dpi must be positive.")
    if args.analysis_mode == "tensorly_tr" and len(args.ranks) < 2:
        raise ValueError("--ranks must contain at least two ranks to compute adjacent-rank metrics.")

    seed_everything(args.seed)
    tl.set_backend("pytorch")

    if args.plot_only:
        if args.analysis_mode != "rank_growth":
            raise ValueError("--plot_only currently supports --analysis_mode rank_growth only.")
        output_dir = Path(args.output_dir)
        history_path = output_dir / "rank_growth_history.csv"
        summary_path = output_dir / "rank_growth_summary.csv"
        if not history_path.exists() or not summary_path.exists():
            raise FileNotFoundError(
                "Rank-growth plot-only mode requires existing rank_growth_history.csv "
                f"and rank_growth_summary.csv under {output_dir}."
            )
        history_rows = read_csv_rows(history_path)
        summary_rows = read_csv_rows(summary_path)
        generate_rank_growth_visualizations(
            history_rows=history_rows,
            summary_rows=summary_rows,
            output_dir=output_dir,
            plot_format=args.plot_format,
            dpi=args.plot_dpi,
        )
        print(f"Saved rank-growth plots to: {output_dir / 'plots'}")
        return

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    config_name = normalize_config_name(args.config)
    config, config_path = load_config(args.dataset, config_name)
    strategy = config.get("strategy", "3d_interpolate")
    purify_args = SimpleNamespace(dataset=args.dataset, config=config_name)
    resolved_consistency_tag = build_consistency_tag(
        consistency_version=args.consistency_version,
        consistency_rank_tag=args.consistency_rank_tag,
        pair_sample_num=args.pair_sample_num,
        eps=args.eps,
        consistency_tag=args.consistency_tag,
    )
    resolved_pair_kind = build_pair_kind(args.consistency_version, args.pair_kind)
    resolved_adv_model_tag = build_adv_model_tag(
        consistency_version=args.consistency_version,
        consistency_rank_tag=args.consistency_rank_tag,
        pair_sample_num=args.pair_sample_num,
        eps=args.eps,
        consistency_tag=args.consistency_tag,
        adv_model_tag=args.adv_model_tag,
    )
    pair_path, ad_data_path, checkpoint_path = resolve_input_paths(
        args,
        config_name=config_name,
        adv_model_tag=resolved_adv_model_tag,
    )

    output_dir = prepare_output_dir(args.output_dir, args.overwrite, args.analysis_mode)

    dataset, info = load_thubenchmark()
    if args.data_source == "pair" or pair_path:
        data_payload = load_pair_payload(pair_path)
    else:
        data_payload = load_test_ad_payload(args, dataset, info, ad_data_path)

    selected_positions = select_samples(data_payload["labels"].numel(), args.sample_num, args.seed)
    selected_source_indices = [data_payload["source_indices"][idx] for idx in selected_positions.tolist()]
    labels = data_payload["labels"][selected_positions]
    clean_samples = data_payload["x"][selected_positions]
    adv_samples = data_payload["x_adv"][selected_positions]

    sampling_rate = info["sampling_rate"]
    model = load_classifier(checkpoint_path, info, device)

    probe_pre_data = interpolate(purify_args, clean_samples[0], sampling_rate).detach().cpu().float()
    if args.analysis_mode == "rank_growth":
        if config.get("model") != "PTR_3d_rank_growth":
            raise ValueError(
                "--analysis_mode rank_growth requires a PTR_3d_rank_growth config, "
                f"got model={config.get('model')} from {config_path}."
            )

        rank_growth_config = build_rank_growth_config(config, args)
        history_rows = []
        traces = {}
        for source_type, samples in (("clean", clean_samples), ("adv", adv_samples)):
            source_rows, source_traces = predict_rank_growth_for_source(
                source_type=source_type,
                samples=samples,
                labels=labels,
                source_indices=selected_source_indices,
                model=model,
                purify_args=purify_args,
                config=rank_growth_config,
                strategy=strategy,
                sampling_rate=sampling_rate,
                device=device,
            )
            history_rows.extend(source_rows)
            traces[source_type] = source_traces

        summary_rows = build_rank_growth_summary_rows(history_rows)
        meta = {
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "analysis_mode": args.analysis_mode,
            "dataset": args.dataset,
            "model": "eegnet",
            "eps": args.eps,
            "fold": args.fold,
            "pair_sample_num": args.pair_sample_num,
            "consistency_version": args.consistency_version,
            "consistency_rank_tag": args.consistency_rank_tag,
            "consistency_tag": resolved_consistency_tag,
            "pair_kind": resolved_pair_kind,
            "data_source": "pair" if pair_path else "ad_data",
            "checkpoint_lr": args.checkpoint_lr,
            "checkpoint_weight_decay": args.checkpoint_weight_decay,
            "pair_path": pair_path,
            "ad_data_path": ad_data_path,
            "ad_data_dir": args.ad_data_dir,
            "adv_model_tag": resolved_adv_model_tag,
            "attack": args.attack,
            "at_strategy": args.at_strategy,
            "use_ea": args.use_ea,
            "checkpoint_path": checkpoint_path,
            "config": config_name,
            "config_path": str(config_path),
            "strategy": strategy,
            "sample_num": args.sample_num,
            "rank_growth_ranks": rank_growth_config.get("rank_growth_ranks"),
            "rank_growth_steps_per_rank": rank_growth_config.get("rank_growth_steps_per_rank"),
            "rank_growth_js_threshold": rank_growth_config.get("rank_growth_js_threshold"),
            "rank_growth_max_mse_to_input": rank_growth_config.get("rank_growth_max_mse_to_input"),
            "rank_growth_lr_decay_factor": rank_growth_config.get("rank_growth_lr_decay_factor"),
            "pre_data_shape": tuple(probe_pre_data.shape),
            "input_shape": tuple(clean_samples.shape[1:]),
            "selected_positions": selected_positions.tolist(),
            "source_indices": selected_source_indices,
            "sampling_rate": sampling_rate,
            "device": str(device),
            "visualize": not args.no_visualize,
            "input_meta": data_payload["meta"],
            "pair_meta": data_payload["meta"],
            "outputs": {
                "predictions": "rank_growth_predictions.pt",
                "history": "rank_growth_history.csv",
                "summary": "rank_growth_summary.csv",
                "meta": "meta.json",
            },
        }

        torch.save(
            {
                "selected_positions": selected_positions,
                "source_indices": selected_source_indices,
                "labels": labels,
                "history_rows": history_rows,
                "traces": traces,
                "meta": meta,
            },
            output_dir / "rank_growth_predictions.pt",
        )
        write_csv(
            output_dir / "rank_growth_history.csv",
            history_rows,
            [
                "sample_id",
                "source_index",
                "source_type",
                "label",
                "rank",
                "mse_to_input",
                "js_to_prev",
                "top1",
                "confidence",
                "correct",
                "mse_rel_delta_to_prev",
                "top1_unchanged",
                "fidelity_gate_pass",
                "rank_growth_max_mse_to_input",
                "rejected_by_mse_gate",
                "loss",
                "selected",
            ],
        )
        write_csv(
            output_dir / "rank_growth_summary.csv",
            summary_rows,
            [
                "source_type",
                "rank",
                "evaluated_count",
                "selected_count",
                "selected_rate",
                "mse_mean",
                "mse_std",
                "js_mean",
                "js_std",
                "top1_change_rate",
                "confidence_mean",
                "rejected_by_mse_gate_count",
            ],
        )
        with open(output_dir / "meta.json", "w", encoding="utf-8") as file:
            json.dump(to_jsonable(meta), file, ensure_ascii=False, indent=2)

        if not args.no_visualize:
            generate_rank_growth_visualizations(
                history_rows=history_rows,
                summary_rows=summary_rows,
                output_dir=output_dir,
                plot_format=args.plot_format,
                dpi=args.plot_dpi,
            )

        print(f"Saved rank-growth analysis results to: {output_dir}")
        return

    validate_uniform_tr_ranks(probe_pre_data.shape, args.ranks, args.tr_mode)

    results = {}
    for source_type, samples in (("clean", clean_samples), ("adv", adv_samples)):
        results[source_type] = predict_for_source(
            source_type=source_type,
            samples=samples,
            labels=labels,
            ranks=args.ranks,
            model=model,
            purify_args=purify_args,
            strategy=strategy,
            sampling_rate=sampling_rate,
            tr_mode=args.tr_mode,
            batch_size=args.batch_size,
            device=device,
        )

    per_rank_rows = build_per_rank_rows(results, selected_source_indices, labels, args.ranks)
    adjacent_rows, adjacent_metrics = build_adjacent_rows(results, selected_source_indices, labels, args.ranks)
    summary_rows = build_summary_rows(results, adjacent_metrics, args.ranks)

    meta = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "analysis_mode": args.analysis_mode,
        "dataset": args.dataset,
        "model": "eegnet",
        "eps": args.eps,
        "fold": args.fold,
        "pair_sample_num": args.pair_sample_num,
        "consistency_version": args.consistency_version,
        "consistency_rank_tag": args.consistency_rank_tag,
        "consistency_tag": resolved_consistency_tag,
        "pair_kind": resolved_pair_kind,
        "data_source": "pair" if pair_path else "ad_data",
        "checkpoint_lr": args.checkpoint_lr,
        "checkpoint_weight_decay": args.checkpoint_weight_decay,
        "pair_path": pair_path,
        "ad_data_path": ad_data_path,
        "ad_data_dir": args.ad_data_dir,
        "adv_model_tag": resolved_adv_model_tag,
        "attack": args.attack,
        "at_strategy": args.at_strategy,
        "use_ea": args.use_ea,
        "checkpoint_path": checkpoint_path,
        "config": config_name,
        "config_path": str(config_path),
        "strategy": strategy,
        "sample_num": args.sample_num,
        "ranks": args.ranks,
        "rank_specs": [[rank] * (probe_pre_data.dim() + 1) for rank in args.ranks],
        "tr_mode": args.tr_mode,
        "pre_data_shape": tuple(probe_pre_data.shape),
        "input_shape": tuple(clean_samples.shape[1:]),
        "selected_positions": selected_positions.tolist(),
        "source_indices": selected_source_indices,
        "sampling_rate": sampling_rate,
        "device": str(device),
        "visualize": not args.no_visualize,
        "plot_format": args.plot_format,
        "plot_dpi": args.plot_dpi,
        "input_meta": data_payload["meta"],
        "pair_meta": data_payload["meta"],
    }

    torch.save(
        {
            "ranks": args.ranks,
            "rank_specs": meta["rank_specs"],
            "selected_positions": selected_positions,
            "source_indices": selected_source_indices,
            "labels": labels,
            "logits": {key: value["logits"] for key, value in results.items()},
            "probs": {key: value["probs"] for key, value in results.items()},
            "entropy": {key: value["entropy"] for key, value in results.items()},
            "confidence": {key: value["confidence"] for key, value in results.items()},
            "top1": {key: value["top1"] for key, value in results.items()},
            "correct": {key: value["correct"] for key, value in results.items()},
            "adjacent_metrics": adjacent_metrics,
            "meta": meta,
        },
        output_dir / "rank_predictions.pt",
    )

    write_csv(
        output_dir / "per_rank_metrics.csv",
        per_rank_rows,
        [
            "sample_id",
            "source_index",
            "source_type",
            "label",
            "rank",
            "rank_spec",
            "entropy",
            "top1",
            "confidence",
            "correct",
        ],
    )
    write_csv(
        output_dir / "adjacent_rank_metrics.csv",
        adjacent_rows,
        [
            "sample_id",
            "source_index",
            "source_type",
            "label",
            "rank_prev",
            "rank_next",
            "rank_spec_prev",
            "rank_spec_next",
            "js_divergence",
            "top1_prev",
            "top1_next",
            "top1_changed",
            "confidence_prev",
            "confidence_next",
            "confidence_delta",
            "entropy_prev",
            "entropy_next",
            "entropy_delta",
        ],
    )
    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        [
            "source_type",
            "scope",
            "rank",
            "rank_pair",
            "count",
            "accuracy",
            "entropy_mean",
            "entropy_std",
            "confidence_mean",
            "confidence_std",
            "js_mean",
            "js_std",
            "top1_change_rate",
            "entropy_delta_mean",
            "confidence_delta_mean",
        ],
    )
    with open(output_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(to_jsonable(meta), file, ensure_ascii=False, indent=2)

    if not args.no_visualize:
        generate_visualizations(
            results=results,
            adjacent_metrics=adjacent_metrics,
            ranks=args.ranks,
            source_indices=selected_source_indices,
            output_dir=output_dir,
            plot_format=args.plot_format,
            dpi=args.plot_dpi,
        )

    print(f"Saved Tensor Ring rank analysis results to: {output_dir}")


if __name__ == "__main__":
    main()
