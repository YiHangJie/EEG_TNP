"""EXP-032 的只读产物加载、统一子集与可审计统计。"""

import hashlib
import json
from pathlib import Path

from runtime_env import configure_runtime_env

configure_runtime_env()

import torch

from rpcf.core import DATASET_LOADERS, load_model_checkpoint, seed_everything
from rpcf.exp031 import checkpoint
from utils.reproducibility import stable_subset_indices


SOURCE_RUN = "exp031_full_20260729_174215"


def load_payload(path):
    """大张量按需映射；新攻击仅持久化 adv，clean 从既有同源产物恢复。"""
    try:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except RuntimeError as error:
        if "mmap" not in str(error):
            raise
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "clean_reference_path" in payload and "clean" not in payload:
        reference = torch.load(payload["clean_reference_path"], map_location="cpu", mmap=True, weights_only=False)
        indices = [int(i) for i in payload["source_indices"]]
        lookup = {int(value): i for i, value in enumerate(reference["source_indices"])}
        positions = torch.tensor([lookup[i] for i in indices])
        if not torch.equal(reference["labels"].index_select(0, positions).long(), payload["labels"].long()):
            raise ValueError("Referenced clean labels mismatch")
        payload["clean"] = reference["clean"].index_select(0, positions).float()
    return payload


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint_path(args, method):
    run = args.run_id if method == "clean" else args.source_run
    path = checkpoint(run, args.dataset, args.model, args.seed, method)
    return path.replace("_clean_eps0.03_", "_clean_eps0_") if method == "clean" else path


def align_payload(payload, indices, labels=None, clean=None):
    source = [int(i) for i in payload["source_indices"]]
    if len(set(source)) != len(source):
        raise ValueError("Duplicate source indices")
    for key in ("clean", "adversarial", "labels", "clean_pur_by_rank", "adv_pur_by_rank"):
        if key in payload and len(payload[key]) != len(source):
            raise ValueError(f"{key} length does not match source indices")
    if "clean_pur_by_rank" in payload:
        ranks = list(payload["ranks"])
        if len(ranks) != len(set(ranks)) or payload["clean_pur_by_rank"].shape[1] != len(ranks):
            raise ValueError("Invalid rank axis")
    positions = {value: index for index, value in enumerate(source)}
    missing = set(indices) - positions.keys()
    if missing:
        raise ValueError(f"Missing canonical samples: {sorted(missing)[:5]}")
    order = torch.tensor([positions[i] for i in indices], dtype=torch.long)
    result = {key: payload[key].index_select(0, order) for key in (
        "clean", "adversarial", "labels", "clean_pur_by_rank", "adv_pur_by_rank", "subject_indices"
    ) if key in payload}
    if labels is not None and not torch.equal(result["labels"].long(), labels.long()):
        raise ValueError("Canonical labels mismatch")
    if clean is not None and not torch.equal(result["clean"].float(), clean.float()):
        raise ValueError("Canonical clean tensor mismatch")
    return result


def load_condition(args):
    """实际 test split 决定规范样本身份，复用既有随机库和种子规则。"""
    from data.subject_ea import prepare_subject_fold
    from utils.experiment_artifacts import eeg_classification_collate

    seed_everything(args.seed)
    data, info = DATASET_LOADERS[args.dataset]()
    train, val, test, split = prepare_subject_fold(
        args.dataset, data, info, fold_id=0, seed=args.seed, use_ea=False
    )
    indices, selection_seed = stable_subset_indices(len(test), min(args.sample_num, len(test)), args.seed, 0)
    clean, labels = eeg_classification_collate([test[i] for i in indices])
    return data, info, train, val, clean.float(), labels.long(), indices, {
        "split_path": split, "full_test_size": len(test), "selection_seed": selection_seed,
        "selection_seed_rule": "seed + fold * 1000", "fold": 0,
    }


def load_classifier(args, info, method, data=None):
    path = checkpoint_path(args, method)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if method == "ea_forward":
        from data.subject_ea import prepare_subject_ea_forward_fold
        from models.eegnet_ea_forward import build_subject_ea_model
        from attack_ea_forward import SubjectBatchModelWrapper

        _, _, test, _, matrices, _ = prepare_subject_ea_forward_fold(
            args.dataset, data, info, fold_id=0, seed=args.seed
        )
        inner = build_subject_ea_model(f"{args.model}_ea_forward", args.dataset, info, matrices)
        state = load_payload(path)
        inner.load_state_dict(state.get("state_dict", state))
        model = SubjectBatchModelWrapper(inner).to(device).eval()
        return model, test, path
    model = load_model_checkpoint(args.model, args.dataset, info, path, device).eval()
    return model, None, path


def predict(model, inputs, batch_size=32, subject_ids=None):
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            if subject_ids is not None:
                model.set_subject_ids(subject_ids[start:start + batch_size].to(device))
            predictions.extend(model(inputs[start:start + batch_size].to(device)).argmax(1).cpu().tolist())
    return predictions


def metric_row(args, method, attack, clean_predictions, adv_predictions, labels, **extra):
    target = labels.tolist()
    n = len(target)
    return {
        "dataset": args.dataset, "model": args.model, "seed": args.seed, "fold": 0,
        "method": method, "attack": attack, "sample_num": n, "scope": "S",
        "standard_accuracy": sum(a == b for a, b in zip(clean_predictions, target)) / n,
        "robust_accuracy": sum(a == b for a, b in zip(adv_predictions, target)) / n,
        "clean_predictions": clean_predictions, "adv_predictions": adv_predictions,
        **extra,
    }


def state_changes(before, model):
    """检测 eval forward 的权重约束等副作用，不能仅假设 eval 不改变参数。"""
    return {key: float((value.cpu().double() - before[key].double()).abs().max())
            for key, value in model.state_dict().items()
            if not torch.equal(value.cpu(), before[key])}


def norm_audit(clean, adversarial, norm=None, eps=None):
    delta = (adversarial - clean).flatten(1)
    if not torch.isfinite(delta).all():
        raise ValueError("Non-finite perturbation")
    l2, linf = delta.norm(p=2, dim=1), delta.abs().amax(1)
    result = {"dimension": delta.shape[1], "l2_mean": l2.mean().item(), "l2_max": l2.max().item(),
              "linf_mean": linf.mean().item(), "linf_max": linf.max().item()}
    if norm in {"L2", "Linf"} and eps is not None:
        observed = l2 if norm == "L2" else linf
        result["budget_violations"] = int((observed > eps + max(1e-5, eps * 1e-4)).sum())
        if result["budget_violations"]:
            raise ValueError(f"Attack exceeded budget: {result}")
    return result
