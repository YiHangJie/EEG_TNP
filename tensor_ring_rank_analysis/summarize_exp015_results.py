import argparse
import csv
import glob
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_artifacts import safe_token, torch_load_cpu


CONFIG_STEM = "PTR3d_rank_growth_js_mse_exp015_8_2048_r5-40_3d_interpolate"


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize EXP-015 matrix artifacts.")
    parser.add_argument("--log_root", required=True, help="EXP-015 run directory under logs/exp015/.")
    parser.add_argument("--output_csv", default=None, help="summary CSV path; defaults to <log_root>/summary.csv.")
    return parser.parse_args()


def eps_tag(eps):
    return str(eps).replace(".", "p")


def read_tasks(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_meta(path):
    payload = torch_load_cpu(path)
    if isinstance(payload, (tuple, list)) and len(payload) == 3:
        return payload[2] if isinstance(payload[2], dict) else {}
    if isinstance(payload, dict):
        return payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}
    return {}


def newest_match(pattern):
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=lambda item: os.path.getmtime(item))


def main_artifact(task):
    run_tag = f"consistancy_rank25-30_n*_eps{eps_tag(task['eps'])}"
    pattern = (
        "purified_data/attacked/"
        f"{task['dataset']}_{task['model']}_no_ea_{run_tag}_autoattack_eps{task['eps']}_"
        f"seed{task['seed']}_fold{task['fold']}_{CONFIG_STEM}_n*_ad.pth"
    )
    return newest_match(pattern)


def baseline_artifact(task):
    if task["method"] == "abat":
        pattern = (
            "ad_data/"
            f"{task['dataset']}_{task['model']}_no_ea_ea_forward_madry_autoattack_eps{task['eps']}_"
            f"seed{task['seed']}_fold{task['fold']}.pth"
        )
    else:
        pattern = (
            "ad_data/"
            f"{task['dataset']}_{task['model']}_no_ea_{task['method']}_autoattack_eps{task['eps']}_"
            f"seed{task['seed']}_fold{task['fold']}.pth"
        )
    return newest_match(pattern)


def clean_artifact(task):
    pattern = (
        "checkpoints/"
        f"{task['dataset']}_{task['model']}_train_only_subject_no_ea_subject_split_clean_eps0_"
        f"{task['seed']}_fold{task['fold']}_best.pth"
    )
    return newest_match(pattern)


def log_status(log_path):
    path = Path(log_path)
    if not path.exists():
        return "missing_log"
    text = path.read_text(errors="replace")
    if "Traceback" in text or "RuntimeError" in text or "failed" in text.lower():
        return "failed"
    if "finished" in text or "runner complete" in text:
        return "finished"
    return "running_or_incomplete"


def summarize_task(task):
    method = task["method"]
    artifact = None
    if method == "main_js_mse":
        artifact = main_artifact(task)
    elif method == "clean":
        artifact = clean_artifact(task)
    else:
        artifact = baseline_artifact(task)

    meta = {}
    artifact_status = "missing_artifact"
    if artifact:
        artifact_status = "artifact_found"
        if method != "clean":
            meta = load_meta(artifact)

    status = "success" if artifact else log_status(task["log_path"])
    row = dict(task)
    row.update({
        "status": status,
        "artifact_status": artifact_status,
        "artifact_path": artifact or "",
        "sample_num": meta.get("sample_num", ""),
        "requested_sample_num": meta.get("requested_sample_num", ""),
        "clean_acc": meta.get("clean_accuracy", ""),
        "adv_acc": meta.get("ad_accuracy", meta.get("adv_accuracy", "")),
        "purified_clean_acc": meta.get("purified_clean_accuracy", ""),
        "purified_adv_acc": meta.get("purified_ad_accuracy", ""),
        "mse": meta.get("mse", meta.get("mean_ad_mse", "")),
        "model_tag": safe_token(meta.get("model_tag", "")) if meta else "",
    })
    return row


def main():
    args = parse_args()
    log_root = Path(args.log_root)
    task_path = log_root / "planned_tasks.csv"
    if not task_path.exists():
        raise FileNotFoundError(f"planned_tasks.csv not found: {task_path}")
    rows = [summarize_task(task) for task in read_tasks(task_path)]
    output_csv = Path(args.output_csv) if args.output_csv else log_root / "summary.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phase", "method", "dataset", "model", "seed", "fold", "eps", "status",
        "artifact_status", "clean_acc", "adv_acc", "purified_clean_acc",
        "purified_adv_acc", "mse", "sample_num", "requested_sample_num",
        "model_tag", "artifact_path", "log_path",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(output_csv)


if __name__ == "__main__":
    main()
