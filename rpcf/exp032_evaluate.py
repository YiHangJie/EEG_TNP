"""EXP-032 子集重评估、新攻击与外部净化评估入口。"""

import argparse
import gc
import json
from pathlib import Path

from rpcf.exp032_common import (
    SOURCE_RUN, align_payload, checkpoint_path, load_classifier, load_condition,
    load_payload, metric_row, norm_audit, predict, sha256, state_changes, write_json,
)
import torch
from rpcf.core import build_attack, seed_everything
from rpcf.exp031 import RAW_METHODS, ATTACKS, attack_path, tnp_path
from rpcf.exp031_artifacts import atomic_torch_save
from rpcf.exp032_attacks import pgd_l2


def subjects_from_test(test, indices):
    return torch.tensor([int(test[i][2]) for i in indices], dtype=torch.long)


def ea_inputs(test, indices, labels):
    """EA-forward 使用 EA 前的缓存值，不套普通分支的二次标准化。"""
    from utils.experiment_artifacts import eeg_subject_classification_collate
    x, y, subjects = eeg_subject_classification_collate([test[i] for i in indices])
    if not torch.equal(y.long(), labels.long()):
        raise ValueError("EA-forward labels mismatch")
    return x.float(), subjects.long()


def validate_meta(payload, args, method, attack):
    meta = payload["meta"]
    for key, expected in {"dataset": args.dataset, "seed": args.seed, "fold": 0, "attack": attack}.items():
        if meta.get(key) != expected:
            raise ValueError(f"{key}: {meta.get(key)} != {expected}")
    if Path(meta["checkpoint_path"]).resolve() != Path(checkpoint_path(args, method)).resolve():
        raise ValueError("Checkpoint provenance mismatch")


def audit(args):
    data, info, _, _, clean, labels, indices, split = load_condition(args)
    rows, checks = [], []
    for method in RAW_METHODS:
        model, ea_test, path = load_classifier(args, info, method, data)
        method_clean, subjects = ea_inputs(ea_test, indices, labels) if ea_test is not None else (clean, None)
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        cp = predict(model, method_clean, args.batch_size, subjects)
        cp_repeat = predict(model, method_clean, args.batch_size, subjects)
        if cp != cp_repeat:
            raise ValueError(f"Repeated clean inference differs: {method}")
        changes = state_changes(initial, model)
        archive_clean = {}
        for attack in ATTACKS:
            source = attack_path(args.source_run, args.dataset, args.model, args.seed, method, attack)
            payload = load_payload(source)
            validate_meta(payload, args, method, attack)
            aligned = align_payload(payload, indices, labels, method_clean)
            protocol = payload["meta"]["attack_protocol"]
            norms = norm_audit(method_clean, aligned["adversarial"], protocol["norm"],
                               protocol.get("eps") if attack != "cw" else None)
            ap = predict(model, aligned["adversarial"], args.batch_size, subjects)
            rows.append(metric_row(args, method, attack, cp, ap, labels,
                                   attack_protocol=protocol, norm_audit=norms, source_path=source,
                                   evaluation="classifier_whitebox", checkpoint_sha256=sha256(path)))
            archive_clean[attack] = payload["meta"]["clean_accuracy"]
            del aligned, payload
            if method in {"madry", "rpcf_at"}:
                source_tnp = tnp_path(args.source_run, args.dataset, args.model, args.seed, method, attack)
                purified = load_payload(source_tnp)
                aligned_pur = align_payload(purified, indices, labels, clean)
                for position, rank in enumerate(purified["ranks"]):
                    cp_pur = predict(model, aligned_pur["clean_pur_by_rank"][:, position], args.batch_size)
                    ap_pur = predict(model, aligned_pur["adv_pur_by_rank"][:, position], args.batch_size)
                    rows.append(metric_row(args, f"{method}_tnp_r{rank}", attack, cp_pur, ap_pur, labels,
                                           attack_protocol=protocol, source_path=source_tnp,
                                           evaluation="nonadaptive_purification", rank=int(rank)))
                del purified, aligned_pur
            print(f"AUDIT {method} {attack} n={len(indices)}", flush=True)
        status_kind = "train_ea" if method == "ea_forward" else "rpcf_at" if method == "rpcf_at" else "train_standard"
        status_path = Path(f"logs/exp031/{args.source_run}/status/{status_kind}_{args.dataset}_{args.model}_seed{args.seed}_{method}.json")
        training_status = json.loads(status_path.read_text()) if status_path.exists() else {}
        training_command = training_status.get("command", [])
        training_options = {}
        for option in ("--epochs", "--patience", "--batch_size", "--lr", "--weight_decay",
                       "--pgd_steps", "--pgd_step_size", "--trades_beta", "--fbf_replays",
                       "--all_layers", "--static_rank_weights", "--feature_objective"):
            if option in training_command:
                position = training_command.index(option)
                value = training_command[position + 1] if position + 1 < len(training_command) else True
                training_options[option] = True if str(value).startswith("--") else value
        checks.append({"method": method, "training_options": training_options,
                       "actual_batch_size": training_status.get("actual_batch_size"), "checkpoint_path": path, "checkpoint_sha256": sha256(path),
                       "state_changes_after_clean": changes, "state_changes_after_all": state_changes(initial, model),
                       "input_difference_from_standard_max": (method_clean - clean).abs().max().item(),
                       "archive_full_clean_by_attack": archive_clean,
                       "archive_full_clean_span": max(archive_clean.values()) - min(archive_clean.values()),
                       "training_status_path": str(status_path), "training_status_exists": status_path.exists()})
        del model, initial
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output_path, {"experiment_id": "EXP-032", "kind": "audit", "source_indices": indices,
                                "labels": labels.tolist(), "split": split, "checks": checks, "rows": rows})


def generate(args):
    data, info, _, _, clean, labels, indices, split = load_condition(args)
    model, ea_test, path = load_classifier(args, info, args.method, data)
    clean, subjects = ea_inputs(ea_test, indices, labels) if ea_test is not None else (clean, None)
    seed_everything(args.seed)
    cp = predict(model, clean, args.batch_size, subjects)
    device = next(model.parameters()).device
    parts = []
    if args.method == "ea_forward" and args.attack != "pgd_l2":
        raise ValueError("EXP-032 only adds full-batch PGD-L2 for EA-forward; use subject-aware legacy attacks")
    if args.attack != "pgd_l2":
        attack = build_attack(args.attack, model, 0.03, info, device, args.seed)
    for start in range(0, len(clean), args.batch_size):
        x, y = clean[start:start + args.batch_size].to(device), labels[start:start + args.batch_size].to(device)
        if subjects is not None:
            model.set_subject_ids(subjects[start:start + args.batch_size].to(device))
        model.zero_grad(set_to_none=True)
        if args.attack == "pgd_l2":
            adv = pgd_l2(model, x, y, eps=1.0, alpha=0.1, steps=args.l2_steps, restarts=args.l2_restarts)
        else:
            adv = attack(x, y)
        parts.append(adv.detach().cpu())
        print(f"ATTACK {args.method}/{args.attack} {start + len(x)}/{len(clean)}", flush=True)
    adv = torch.cat(parts)
    norm = "L2" if args.attack in {"pgd_l2", "cw"} else "Linf"
    budget = None if args.attack == "cw" else 1.0 if args.attack == "pgd_l2" else 0.03
    norms = norm_audit(clean, adv, norm, budget)
    ap = predict(model, adv, args.batch_size, subjects)
    protocol = {"norm": norm, "eps": budget, "coordinate_system": "standardized_full_EEG_trial"}
    if args.attack == "pgd_l2":
        protocol.update(steps=args.l2_steps, alpha=0.1, restarts=args.l2_restarts, random_start=True,
                        candidate_selection="success_then_cross_entropy_all_iterates", input_clamp=False)
    else:
        protocol.update({
            "fgsm": {"steps": 1}, "pgd": {"steps": 200, "alpha": 2 / 255, "random_start": False},
            "autoattack": {"version": "standard"},
            "cw": {"steps": 200, "lr": 0.1, "c": 10000, "kappa": 1, "eps_is_constraint": False},
        }[args.attack])
    payload = {"clean": clean, "adversarial": adv, "labels": labels, "source_indices": indices,
               "meta": {"kind": "rpcf_attack_eval", "experiment_id": "EXP-032", "dataset": args.dataset,
                        "model": args.model, "method_tag": args.method, "fold": 0, "seed": args.seed,
                        "eps": budget, "classifier_train_eps": 0.03, "attack": args.attack, "attack_protocol": protocol,
                        "checkpoint_path": path, "checkpoint_sha256": sha256(path),
                        "protocol": "train_only_subject_no_ea_subject_split", "source_split": "test",
                        "sample_num": len(indices), "selection_strategy": "random_without_replacement",
                        "clean_accuracy": sum(a == b for a, b in zip(cp, labels.tolist())) / len(labels),
                        "adv_accuracy": sum(a == b for a, b in zip(ap, labels.tolist())) / len(labels),
                        "norm_audit": norms, **split}}
    if subjects is not None:
        payload["subject_indices"] = subjects
    reference_path = attack_path(args.source_run, args.dataset, "eegnet", args.seed,
                                 "ea_forward" if args.method == "ea_forward" else "madry", "autoattack")
    reference = load_payload(reference_path)
    align_payload(reference, indices, labels, clean)
    del reference
    payload.pop("clean")
    payload["clean_reference_path"] = reference_path
    payload["meta"]["storage_policy"] = "adv_float32_plus_existing_clean_reference"
    import shutil
    free = shutil.disk_usage(Path(args.output_path).parent if Path(args.output_path).parent.exists() else Path(".")).free
    if free < adv.numel() * adv.element_size() + 32 * 1024**3:
        raise RuntimeError("Insufficient disk space: reserve 32GiB and preserve prior artifacts")
    atomic_torch_save(payload, args.output_path)
    write_json(str(args.output_path) + ".metrics.json", {
        "source_indices": indices, "labels": labels.tolist(), "rows": [metric_row(
            args, args.method, args.attack, cp, ap, labels, attack_protocol=protocol,
            evaluation="classifier_whitebox", norm_audit=norms)]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "attack"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run", default=SOURCE_RUN)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--method", default="clean")
    parser.add_argument("--attack", default="pgd_l2", choices=[*ATTACKS, "pgd_l2"])
    parser.add_argument("--sample-num", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--l2-steps", type=int, default=200)
    parser.add_argument("--l2-restarts", type=int, default=5)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    if Path(args.output_path).exists():
        raise FileExistsError(args.output_path)
    (audit if args.action == "audit" else generate)(args)


if __name__ == "__main__":
    main()
