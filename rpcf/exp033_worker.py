"""EXP-033 隔离 worker：复用旧产物，新增评估保持可恢复随机流。"""

import argparse
import copy
import gc
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from rpcf.exp033_common import (DATASET, MODEL, ATTACK_PROTOCOL, read_json, write_json,
    file_hash, fingerprint, source_paths, context, load_attack, load_model, row,
    config_for_rank, training_arguments)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def result_path(root, task_id):
    return root / "metrics" / f"{task_id}.json"


def read_result(root, task_id):
    return read_json(result_path(root, task_id))


def load_state(path, identity, default):
    """载入同一输入、同一任务的断点；恢复随机状态必须晚于模型初始化。"""
    from rpcf.exp032_common import load_payload
    from rpcf.exp032_tnp import restore_rng
    if not path.exists():
        return copy.deepcopy(default)
    state = load_payload(path)
    if state["identity"] != identity:
        raise ValueError("Partial source/config mismatch")
    restore_rng(state["rng_state"])
    return state["progress"]


def save_state(path, identity, progress):
    from rpcf.exp031_artifacts import atomic_torch_save
    from rpcf.exp032_tnp import rng_state
    atomic_torch_save(dict(identity=identity, progress=progress, rng_state=rng_state()), path)


def probabilities(model, tensor):
    import torch
    device = next(model.parameters()).device
    predictions, confidences = [], []
    with torch.no_grad():
        for batch in tensor.split(32):
            confidence, pred = model(batch.to(device)).softmax(-1).max(-1)
            predictions.extend(pred.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    return predictions, confidences


def artifact_records(result):
    """给本任务引用的持久产物绑定内容指纹，避免只有 JSON 完成而文件丢失。"""
    names = [result[k] for k in ("checkpoint_path", "history_path", "attack_path", "capture_path", "bundle_path", "smoke_validation_path") if result.get(k)]
    names.extend(result.get("figures", []))
    records = []
    for name in dict.fromkeys(names):
        path = Path(name)
        stat = path.stat()
        records.append(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=file_hash(path)))
    return records


def verify_artifacts(result):
    for entry in result.get("artifacts", []):
        path = Path(entry["path"])
        stat = path.stat()
        if (stat.st_size != entry["size"] or stat.st_mtime_ns != entry["mtime_ns"]) and file_hash(path) != entry["sha256"]:
            raise ValueError(f"Generated artifact changed: {path}")


def frozen_sources(root, task, ctx=None):
    snap = read_result(root, f"sources_seed{task['seed']}")
    for entry in snap["files"].values():
        p = Path(entry["path"])
        stat = p.stat()
        if stat.st_size != entry["size"] or stat.st_mtime_ns != entry["mtime_ns"]:
            if file_hash(p) != entry["sha256"]:
                raise ValueError(f"Frozen source changed: {p}")
    if ctx is not None:
        if snap["source_indices"] != ctx.indices or snap["labels"] != ctx.labels.tolist() or snap["clean_sha256"] != tensor_hash(ctx.clean):
            raise ValueError("Canonical input changed")
    return snap


def sources(manifest, task, ctx, root, device):
    from importlib.metadata import version
    from rpcf.exp032_common import load_payload, align_payload
    from rpcf.exp033_common import WEIGHTS
    paths = source_paths(manifest, task["seed"])
    audit = read_json(paths["audit"])
    for method in ("madry", "rpcf_at"):
        check = next(c for c in audit["checks"] if c["method"] == method)
        if Path(check["checkpoint_path"]).resolve() != Path(paths[f"checkpoint_{method}"]).resolve() or check["checkpoint_sha256"] != file_hash(paths[f"checkpoint_{method}"]):
            raise ValueError("Historical checkpoint audit differs")
    audit_labels = dict(zip(audit["source_indices"], audit["labels"]))
    if any(audit_labels.get(index) != label for index, label in zip(ctx.indices, ctx.labels.tolist())):
        raise ValueError("Historical sample identity differs")
    checks = {}
    for method in ("madry", "rpcf_at", "clean"):
        _, checks[method] = load_attack(paths[f"attack_{method}"], ctx, paths[f"checkpoint_{method}"])
    for key in ("tnp_madry", "tnp_rpcf_at", "canonical_clean_tnp"):
        payload = load_payload(paths[key])
        if payload["ranks"] != [25, 30]:
            raise ValueError("Source ranks changed")
        aligned = align_payload(payload, ctx.indices, ctx.labels, ctx.clean)
        if key != "canonical_clean_tnp":
            method = key.removeprefix("tnp_")
            meta = payload["meta"]
            if Path(meta["checkpoint_path"]).resolve() != Path(paths[f"checkpoint_{method}"]).resolve() or Path(meta["attack_path"]).resolve() != Path(paths[f"attack_{method}"]).resolve():
                raise ValueError("TNP classifier/attack provenance differs")
            expected_adv, _ = load_attack(paths[f"attack_{method}"],ctx,paths[f"checkpoint_{method}"])
            import torch
            if not torch.equal(aligned["adversarial"],expected_adv):
                raise ValueError("TNP adversarial inputs differ")
            if any(meta["attack_meta"]["attack_protocol"].get(k) != v for k,v in ATTACK_PROTOCOL.items()):
                raise ValueError("TNP nested attack protocol differs")
        for name, value in (("dataset", DATASET), ("model", MODEL), ("seed", task["seed"]), ("fold", 0)):
            if payload["meta"].get(name) != value:
                raise ValueError(f"TNP provenance mismatch: {key}/{name}")
    history = read_json(paths["training_history"])
    if not history["all_layers"] or not history["static_rank_weights"] or not history["online_madry_at"] or history["cached_adv_loss_enabled"]:
        raise ValueError("CAF reference protocol differs")
    names = dict(clean_ce_weight="clean_ce", pur_ce_weight="pur_ce", adv_pur_ce_weight="adv_pur_ce",
                 lambda_pur="pur_kl", lambda_adv_pur="adv_pur_kl")
    if any(history["loss_weights"][names[k]] != v for k, v in WEIGHTS.items()):
        raise ValueError("CAF reference loss weights differ")
    if read_json(paths["training_status"])["status"] != "completed":
        raise ValueError("Source training is incomplete")
    for name in ("magnet", "dcae"):
        payload = load_payload(paths[f"purifier_{name}"])
        for key, value in (("method", name), ("dataset", DATASET), ("seed", task["seed"]), ("fold", 0)):
            if payload["meta"].get(key) != value:
                raise ValueError("External purifier provenance mismatch")
    files = {}
    for key, name in paths.items():
        p = Path(name)
        stat = p.stat()
        files[key] = dict(path=str(p), size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=file_hash(p))
        print(f"SOURCE {key} verified", flush=True)
    return dict(files=files, checks=checks, clean_sha256=tensor_hash(ctx.clean), split=ctx.split,
                versions={name: version(name) for name in ("torch", "tensorly", "numpy", "scipy", "quimb", "tntorch")}, rows=[])


def purifier_args(manifest, rank):
    return SimpleNamespace(dataset=DATASET, model=MODEL, seed=42, visualize=False,
                           config=config_for_rank(manifest, rank))


def reconstruct(manifest, sample, spec, device, index=0):
    """公共变换逐式沿用 purify；配置读取和导入均在计时区外。"""
    import numpy as np
    import torch
    import yaml
    from purify import fft_resample, ToInterpolatedGrid, dataset_channel_location_dicts
    from TN.opt import Config
    from TN.utils import get_TN_args
    from TN.PTR_3d import PTR_3d
    from rpcf.exp033_structures import decompose
    args = purifier_args(manifest, spec.get("budget_rank", 25))
    config = Config()
    for key, value in yaml.safe_load(Path(args.config).read_text()).items():
        setattr(config, key, value)
    if config.strategy != "3d_interpolate":
        raise ValueError("EXP-033 requires the frozen common 3d_interpolate representation")
    config.device_type = "gpu" if device.type == "cuda" else "cpu"
    locations = dataset_channel_location_dicts[DATASET]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    # 与 purify.interpolate 的 3d_interpolate 分支相同，只移走 YAML 读取。
    data = sample.detach().cpu().clone().permute(1, 2, 0)
    grid = ToInterpolatedGrid(locations)
    pre = torch.from_numpy(grid(eeg=data.squeeze().numpy())["eeg"]).permute(1, 2, 0).float()
    target_len = round(2 ** np.ceil(np.log2(data.shape[1]) + config.c))
    pre = fft_resample(pre, target_len=target_len).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decomposition_start = time.perf_counter()
    if spec["method"] == "ptr":
        net = PTR_3d(**get_TN_args(config, pre.clone(), 250, None, config.device_type))
        rebuilt, _, _ = net.train(pre.clone(), config, index, logging=logging)
        diagnostic = dict(actual_parameters=net.count_parameters(), actual_rank=spec["budget_rank"])
        rebuilt = rebuilt.detach()
        del net
    else:
        with torch.no_grad():
            rebuilt, diagnostic = decompose(pre, spec)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decomposition_time = time.perf_counter() - decomposition_start
    # 与 purify.inv_interpolate 的对应分支相同。
    restored = fft_resample(rebuilt.detach().clone(), target_len=sample.shape[-1])
    grid = ToInterpolatedGrid(locations)
    purified = torch.from_numpy(grid.reverse(eeg=restored.permute(2, 0, 1).cpu().numpy())["eeg"]).unsqueeze(0).float()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if purified.shape != sample.shape or not torch.isfinite(purified).all():
        raise ValueError("Invalid purified output")
    diagnostic.update(decomposition_seconds=decomposition_time, total_seconds=time.perf_counter()-start)
    return purified, diagnostic


def calibrate(manifest, task, ctx, root, device):
    import torch
    from purify import interpolate
    from utils.experiment_artifacts import eeg_classification_collate
    from utils.reproducibility import stable_subset_indices
    from rpcf.exp033_structures import METHODS, candidates, decompose, select_candidate
    positions, selection_seed = stable_subset_indices(len(ctx.val), min(manifest["validation_sample_num"],len(ctx.val)), task["seed"], 0)
    x, _ = eeg_classification_collate([ctx.val[i] for i in positions])
    args = purifier_args(manifest, 25)
    tensors = [interpolate(args, item, 250).to(device) for item in x]
    identity = fingerprint(dict(task=task, validation_indices=positions, input_hash=tensor_hash(x), smoke=manifest["smoke"]))
    partial = root / "partials" / f"{task['task_id']}.pth"
    progress = load_state(partial, identity, {"scored": {}, "selected": {}})
    for method in METHODS:
        for budget in (25, 30):
            key = f"{method}_{budget}"
            specs = candidates(method, budget)
            scores = progress["scored"].setdefault(key, [])
            for spec in specs[len(scores):]:
                errors = []
                with torch.no_grad():
                    for tensor in tensors:
                        rebuilt, _ = decompose(tensor, spec)
                        errors.append(float((rebuilt-tensor).norm()/tensor.norm().clamp_min(1e-20)))
                scores.append({**spec, "mean_relative_error": sum(errors)/len(errors)})
                save_state(partial, identity, progress)
                print(f"CALIBRATE {key} {len(scores)}/{len(specs)}", flush=True)
            progress["selected"][key] = select_candidate(scores)
            save_state(partial, identity, progress)
    return dict(rows=[], validation_indices=positions, selection_seed=selection_seed,
                validation_input_sha256=tensor_hash(x), selection_split="validation", **progress)


def reference(manifest, task, ctx, root, device):
    from rpcf.exp032_common import load_payload, align_payload, predict
    paths = source_paths(manifest, task["seed"])
    rows = []
    methods = ("clean", "madry", "rpcf_at") if task["group"] == "ablation" else (task["method"],)
    for method in methods:
        model = load_model(paths[f"checkpoint_{method}"], ctx, device)
        adv, _ = load_attack(paths[f"attack_{method}"], ctx, paths[f"checkpoint_{method}"])
        if task["group"] != "structure":
            rows.append(row(task, ctx, predict(model, ctx.clean), predict(model, adv),
                            method={"rpcf_at":"caf"}.get(method,method), rank=0,
                            checkpoint_sha256=file_hash(paths[f"checkpoint_{method}"])))
        if method != "clean":
            payload = load_payload(paths[f"tnp_{method}"])
            aligned = align_payload(payload, ctx.indices, ctx.labels, ctx.clean)
            for ri, rank in enumerate((25, 30)):
                rows.append(row(task, ctx, predict(model, aligned["clean_pur_by_rank"][:,ri]),
                                predict(model, aligned["adv_pur_by_rank"][:,ri]), rank=rank,
                                budget_rank=rank if task["group"]=="structure" else None,
                                parameters={25:14731,30:19891}[rank] if task["group"]=="structure" else None,
                                method="ptr" if task["group"]=="structure" else "trp_caf" if method=="rpcf_at" else "trp_madry",
                                evaluation="nonadaptive_purification", reused_from=paths[f"tnp_{method}"]))
        del model
    return dict(rows=rows)


def structure(manifest, task, ctx, root, device):
    from rpcf.exp032_common import predict
    paths = source_paths(manifest, task["seed"])
    spec = read_result(root, f"calibrate_seed{task['seed']}")["selected"][f"{task['method']}_{task['budget_rank']}"]
    model = load_model(paths["checkpoint_madry"], ctx, device)
    adv, norms = load_attack(paths["attack_madry"], ctx, paths["checkpoint_madry"])
    identity = fingerprint(dict(task=task, spec=spec, sources=frozen_sources(root,task)["files"]))
    partial = root / "partials" / f"{task['task_id']}.pth"
    state = load_state(partial, identity, {"cp":[], "ap":[], "diagnostics":[]})
    for i in range(len(state["cp"]),len(ctx.clean)):
        pc, dc = reconstruct(manifest, ctx.clean[i], spec, device, i)
        pa, da = reconstruct(manifest, adv[i], spec, device, i)
        state["cp"].extend(predict(model,pc.unsqueeze(0),1))
        state["ap"].extend(predict(model,pa.unsqueeze(0),1))
        state["diagnostics"].append(dict(source_index=ctx.indices[i], clean_mse=float((pc-ctx.clean[i]).square().mean()),
                                         adv_mse_to_clean=float((pa-ctx.clean[i]).square().mean()),
                                         removed_mse=float((pa-adv[i]).square().mean()),
                                         clean_sha256=tensor_hash(pc),adv_sha256=tensor_hash(pa),
                                         actual_parameters=da["actual_parameters"], actual_rank=da["actual_rank"],
                                         equivalence_note=da.get("equivalence_note")))
        save_state(partial,identity,state)
        print(f"STRUCTURE {task['method']} {i+1}/{len(ctx.clean)}",flush=True)
    metrics = row(task,ctx,state["cp"],state["ap"],budget_rank=task["budget_rank"],
                  parameters=spec["parameters"],actual_rank=spec["rank"],
                  budget_deviation=spec["budget_deviation"],budget_exception=spec["budget_exception"],
                  evaluation="nonadaptive_purification")
    return dict(rows=[metrics],spec=spec,norm_audit=norms,sample_diagnostics=state["diagnostics"])


def assert_exclusive(device):
    if device.type != "cuda":
        return
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not physical or "," in physical:
        raise RuntimeError("Timing requires one explicitly isolated GPU")
    output = subprocess.check_output(["nvidia-smi", "-i", physical, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],text=True)
    if any(int(x.strip()) != os.getpid() for x in output.splitlines() if x.strip().isdigit()):
        raise RuntimeError("Timing GPU is shared; preserve partials and retry on an idle GPU")


def timing(manifest,task,ctx,root,device):
    import torch
    from statistics import mean,median
    if device.type != "cuda" and not manifest["smoke"]:
        raise RuntimeError("Formal timing requires CUDA")
    paths=source_paths(manifest,task["seed"])
    adv,_=load_attack(paths["attack_madry"],ctx,paths["checkpoint_madry"])
    spec=(dict(method="ptr",budget_rank=task["budget_rank"],parameters={25:14731,30:19891}[task["budget_rank"]])
          if task["method"]=="ptr" else read_result(root,f"calibrate_seed{task['seed']}")["selected"][f"{task['method']}_{task['budget_rank']}"])
    assert_exclusive(device)
    reconstruct(manifest,ctx.clean[0],spec,device)  # 预热不进入计时统计。
    diagnostics=[]
    for label,inputs in (("clean",ctx.clean[:8]),("adv",adv[:8])):
        for i,sample in enumerate(inputs):
            assert_exclusive(device)
            if device.type=="cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
            _,diag=reconstruct(manifest,sample,spec,device,i)
            diag.update(input_kind=label,source_index=ctx.indices[i],peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0)
            diagnostics.append(diag)
    assert_exclusive(device)
    metrics=dict(group="structure",method=task["method"],seed=task["seed"],budget_rank=task["budget_rank"],
                 evaluation="timing",parameters=spec["parameters"],sample_num=len(diagnostics),
                 decomposition_seconds_mean=mean(d["decomposition_seconds"] for d in diagnostics),
                 total_seconds_mean=mean(d["total_seconds"] for d in diagnostics),
                 total_seconds_median=median(d["total_seconds"] for d in diagnostics),
                 peak_memory_bytes=max(d["peak_memory_bytes"] for d in diagnostics),
                 device=torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU_SMOKE_NOT_FORMAL",
                 native_compression_ratio=ctx.clean[0].numel()/spec["parameters"],
                 tensor_compression_ratio=225280/spec["parameters"])
    return dict(rows=[metrics],samples=diagnostics,spec=spec)


def train(manifest,task,ctx,root,device):
    from rpcf.exp031_artifacts import atomic_torch_save
    from rpcf.exp032_tnp import rng_state
    base=root/"training"/task["task_id"]
    base.mkdir(parents=True,exist_ok=True)
    complete=[]
    for p in sorted(base.glob("attempt*/completion.json")):
        candidate=read_json(p)
        if candidate["task_fingerprint"]==fingerprint(task) and Path(candidate["checkpoint_path"]).exists() and file_hash(candidate["checkpoint_path"])==candidate["checkpoint_sha256"]:
            complete.append(candidate)
    if complete:
        return complete[-1]
    attempt=base/f"attempt{len(list(base.glob('attempt*')))+1:03d}"
    attempt.mkdir()
    module="rpcf.exp033_smoke_finetune" if manifest["smoke"] else "rpcf.finetune"
    command=[sys.executable,"-u","-m",module,*training_arguments(manifest,task,attempt)]
    write_json(attempt/"command.json",command)
    atomic_torch_save(rng_state(),attempt/"initial_rng.pth")
    # 原训练器仅保存最终 epoch；中断从原初始化/seed 重跑该变体，不覆盖旧尝试。
    subprocess.run(command,check=True)
    cp=attempt/"checkpoint.pth"
    history=read_json(attempt/"history.json")
    if not cp.exists() or history["cached_adv_loss_enabled"]:
        raise ValueError("Training completion invalid")
    result=dict(rows=[],checkpoint_path=str(cp),checkpoint_sha256=file_hash(cp),
                history_path=str(attempt/"history.json"),command=command,task_fingerprint=fingerprint(task))
    if manifest["smoke"]:
        result["smoke_validation_path"]=str(attempt/"smoke_validation.json")
    write_json(attempt/"completion.json",result)
    return result


def attack(manifest,task,ctx,root,device):
    import torch
    from attack.pgd import PGD
    from rpcf.exp032_common import predict,norm_audit
    from rpcf.exp031_artifacts import atomic_torch_save
    trained=read_result(root,task["training_task"])
    if file_hash(trained["checkpoint_path"]) != trained["checkpoint_sha256"]:
        raise ValueError("Trained checkpoint contents changed")
    model=load_model(trained["checkpoint_path"],ctx,device)
    model.requires_grad_(False)
    attacker=PGD(model,device=device,eps=.03,alpha=2/255,steps=200)
    identity=fingerprint(dict(task=task,checkpoint=trained["checkpoint_sha256"],indices=ctx.indices))
    partial=root/"partials"/f"{task['task_id']}.pth"
    state=load_state(partial,identity,{"parts":[]})
    for i in range(len(state["parts"]),len(ctx.clean)):
        adversarial=attacker(ctx.clean[i:i+1].to(device),ctx.labels[i:i+1].to(device)).cpu().float()
        state["parts"].append(adversarial)
        save_state(partial,identity,state)
        print(f"ATTACK {i+1}/{len(ctx.clean)}",flush=True)
    adversarial=torch.cat(state["parts"])
    norms=norm_audit(ctx.clean,adversarial,"Linf",.03)
    cp,ap=predict(model,ctx.clean),predict(model,adversarial)
    dest=root/"attacks"/f"{task['task_id']}.pth"
    if shutil.disk_usage(root).free < adversarial.numel()*4+32*1024**3:
        raise RuntimeError("Preserve 32GiB storage reserve")
    payload=dict(adversarial=adversarial,labels=ctx.labels,source_indices=ctx.indices,
                 clean_reference_path=source_paths(manifest,task["seed"])["attack_madry"],
                 meta=dict(kind="rpcf_attack_eval",experiment_id="EXP-033",dataset=DATASET,model=MODEL,
                           seed=task["seed"],fold=0,attack="pgd",attack_protocol=ATTACK_PROTOCOL,
                           checkpoint_path=trained["checkpoint_path"],checkpoint_sha256=trained["checkpoint_sha256"]))
    atomic_torch_save(payload,dest)
    return dict(rows=[row(task,ctx,cp,ap,method="caf",rank=0,evaluation="classifier_whitebox")],
                attack_path=str(dest),attack_sha256=file_hash(dest),norm_audit=norms,
                checkpoint_path=trained["checkpoint_path"])


def external_models(manifest,task,device):
    from rpcf.exp032_common import load_payload
    from rpcf.exp032_purifiers import EEGReformer
    models={}
    for name in ("magnet","dcae"):
        payload=load_payload(source_paths(manifest,task["seed"])[f"purifier_{name}"])
        meta=payload["meta"]
        net=EEGReformer(name,meta["low"],meta["high"]).to(device)
        net.load_state_dict(payload["state_dict"])
        models[name]=net.eval()
    return models


CATEGORIES=("success","failure","clean_damage","disagreement")


def select_cases(indices,labels,raw_cp,raw_ap,ranks,external_preds):
    """按 source index 排序选取不同案例；分歧包括所有方法的预测类别。"""
    selected={}
    for category in CATEGORIES:
        for i in sorted(range(len(indices)),key=lambda j:indices[j]):
            if indices[i] in selected.values():continue
            attacked=raw_cp[i]==labels[i] and raw_ap[i]!=labels[i]
            predictions=[ranks[r]["ap"][i] for r in (25,30)] + [v[i] for v in external_preds.values()]
            eligible=dict(success=attacked and ranks[25]["ap"][i]==labels[i],
                          failure=attacked and ranks[25]["ap"][i]!=labels[i],
                          clean_damage=raw_cp[i]==labels[i] and ranks[25]["cp"][i]!=labels[i],
                          disagreement=len(set(predictions))>1)
            if eligible[category]:
                selected[category]=indices[i]
                break
    return selected


def tnp(manifest,task,ctx,root,device):
    import torch
    from purify import purify
    from rpcf.exp032_common import load_payload,align_payload,predict
    from rpcf.exp031_artifacts import atomic_torch_save
    paths=source_paths(manifest,task["seed"])
    checkpoint=paths[f"checkpoint_{task['method']}"]
    attack_file=paths[f"attack_{task['method']}"]
    if task.get("attack_task"):
        generated=read_result(root,task["attack_task"])
        checkpoint,attack_file=generated["checkpoint_path"],generated["attack_path"]
    model=load_model(checkpoint,ctx,device)
    adversarial,norms=load_attack(attack_file,ctx,checkpoint)
    canonical=align_payload(load_payload(paths["canonical_clean_tnp"]),ctx.indices,ctx.labels,ctx.clean)
    raw_cp,raw_ap=predict(model,ctx.clean),predict(model,adversarial)
    external_preds={}
    if task.get("capture"):
        from rpcf.exp032_tnp import rng_state, restore_rng
        before_external=rng_state()
        for name,net in external_models(manifest,task,device).items():
            external_preds[name]=predict(torch.nn.Sequential(net,model),adversarial,8)
        restore_rng(before_external)
        del net
    identity=fingerprint(dict(task=task,checkpoint=file_hash(checkpoint),attack=file_hash(attack_file),
                              sources=frozen_sources(root,task)["files"],smoke=manifest["smoke"]))
    partial=root/"partials"/f"{task['task_id']}.pth"
    state=load_state(partial,identity,dict(ranks={}))
    for rank in task["ranks"]:
        progress=state["ranks"].setdefault(rank,dict(cp=[],ap=[],clean_confidence=[],adv_confidence=[],diagnostics=[],rng_before=[]))
        config=purifier_args(manifest,rank);config.seed=task["seed"]
        for i in range(len(progress["cp"]),len(ctx.clean)):
            if task.get("capture"):
                from rpcf.exp032_tnp import rng_state
                progress["rng_before"].append(rng_state())
            if rank in (25,30):
                clean_pur=canonical["clean_pur_by_rank"][i,(25,30).index(rank)].clone()
            else:
                clean_pur,_=purify(config,i,ctx.clean[i],250,device,logging,classifier=model)
            adv_pur,_=purify(config,i+len(ctx.clean),adversarial[i],250,device,logging,classifier=model)
            clean_pur,adv_pur=clean_pur.cpu().float(),adv_pur.cpu().float()
            if not torch.isfinite(clean_pur).all() or not torch.isfinite(adv_pur).all():raise ValueError("Non-finite TNP output")
            cp,cc=probabilities(model,clean_pur.unsqueeze(0));ap,ac=probabilities(model,adv_pur.unsqueeze(0))
            progress["cp"].extend(cp);progress["ap"].extend(ap)
            progress["clean_confidence"].extend(cc);progress["adv_confidence"].extend(ac)
            progress["diagnostics"].append(dict(source_index=ctx.indices[i],clean_mse=float((clean_pur-ctx.clean[i]).square().mean()),
                                                adv_mse_to_clean=float((adv_pur-ctx.clean[i]).square().mean()),
                                                removed_mse=float((adv_pur-adversarial[i]).square().mean()),
                                                clean_sha256=tensor_hash(clean_pur),adv_sha256=tensor_hash(adv_pur)))
            if (i+1)%8==0 or i+1==len(ctx.clean): save_state(partial,identity,state)
            print(f"TNP rank{rank} {i+1}/{len(ctx.clean)}",flush=True)
    rows=[]
    for rank,progress in state["ranks"].items():
        rows.append(row(task,ctx,progress["cp"],progress["ap"],rank=rank,
                        method="trp_clean" if task["method"]=="clean" else "trp_caf",
                        clean_confidence=progress["clean_confidence"],adv_confidence=progress["adv_confidence"],
                        evaluation="nonadaptive_purification"))
    result=dict(rows=rows,norm_audit=norms,sample_diagnostics={str(k):v["diagnostics"] for k,v in state["ranks"].items()},
                checkpoint_path=checkpoint,attack_path=attack_file)
    if task.get("capture"):
        from rpcf.exp032_tnp import rng_state,restore_rng
        selected=select_cases(ctx.indices,ctx.labels.tolist(),raw_cp,raw_ap,state["ranks"],external_preds)
        case_tensors={}
        final_rng=rng_state()
        try:
            for category,sid in selected.items():
                i=ctx.indices.index(sid)
                signals={}
                for ri,rank in enumerate((25,30)):
                    restore_rng(state["ranks"][rank]["rng_before"][i])
                    cfg=purifier_args(manifest,rank);cfg.seed=task["seed"]
                    purified,_=purify(cfg,i+len(ctx.clean),adversarial[i],250,device,logging,classifier=model)
                    purified=purified.cpu().float()
                    if tensor_hash(purified)!=state["ranks"][rank]["diagnostics"][i]["adv_sha256"]:
                        raise ValueError("Case RNG replay differs from evaluated purification")
                    signals[f"trp{rank}_clean"]=canonical["clean_pur_by_rank"][i,ri].clone()
                    signals[f"trp{rank}_adv"]=purified
                case_tensors[sid]=signals
        finally:
            restore_rng(final_rng)
        capture_path=root/"cases"/f"seed{task['seed']}_trp.pth"
        atomic_torch_save(dict(cases=case_tensors,categories={sid:cat for cat,sid in selected.items()},
                              source_indices=ctx.indices,labels=ctx.labels),capture_path)
        result["capture_path"]=str(capture_path)
    return result


def visualize(manifest,task,ctx,root,device):
    import torch
    from rpcf.exp032_common import load_payload
    from rpcf.exp031_artifacts import atomic_torch_save
    from rpcf.exp033_report import render_cases
    result=read_result(root,task["ablation_task"])
    captures=load_payload(result["capture_path"])
    model=load_model(result["checkpoint_path"],ctx,device)
    adv,_=load_attack(result["attack_path"],ctx,result["checkpoint_path"])
    raw_cp,raw_cc=probabilities(model,ctx.clean)
    raw_ap,raw_ac=probabilities(model,adv)
    predictions={"raw":dict(clean=raw_cp,adv=raw_ap,clean_confidence=raw_cc,adv_confidence=raw_ac)}
    rows=[row(task,ctx,raw_cp,raw_ap,method="raw",rank=0)]
    for r in result["rows"]:
        predictions[f"trp{r['rank']}"]=dict(clean=r["clean_predictions"],adv=r["adv_predictions"],
                                            clean_confidence=r["clean_confidence"],adv_confidence=r["adv_confidence"])
        rows.append({**r,"group":"visualize"})
    case_signals={int(sid):{**values,"clean":ctx.clean[ctx.indices.index(int(sid))],"adv":adv[ctx.indices.index(int(sid))]}
                  for sid,values in captures["cases"].items()}
    statistics=dict(trp=result["sample_diagnostics"],external={})
    for name,net in external_models(manifest,task,device).items():
        cp,ap,cc,ac,diags=[],[],[],[],[]
        for i in range(len(ctx.clean)):
            with torch.no_grad():
                pc=net(ctx.clean[i:i+1].to(device)).cpu()
                pa=net(adv[i:i+1].to(device)).cpu()
            p,c=probabilities(model,pc);cp.extend(p);cc.extend(c)
            p,c=probabilities(model,pa);ap.extend(p);ac.extend(c)
            diags.append(dict(source_index=ctx.indices[i],clean_mse=float((pc[0]-ctx.clean[i]).square().mean()),
                              adv_mse_to_clean=float((pa[0]-ctx.clean[i]).square().mean()),removed_mse=float((pa[0]-adv[i]).square().mean())))
            if ctx.indices[i] in case_signals:
                case_signals[ctx.indices[i]][f"{name}_clean"]=pc[0]
                case_signals[ctx.indices[i]][f"{name}_adv"]=pa[0]
        predictions[name]=dict(clean=cp,adv=ap,clean_confidence=cc,adv_confidence=ac)
        statistics["external"][name]=diags
        rows.append(row(task,ctx,cp,ap,method=name))
    cases=[]
    for sid,signals in case_signals.items():
        i=ctx.indices.index(sid)
        cases.append(dict(category=captures["categories"][sid],source_index=sid,label=int(ctx.labels[i]),sampling_rate=250,
                          signals=signals,predictions={name:{key:values[i] for key,values in fields.items()} for name,fields in predictions.items()}))
    bundle=dict(seed=task["seed"],smoke=manifest["smoke"],case_list=cases,empty_categories=[c for c in CATEGORIES if c not in captures["categories"].values()],
                statistics=statistics,source_indices=ctx.indices,labels=ctx.labels.tolist())
    bundle_path=root/"cases"/f"seed{task['seed']}_visualize.pth"
    atomic_torch_save(bundle,bundle_path)
    figures=render_cases(bundle_path,root/"figures"/f"seed{task['seed']}")
    return dict(rows=rows,bundle_path=str(bundle_path),figures=[str(p) for p in figures],
                empty_categories=bundle["empty_categories"],statistics=statistics)


HANDLERS={"sources":sources,"calibrate":calibrate,"reference":reference,"structure":structure,
          "timing":timing,"train":train,"attack":attack,"tnp":tnp,"visualize":visualize}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--task-id",required=True)
    args=parser.parse_args()
    from rpcf.exp033 import verify_plan
    manifest,tasks=verify_plan(args.run_dir)
    task=next(t for t in tasks if t["task_id"]==args.task_id)
    output=Path(task["output_path"])
    if output.exists():
        existing=read_json(output)
        if existing.get("task_fingerprint")!=fingerprint(task):raise ValueError("Existing output belongs to another task")
        frozen_sources(args.run_dir,task)
        verify_artifacts(existing)
        print("Verified existing worker output",flush=True)
        return
    import torch
    from utils.reproducibility import seed_everything
    torch.set_num_threads(2)
    seed_everything(task["seed"])
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s")
    ctx=context(manifest,task["seed"])
    if task["kind"]!="sources":frozen_sources(args.run_dir,task,ctx)
    for dep in task["dependencies"]:
        if not result_path(args.run_dir,dep).exists():raise ValueError(f"Missing dependency {dep}")
        verify_artifacts(read_result(args.run_dir,dep))
    result=HANDLERS[task["kind"]](manifest,task,ctx,args.run_dir,device)
    result.update(experiment_id="EXP-033",task_id=task["task_id"],task_fingerprint=fingerprint(task),
                  smoke=manifest["smoke"],source_indices=ctx.indices,labels=ctx.labels.tolist(),
                  dataset=DATASET,model=MODEL,seed=task["seed"],fold=0)
    result["artifacts"]=artifact_records(result)
    write_json(output,result)


if __name__=="__main__":main()
