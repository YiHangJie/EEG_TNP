import argparse
import logging
import os

from runtime_env import configure_runtime_env

configure_runtime_env()

import torch

from rpcf.core import (
    DATASET_LOADERS,
    compute_interlayer_sensitivity,
    feature_tensor,
    load_model_checkpoint,
    load_rpcf_cache,
    logical_layer_names,
    normalized_feature_shift,
    seed_everything,
    select_sensitive_layers,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="计算 RPCF purification-sensitive logical layers。"
    )
    parser.add_argument("--cache_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--dataset", required=True, choices=DATASET_LOADERS)
    parser.add_argument(
        "--model", required=True, choices=["eegnet", "tsception", "atcnet", "conformer"]
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sensitive_ratio", type=float, default=0.4)
    parser.add_argument("--feature_eps", type=float, default=1e-12)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_prefix", required=True)
    return parser.parse_args()


class FeatureCollector:
    def __init__(self, model, layer_names):
        self.features = {}
        self.handles = []
        for layer_name in layer_names:
            module = model.get_submodule(layer_name)
            self.handles.append(
                module.register_forward_hook(self._make_hook(layer_name))
            )

    def _make_hook(self, layer_name):
        def hook(_module, _inputs, output):
            self.features[layer_name] = feature_tensor(output).detach()

        return hook

    def run(self, model, data):
        self.features = {}
        model(data)
        missing = [
            layer_name
            for layer_name in (handle_layer for handle_layer in self.layer_names)
            if layer_name not in self.features
        ]
        if missing:
            raise RuntimeError(f"Logical layers were not executed: {missing}")
        return dict(self.features)

    @property
    def layer_names(self):
        return tuple(self._layer_names)

    @layer_names.setter
    def layer_names(self, value):
        self._layer_names = tuple(value)

    def close(self):
        for handle in self.handles:
            handle.remove()


def collect_sensitivity(model, cache, layer_names, batch_size, device, eps):
    collector = FeatureCollector(model, layer_names)
    collector.layer_names = layer_names
    ranks = cache["ranks"]
    sums_pur = {
        layer_name: torch.zeros(len(ranks), dtype=torch.float64)
        for layer_name in layer_names
    }
    sums_advpur = {
        layer_name: torch.zeros(len(ranks), dtype=torch.float64)
        for layer_name in layer_names
    }
    sums_input_pur = torch.zeros(len(ranks), dtype=torch.float64)
    sums_input_advpur = torch.zeros(len(ranks), dtype=torch.float64)
    sample_count = cache["x"].size(0)

    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, sample_count, batch_size):
                stop = min(start + batch_size, sample_count)
                clean = cache["x"][start:stop].to(device)
                anchor_features = collector.run(model, clean)
                for rank_index in range(len(ranks)):
                    clean_pur = cache["x_pur_by_rank"][
                        start:stop, rank_index
                    ].to(device)
                    clean_pur_features = collector.run(model, clean_pur)
                    adv_pur = cache["x_adv_pur_by_rank"][
                        start:stop, rank_index
                    ].to(device)
                    adv_pur_features = collector.run(model, adv_pur)
                    sums_input_pur[rank_index] += normalized_feature_shift(
                        clean, clean_pur, eps
                    ).double().sum().cpu()
                    sums_input_advpur[rank_index] += normalized_feature_shift(
                        clean, adv_pur, eps
                    ).double().sum().cpu()
                    for layer_name in layer_names:
                        anchor = anchor_features[layer_name]
                        sums_pur[layer_name][rank_index] += normalized_feature_shift(
                            anchor, clean_pur_features[layer_name], eps
                        ).double().sum().cpu()
                        # RPCF 定义：adversarial purified feature 仍以 clean feature 为 anchor。
                        sums_advpur[layer_name][rank_index] += normalized_feature_shift(
                            anchor, adv_pur_features[layer_name], eps
                        ).double().sum().cpu()
                logging.info(
                    "Sensitivity progress: %d/%d samples", stop, sample_count
                )
    finally:
        collector.close()

    input_pur_by_rank = sums_input_pur / sample_count
    input_advpur_by_rank = sums_input_advpur / sample_count
    raw_pur = {
        layer_name: sums_pur[layer_name] / sample_count
        for layer_name in layer_names
    }
    raw_advpur = {
        layer_name: sums_advpur[layer_name] / sample_count
        for layer_name in layer_names
    }
    relative_pur = compute_interlayer_sensitivity(
        raw_pur, layer_names, input_pur_by_rank, eps
    )
    relative_advpur = compute_interlayer_sensitivity(
        raw_advpur, layer_names, input_advpur_by_rank, eps
    )

    rows = []
    scores = {}
    details = {}
    for layer_index, layer_name in enumerate(layer_names):
        pur_by_rank = raw_pur[layer_name].tolist()
        advpur_by_rank = raw_advpur[layer_name].tolist()
        relative_pur_by_rank = relative_pur[layer_name].tolist()
        relative_advpur_by_rank = relative_advpur[layer_name].tolist()
        pur_mean = float(sum(pur_by_rank) / len(ranks))
        advpur_mean = float(sum(advpur_by_rank) / len(ranks))
        relative_pur_mean = float(sum(relative_pur_by_rank) / len(ranks))
        relative_advpur_mean = float(sum(relative_advpur_by_rank) / len(ranks))
        score = 0.5 * (relative_pur_mean + relative_advpur_mean)
        scores[layer_name] = score
        details[layer_name] = {
            "previous_stage": "input" if layer_index == 0 else layer_names[layer_index - 1],
            "pur_by_rank": {
                str(rank): float(value) for rank, value in zip(ranks, pur_by_rank)
            },
            "advpur_by_rank": {
                str(rank): float(value)
                for rank, value in zip(ranks, advpur_by_rank)
            },
            "pur_mean": pur_mean,
            "advpur_mean": advpur_mean,
            "relative_pur_by_rank": {
                str(rank): float(value)
                for rank, value in zip(ranks, relative_pur_by_rank)
            },
            "relative_advpur_by_rank": {
                str(rank): float(value)
                for rank, value in zip(ranks, relative_advpur_by_rank)
            },
            "relative_pur_mean": relative_pur_mean,
            "relative_advpur_mean": relative_advpur_mean,
            "score": score,
        }
        for rank, pur_value, advpur_value, relative_pur_value, relative_advpur_value in zip(
            ranks,
            pur_by_rank,
            advpur_by_rank,
            relative_pur_by_rank,
            relative_advpur_by_rank,
        ):
            rows.append(
                {
                    "layer": layer_name,
                    "previous_stage": details[layer_name]["previous_stage"],
                    "rank": rank,
                    "pur_sensitivity": pur_value,
                    "advpur_sensitivity": advpur_value,
                    "relative_pur_sensitivity": relative_pur_value,
                    "relative_advpur_sensitivity": relative_advpur_value,
                    "combined_rank_sensitivity": 0.5
                    * (relative_pur_value + relative_advpur_value),
                }
            )
    input_details = {
        "pur_by_rank": {
            str(rank): float(value)
            for rank, value in zip(ranks, input_pur_by_rank.tolist())
        },
        "advpur_by_rank": {
            str(rank): float(value)
            for rank, value in zip(ranks, input_advpur_by_rank.tolist())
        },
    }
    return scores, details, rows, input_details


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)
    logging.basicConfig(
        filename=f"{args.output_prefix}.log",
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    cache = load_rpcf_cache(
        args.cache_path,
        expected={
            "dataset": args.dataset,
            "model": args.model,
            "fold": args.fold,
            "seed": args.seed,
            "eps": args.eps,
        },
    )
    dataset, info = DATASET_LOADERS[args.dataset]()
    del dataset
    model = load_model_checkpoint(
        args.model, args.dataset, info, args.checkpoint_path, device
    )
    layer_names = logical_layer_names(args.model, model)
    scores, details, rows, input_details = collect_sensitivity(
        model,
        cache,
        layer_names,
        args.batch_size,
        device,
        args.feature_eps,
    )
    selected_layers = select_sensitive_layers(scores, args.sensitive_ratio)
    selected_params = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if any(name == layer or name.startswith(f"{layer}.") for layer in selected_layers)
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    artifact = {
        "kind": "rpcf_sensitivity",
        "dataset": args.dataset,
        "model": args.model,
        "fold": args.fold,
        "seed": args.seed,
        "eps": args.eps,
        "cache_path": args.cache_path,
        "checkpoint_path": args.checkpoint_path,
        "ranks": cache["ranks"],
        "feature_anchor": "clean",
        "score_rule": (
            "0.5 * mean_rank(C_pur) + 0.5 * mean_rank(C_advpur), "
            "C_l=S_l/(S_previous+eps)"
        ),
        "first_layer_previous_stage": "input",
        "sensitive_ratio": args.sensitive_ratio,
        "logical_layers": layer_names,
        "selected_layers": selected_layers,
        "selected_params": selected_params,
        "total_params": total_params,
        "selected_param_ratio": selected_params / total_params,
        "input_sensitivity": input_details,
        "layers": details,
    }
    write_json(f"{args.output_prefix}.json", artifact)
    torch.save(artifact, f"{args.output_prefix}.pth")
    write_csv(
        f"{args.output_prefix}.csv",
        [
            "layer",
            "previous_stage",
            "rank",
            "pur_sensitivity",
            "advpur_sensitivity",
            "relative_pur_sensitivity",
            "relative_advpur_sensitivity",
            "combined_rank_sensitivity",
        ],
        rows,
    )
    logging.info("Selected layers: %s", selected_layers)
    print(f"{args.output_prefix}.json")


if __name__ == "__main__":
    main()
