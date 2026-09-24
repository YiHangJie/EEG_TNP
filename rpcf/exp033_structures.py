"""EXP-033 参数预算匹配的标准张量分解；不改变既有 PTR 净化路径。

本模块仅接收共同插值后的三维张量，插值、样本选择与分类由实验入口负责。
依赖在分解时导入，因此建清单不要求初始化 PyTorch 或 CUDA。
"""

import math
from functools import lru_cache


METHODS = ("tr_dense", "tt_dense", "tt_time", "tucker", "svd")
SHAPE = (10, 11, 2048)
BUDGETS = {25: 14731, 30: 19891}


def _shape(shape):
    shape = tuple(shape)
    if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("Expected three positive integer dimensions")
    return shape


def _time_shape(shape):
    h, w, t = _shape(shape)
    if t < 2 or t & (t - 1):
        raise ValueError("Time tensorization requires a power-of-two time dimension >= 2")
    return (h, w) + (2,) * (t.bit_length() - 1)


def _tt_cap_ranks(shape, cap):
    """逐切分截断统一上限，避免把 TensorLy 实际剪裁掉的参数计入预算。"""
    return [1] + [min(cap, math.prod(shape[:i]), math.prod(shape[i:]))
                  for i in range(1, len(shape))] + [1]


def parameter_count(method, rank, shape=SHAPE):
    """计算实际存储元素数；SVD 包含 U、S、V，Tucker 包含 core 与各因子。"""
    h, w, t = _shape(shape)
    if method == "svd":
        return int(rank) * (h * w + t + 1)
    if method == "tucker":
        a, b, c = rank
        return a * b * c + h * a + w * b + t * c
    dims = _time_shape(shape) if method == "tt_time" else (h, w, t)
    if method not in ("tr_dense", "tt_dense", "tt_time"):
        raise ValueError(f"Unknown structure: {method}")
    if method == "tt_time" and isinstance(rank, int):
        rank = _tt_cap_ranks(dims, rank)
    if len(rank) != len(dims) + 1:
        raise ValueError("Rank vector does not match decomposition dimensions")
    return sum(rank[i] * n * rank[i + 1] for i, n in enumerate(dims))


def _rank_key(rank):
    return (rank,) if isinstance(rank, int) else tuple(rank)


def _nondominated(specs):
    # 预算范围已过滤；只去除在所有 rank 上均不占优、且至少一维较小的配置。
    ranks = [_rank_key(spec["rank"]) for spec in specs]
    return [spec for i, spec in enumerate(specs)
            if not any(i != j and all(a <= b for a, b in zip(ranks[i], other))
                       and ranks[i] != other for j, other in enumerate(ranks))]


@lru_cache(maxsize=None)
def _candidate_specs(method, budget_rank, shape):
    if method not in METHODS:
        raise ValueError(f"Unknown structure: {method}")
    if budget_rank not in BUDGETS:
        raise ValueError(f"Unknown budget rank: {budget_rank}")
    h, w, t = _shape(shape)
    target = BUDGETS[budget_rank]
    lower, upper = target * .95, target * 1.05
    specs = []

    def add(rank, exception=False, **extra):
        count = parameter_count(method, rank, shape)
        if exception or lower <= count <= upper:
            specs.append(dict(method=method, rank=rank, shape=list(shape),
                              parameters=count, target_parameters=target,
                              budget_deviation=count / target - 1,
                              budget_exception=exception, **extra))

    if method == "tr_dense":
        if shape == SHAPE and budget_rank == 25:
            add([3, 22, 2, 3], exception=True)
        else:
            # 时间维先分解要求 a*c <= min(T,H*W)，第二轮要求 b <= min(H*a,W*c)。
            product_limit = min(t, h * w, int(upper // t))
            for a in range(2, product_limit // 2 + 1):
                for c in range(2, product_limit // a + 1):
                    for b in range(2, min(h * a, w * c) + 1):
                        add([a, b, c, a])
    elif method == "tt_dense":
        for a in range(1, min(h, w * t) + 1):
            for b in range(1, min(a * w, t) + 1):
                if a <= w * b:
                    add([1, a, b, 1])
    elif method == "tt_time":
        dims = _time_shape(shape)
        max_cap = max(min(math.prod(dims[:i]), math.prod(dims[i:]))
                      for i in range(1, len(dims)))
        for cap in range(1, max_cap + 1):
            rank = _tt_cap_ranks(dims, cap)
            count = parameter_count(method, rank, shape)
            if count > upper:
                break
            add(rank, rank_cap=cap, tensorized_shape=list(dims))
    elif method == "tucker":
        for a in range(1, h + 1):
            for b in range(1, w + 1):
                for c in range(1, min(t, a * b, int(upper // t)) + 1):
                    if a <= b * c and b <= a * c:
                        add([a, b, c])
    else:
        for rank in range(1, min(h * w, t) + 1):
            add(rank)
    return tuple(_nondominated(specs))


def candidates(method, budget_rank, shape=SHAPE):
    """返回预算内的非支配候选；低预算普通 TR 仅保留批准的 −8.8% 例外。

    无法满足预算时返回空列表，由实验入口阻止无候选条件进入测试。
    返回副本，避免调用者写入验证误差等字段时污染后续 seed 的候选。
    """
    import copy
    return copy.deepcopy(list(_candidate_specs(method, budget_rank, _shape(shape))))


def select_candidate(scored):
    """依次按 validation 平均相对误差、预算偏差绝对值、rank 字典序选择。

    仅使用调用者提供的验证误差；误差采用精确浮点相等作为并列条件。
    """
    scored = list(scored)
    if not scored:
        raise ValueError("No scored validation candidates")
    for item in scored:
        if not math.isfinite(item["mean_relative_error"]) or item["mean_relative_error"] < 0:
            raise ValueError("Validation relative error must be finite and nonnegative")
    return min(scored, key=lambda item: (item["mean_relative_error"],
                                       abs(item["budget_deviation"]),
                                       _rank_key(item["rank"])))


def _validate_rank(method, rank, shape):
    h, w, t = shape
    values = _rank_key(rank)
    if any(type(n) is not int or n < 1 for n in values):
        raise ValueError("Ranks must be positive integers")
    if method == "svd":
        if not isinstance(rank, int) or rank > min(h * w, t):
            raise ValueError("Invalid matrix SVD rank")
    elif method == "tucker":
        if len(values) != 3 or any(r > n for r, n in zip(values, shape)):
            raise ValueError("Invalid Tucker dimensions")
        a, b, c = values
        if a > b * c or b > a * c or c > a * b:
            raise ValueError("Redundant Tucker multilinear ranks")
    elif method == "tr_dense":
        if len(values) != 4 or values[0] != values[-1] or min(values) < 2:
            raise ValueError("Ordinary TR requires a closed rank vector with every bond >= 2")
        a, b, c, _ = values
        if a * c > min(t, h * w) or b > min(h * a, w * c):
            raise ValueError("TR-SVD rank violates rotated unfolding dimensions")
    else:
        dims = _time_shape(shape) if method == "tt_time" else shape
        if len(values) != len(dims) + 1 or values[0] != 1 or values[-1] != 1:
            raise ValueError("Invalid TT rank vector")
        for i in range(1, len(dims)):
            if values[i] > min(values[i - 1] * dims[i - 1], math.prod(dims[i:])):
                raise ValueError("TT rank would be clipped by TensorLy")


def decompose(tensor, spec):
    """使用标准分解重构共同张量，并核对实际因子数量与预登记预算。

    普通 TR 显式旋转张量及闭环 rank，规避 TensorLy 0.9 非零 mode 的
    非均匀 rank 重排问题；backend context 退出时恢复调用前全局后端。
    """
    import torch
    import tensorly as tl
    from tensorly.decomposition import tensor_ring, tensor_train, tucker
    from tensorly.tr_tensor import tr_to_tensor
    from tensorly.tt_tensor import tt_to_tensor
    from tensorly.tucker_tensor import tucker_to_tensor

    method = spec["method"]
    if method not in METHODS:
        raise ValueError(f"Unknown structure: {method}")
    shape = _shape(spec.get("shape", tuple(tensor.shape)))
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
        raise ValueError("Input tensor shape differs from candidate shape")
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise ValueError("Decomposition requires finite floating-point input")
    rank = spec["rank"]
    if method == "tt_time" and isinstance(rank, int):
        rank = _tt_cap_ranks(_time_shape(shape), rank)
    _validate_rank(method, rank, shape)
    equivalence = None
    with torch.no_grad(), tl.backend_context("pytorch"):
        if method == "svd":
            u, s, vh = torch.linalg.svd(tensor.reshape(shape[0] * shape[1], shape[2]),
                                       full_matrices=False)
            factors = [u[:, :rank], s[:rank], vh[:rank]]
            reconstructed = ((factors[0] * factors[1]) @ factors[2]).reshape(shape)
            actual_rank = rank
        elif method == "tucker":
            core, matrices = tucker(tensor, rank=list(rank), init="svd",
                                    svd="truncated_svd", n_iter_max=100, tol=1e-4)
            factors = [core, *matrices]
            actual_rank = list(core.shape)
            reconstructed = tucker_to_tensor((core, matrices))
            if rank[0] == shape[0] and rank[1] == shape[1]:
                equivalence = f"Full spatial Tucker subspaces represent matrix rank <= {rank[2]}"
        elif method == "tr_dense":
            a, b, c, _ = rank
            ring = tensor_ring(tensor.permute(2, 0, 1), rank=[c, a, b, c],
                               mode=0, svd="truncated_svd")
            # 还原原始 H,W,T 次序后再计数/记录 rank，避免输出混用两种表示。
            factors = [ring[1], ring[2], ring[0]]
            actual_rank = [int(factor.shape[0]) for factor in factors] + [int(factors[-1].shape[2])]
            reconstructed = tr_to_tensor(ring).permute(1, 2, 0).contiguous()
        else:
            data = tensor.reshape(_time_shape(shape)) if method == "tt_time" else tensor
            factors = list(tensor_train(data, rank=list(rank), svd="truncated_svd"))
            actual_rank = [int(factor.shape[0]) for factor in factors] + [int(factors[-1].shape[2])]
            reconstructed = tt_to_tensor(factors).reshape(shape)
            if method == "tt_dense" and rank[1] == shape[0]:
                equivalence = f"Full first TT bond represents matrix rank <= {rank[2]}"
        actual_parameters = sum(factor.numel() for factor in factors)
        expected = spec.get("parameters", parameter_count(method, rank, shape))
        if actual_parameters != expected or _rank_key(actual_rank) != _rank_key(rank):
            raise ValueError(f"Actual decomposition differs from registered rank/budget: "
                             f"rank={actual_rank}, parameters={actual_parameters}, expected={expected}")
        if tuple(reconstructed.shape) != shape or not bool(torch.isfinite(reconstructed).all()):
            raise ValueError("Decomposition produced nonfinite or incorrectly shaped output")
        norm = torch.linalg.vector_norm(tensor)
        error = torch.linalg.vector_norm(reconstructed - tensor)
        relative_error = float(error / norm.clamp_min(torch.finfo(tensor.dtype).tiny))
    diagnostics = dict(method=method, actual_parameters=actual_parameters,
                       actual_rank=actual_rank, factor_shapes=[list(factor.shape) for factor in factors],
                       equivalence_note=equivalence, relative_error=relative_error,
                       compression_ratio=math.prod(shape) / actual_parameters)
    return reconstructed, diagnostics
