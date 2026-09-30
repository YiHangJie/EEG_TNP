"""EXP-034 时间张量化 TR-SVD；保留 EXP-033 的预处理和参数预算。

将共同 H×W×T 表示中的时间轴二进制张量化，使用闭环 TR 而非 TT。
首核固定为 H，所有闭环 bond 至少为 2，不采用 rank-1 开环退化。
"""
import math
from functools import lru_cache

from rpcf.exp033_structures import BUDGETS, SHAPE, _time_shape, _nondominated


def feasible_ranks(dims, closure, first, cap):
    """按 TR-SVD 每次 unfolding 的行/列上限预先约束 rank，禁止运行时静默裁剪。"""
    if min(closure, first, cap) < 2 or closure * first > min(dims[0], math.prod(dims[1:])):
        raise ValueError("Infeasible first TR unfolding")
    ranks = [closure, first]
    for i in range(1, len(dims) - 1):
        ranks.append(min(cap, ranks[-1] * dims[i], math.prod(dims[i+1:]) * closure))
    return ranks + [closure]


def parameter_count(rank, shape=SHAPE):
    dims = _time_shape(shape)
    if len(rank) != len(dims)+1 or rank[0] != rank[-1] or min(rank) < 2:
        raise ValueError("Time TR requires a closed, non-degenerate rank vector")
    return sum(rank[i]*n*rank[i+1] for i, n in enumerate(dims))


@lru_cache(maxsize=None)
def _candidates(budget, shape):
    dims, target = _time_shape(shape), BUDGETS[budget]
    specs, seen = [], set()
    for closure in range(2, dims[0]//2+1):
        for first in range(2, dims[0]//closure+1):
            # cap 增大时参数量单调不减；超过上限或rank饱和即可停止。
            previous = None
            for cap in range(max(closure, first), math.prod(dims)+1):
                rank = feasible_ranks(dims, closure, first, cap)
                count = parameter_count(rank, shape)
                if count > target*1.05 or rank == previous:
                    break
                previous = rank
                if count >= target*.95 and tuple(rank) not in seen:
                    seen.add(tuple(rank))
                    specs.append(dict(method="tr_time", rank=rank, shape=list(shape),
                                      tensorized_shape=list(dims), rank_cap=cap, mode=0,
                                      parameters=count, target_parameters=target,
                                      budget_deviation=count/target-1, budget_exception=False))
    return tuple(_nondominated(specs))


def candidates(budget, shape=SHAPE):
    import copy
    result = copy.deepcopy(list(_candidates(budget, tuple(shape))))
    if not result:
        raise ValueError(f"No feasible time-TR candidate within ±5% for budget {budget}")
    return result


def decompose(tensor, spec):
    """确定性 truncated-SVD，核对实际 bond 和参数数目后还原共同三维表示。"""
    import torch
    import tensorly as tl
    from tensorly.decomposition import tensor_ring
    from tensorly.tr_tensor import tr_to_tensor
    shape, rank = tuple(spec['shape']), spec['rank']
    if spec['method'] != 'tr_time' or tuple(tensor.shape) != shape:
        raise ValueError("Time-TR input/spec mismatch")
    if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
        raise ValueError("Time-TR requires finite floating-point input")
    dims = _time_shape(shape)
    expected = parameter_count(rank, shape)
    if rank[0]*rank[1] > min(dims[0], math.prod(dims[1:])):
        raise ValueError("Invalid first TR unfolding")
    for i in range(1, len(dims)-1):
        if rank[i+1] > min(rank[i]*dims[i], math.prod(dims[i+1:])*rank[0]):
            raise ValueError("TR rank would be clipped")
    with torch.no_grad(), tl.backend_context('pytorch'):
        ring = tensor_ring(tensor.reshape(dims), rank=list(rank), mode=0, svd='truncated_svd')
        actual_rank = [int(x.shape[0]) for x in ring] + [int(ring[-1].shape[2])]
        actual = sum(x.numel() for x in ring)
        if actual_rank != rank or actual != expected or actual != spec['parameters']:
            raise ValueError("Actual time-TR ranks/parameters differ from registered candidate")
        rebuilt = tr_to_tensor(ring).reshape(shape)
        if not torch.isfinite(rebuilt).all():
            raise ValueError("Non-finite time-TR reconstruction")
        error = float((rebuilt-tensor).norm()/tensor.norm().clamp_min(torch.finfo(tensor.dtype).tiny))
    return rebuilt, dict(method='tr_time', actual_rank=actual_rank, actual_parameters=actual,
                         factor_shapes=[list(x.shape) for x in ring], relative_error=error,
                         compression_ratio=math.prod(shape)/actual)
