"""从已核验 CSV 生成四部分专题报告，不读取模型或修改实验状态。

运行：python docs/EXP031_results_20260911_142412/build_focused_report.py
"""

import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEEDS = {42, 43, 44, 45, 46}
DATASETS = ('thubenchmark', 'seediv', 'bciciv2a')
MODELS = ('eegnet', 'deepconvnet', 'tsception', 'atcnet', 'conformer', 'tcnet')
AT = ('madry', 'trades', 'fbf', 'ea_forward')
ATTACK_SETS = {
    'PGD': ('pgd',),
    'Mean4': ('autoattack', 'fgsm', 'pgd', 'cw'),
    'MeanLinf3': ('autoattack', 'fgsm', 'pgd'),
    'Clean': ('autoattack',),
}
INDEX = {}
for row in csv.DictReader((ROOT / 'conditions_long.csv').open()):
    key = tuple(row[k] for k in ('dataset', 'model', 'method', 'scope', 'rank', 'attack', 'metric'))
    seed = int(row['seed'])
    value = float(row['value'])
    assert seed in SEEDS and 0 <= value <= 1
    assert seed not in INDEX.setdefault(key, {}), (key, seed)
    INDEX[key][seed] = value

DETAIL = []
SUMMARY = []


def scores(dataset, model, method, scope, rank, endpoint):
    """只保留该终点要求的全部攻击都有结果的种子，先做种子内平均。"""
    metric = 'clean' if endpoint == 'Clean' else 'robust'
    groups = [INDEX.get((dataset, model, method, scope, str(rank), attack, metric), {})
              for attack in ATTACK_SETS[endpoint]]
    seeds = SEEDS.intersection(*(set(group) for group in groups))
    return {seed: 100 * statistics.mean(group[seed] for group in groups) for seed in sorted(seeds)}


def difference(left, right):
    """所有差值使用同种子配对，输入单位为百分点。"""
    return {seed: left[seed] - right[seed] for seed in sorted(left.keys() & right.keys())}


def cell(values, section, dataset, model, condition, endpoint, scope, signed=False):
    """未满五种子不输出均值；同时留存机器可读明细供复核。"""
    complete = set(values) == SEEDS
    base = dict(section=section, dataset=dataset, model=model, condition=condition,
                endpoint=endpoint, scope=scope)
    for seed, value in values.items():
        DETAIL.append(dict(base, seed=seed, value_pp=value))
    mean = statistics.mean(values.values()) if complete else ''
    std = statistics.stdev(values.values()) if complete else ''
    SUMMARY.append(dict(base, seed_count=len(values), status='complete' if complete else 'pending',
                        mean_pp=mean, sample_std_pp=std,
                        positive_seeds=sum(v > 0 for v in values.values()) if signed and complete else ''))
    if not complete:
        return f'Pending({len(values)}/5)'
    return f'{mean:+.2f} ± {std:.2f}' if signed else f'{mean:.2f} ± {std:.2f}'


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] + [
        '| ' + ' | '.join(row) + ' |' for row in rows]


def main_table(endpoint, section):
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            cells = [dataset, model]
            for method in AT:
                cells.append(cell(scores(dataset, model, method, 'full_test', 'raw', endpoint),
                                  section, dataset, model, method, endpoint, 'full_test'))
            for rank in (25, 30):
                cells.append(cell(scores(dataset, model, 'rpcf_at', 'subset_n512', rank, endpoint),
                                  section, dataset, model, f'RPCF+TNP-r{rank}', endpoint, 'subset_n512'))
            rows.append(cells)
    return table(['数据集', 'Backbone', 'Madry [F]', 'TRADES [F]', 'FBF [F]', 'EA-forward [F]',
                  'RPCF+TNP r25 [S]', 'RPCF+TNP r30 [S]'], rows)


manifest = json.loads((ROOT / 'manifest.json').read_text())
lines = [
    '# EXP-031 专题结果报告：主实验、跨攻击均值、组件消融与秩敏感性', '',
    f"源快照：{manifest['snapshot_started_at']}；沿用已核验结果，不纳入此后完成任务。", '',
    f"本次任务完成 {manifest['task_counts'].get('completed', 0)}/3073；相比上一版2731项完成新增 {manifest['task_counts'].get('completed', 0)-2731} 项。上一版快照保留在相邻的 EXP031_results_20260910_171036 目录。", '',
    '## 统计口径与阅读限制', '',
    f"- 本次源产物检查错误 {manifest['errors']} 项，警告 {manifest['warnings']} 项；检查协议、状态、训练记录、索引、标签和形状，未比较输入信号内容，也未重新执行攻击或推理。前一快照的信号抽查不自动覆盖新增产物。",
    '- 准确率为五种子（42–46）、fold0 的均值 ± 样本标准差，单位 %；差值单位为百分点（pp）。未满五种子仅显示 Pending(n/5)。',
    '- [F] 表示完整测试集，[S] 表示确定性最多 512 样本子集。主实验现有普通 AT 为 [F]，RPCF+TNP 为 [S]；该表是现有记录并列展示，**不是严格同样本范围的排名，不能直接将列间差值解释为方法增益**。普通 AT 的统一子集结果尚未整理，正式主表需要补齐同口径评估。',
    '- 普通 AT 指不带 TNP 的 Madry、TRADES、FBF、EA-forward；普通 clean-only 模型未运行。RPCF 未净化及 Madry+TNP 留在组件对照中。',
    '- 常规攻击针对各自分类器生成，非净化链路的自适应攻击；跨模型对照不是固定同一对抗输入的因果实验。',
    '- PGD 为 Linf ε=0.03、200步、步长2/255、无随机起点；AA/FGSM 同为 Linf ε=0.03。CW 为 L2、200步、c=10000、kappa=1、lr=0.1，不受该 Linf ε 约束。',
    '- Mean4：每个种子先计算 (AA+FGSM+PGD+CW)/4，再计算五种子均值与标准差；MeanLinf3 同理，仅包含 AA/FGSM/PGD。标准差不是各攻击标准差的平均。',
    '- Mean4 是异质攻击集合的描述性均值，不是等强度攻击指标，也不是逐样本最坏攻击准确率。任何攻击缺失，该种子不进入该均值；不按已完成攻击数动态改变分母。',
    '- 不跨数据集或 backbone 混合求一个总均值；rank25/30 同时报告，未根据测试表现挑选最优秩。', '',
    '## 1. 主实验：RPCF+TNP 与普通 AT（PGD-200）', '',
    '所有 18 个数据集—backbone 条件均列出。请保留表头 [F]/[S] 标签及上述样本范围限制。', '',
]
lines += main_table('PGD', 'main_pgd')
lines += ['', '## 2. 跨攻击平均：RPCF+TNP 与普通 AT', '',
          '### 2.1 四攻击等权平均 Mean4', '',
          '包括 AA、FGSM、PGD-200、CW-L2；本表仍存在 [F]/[S] 口径差异。', '']
lines += main_table('Mean4', 'main_mean4')
lines += ['', '### 2.2 同范数攻击均值 MeanLinf3', '',
          '去除 CW-L2 后的敏感性对照，用于避免结论仅由混合范数平均驱动；三种攻击共享 Linf 预算，但算法与计算强度不同。', '']
lines += main_table('MeanLinf3', 'main_linf3')
lines += ['', '## 3. RPCF+TNP 内部消融：现有组件对照', '',
          '以下四组均使用 [S]，不混入完整测试集数值：A=Madry，B=RPCF_AT（无 TNP），C=Madry+TNP，D=RPCF_AT+TNP。A/B 的 raw 指同一 n512 子集上的未净化分类器结果，不是主表的完整测试集 [F] 数值。', '',
          '同种子、同攻击类型、同测试索引范围进行比较；攻击分别针对各自模型生成。B/D 含额外 RPCF 适配训练，A/C 未提供等训练预算的额外微调对照，因此这是组件级经验对照，不是排除训练预算混杂的严格机制归因。', '',
          'EXP-031 实际仅采用全层微调、训练秩15/20/25/30/35/40静态均匀权重1/6、无特征损失的配置；RPCF 额外训练100 epochs。未实际运行的层选择、特征损失、动态/静态多秩等内部机制变体不补造数值；本节不能代替这些机制消融。协议来源见 [实验记录](../EXPERIMENTS.md)。', '',
          '差值：B−A 为未净化适配差异；D−C 为净化后适配差异；D−B 为 RPCF 模型上的净化前后变化。所有差值均逐种子计算，再求均值和标准差。', '']
for endpoint_index, endpoint in enumerate(('PGD', 'Mean4')):
    for rank_index, rank in enumerate((25, 30)):
        lines += [f'### 3.{endpoint_index * 2 + rank_index + 1} {endpoint} / rank{rank}', '']
        rows = []
        for dataset in DATASETS:
            for model in MODELS:
                groups = [scores(dataset, model, method, 'subset_n512', r, endpoint)
                          for method, r in [('madry', 'raw'), ('rpcf_at', 'raw'), ('madry', rank), ('rpcf_at', rank)]]
                cells = [dataset, model]
                for label, group in zip(('A', 'B', 'C', 'D'), groups):
                    cells.append(cell(group, 'ablation', dataset, model, f'{label}-r{rank}', endpoint, 'subset_n512'))
                for label, left, right in [('B−A', 1, 0), ('D−C', 3, 2), ('D−B', 3, 1)]:
                    cells.append(cell(difference(groups[left], groups[right]), 'ablation_delta', dataset,
                                      model, f'{label}-r{rank}', endpoint, 'subset_n512', signed=True))
                rows.append(cells)
        lines += table(['数据集', 'Backbone', 'A:Madry', 'B:RPCF', 'C:Madry+TNP', 'D:RPCF+TNP',
                        'B−A (pp)', 'D−C (pp)', 'D−B (pp)'], rows) + ['']
lines += ['## 4. RPCF+TNP rank25 与 rank30：测试净化秩敏感性', '',
          'rank25/30 是测试阶段 TNP 净化秩，不是两种独立训练的 RPCF 配置。正差值表示 rank30 更高；不能只依据鲁棒性忽略 Clean 保真度。Clean 固定使用 AA 产物对应 clean 输入记录。', '']
for endpoint_index, endpoint in enumerate(('Clean', 'PGD', 'Mean4', 'MeanLinf3'), start=1):
    lines += [f'### 4.{endpoint_index} {endpoint}', '']
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            left = scores(dataset, model, 'rpcf_at', 'subset_n512', 25, endpoint)
            right = scores(dataset, model, 'rpcf_at', 'subset_n512', 30, endpoint)
            delta = difference(right, left)
            rows.append([dataset, model,
                         cell(left, 'rank', dataset, model, 'rank25', endpoint, 'subset_n512'),
                         cell(right, 'rank', dataset, model, 'rank30', endpoint, 'subset_n512'),
                         cell(delta, 'rank_delta', dataset, model, 'rank30−rank25', endpoint, 'subset_n512', True),
                         f'{sum(v > 0 for v in delta.values())}/5' if set(delta) == SEEDS else 'Pending'])
    lines += table(['数据集', 'Backbone', 'rank25', 'rank30', 'r30−r25 (pp)', 'r30更高的种子数'], rows) + ['']

lines += ['## 5. 描述性汇总与结论边界', '']
for endpoint in ('PGD', 'Mean4'):
    for rank in (25, 30):
        rows = [r for r in SUMMARY if r['section'] == 'ablation_delta' and r['endpoint'] == endpoint
                and r['condition'] == f'D−C-r{rank}' and r['status'] == 'complete']
        pos = sum(r['mean_pp'] > 0 for r in rows)
        neg = sum(r['mean_pp'] < 0 for r in rows)
        lines.append(f'- {endpoint}、rank{rank}：完整五种子的同范围 RPCF+TNP−Madry+TNP 对照共 {len(rows)} 组，平均差为正 {pos} 组、为负 {neg} 组、持平 {len(rows)-pos-neg} 组。该计数不是显著性检验，也不代表未完成组合。')
for endpoint in ('Clean', 'PGD', 'Mean4', 'MeanLinf3'):
    rows = [r for r in SUMMARY if r['section'] == 'rank_delta' and r['endpoint'] == endpoint and r['status'] == 'complete']
    pos = sum(r['mean_pp'] > 0 for r in rows)
    neg = sum(r['mean_pp'] < 0 for r in rows)
    lines.append(f'- {endpoint} 的完整 rank 对照有 {len(rows)} 组：rank30 平均更高 {pos} 组，rank25 平均更高 {neg} 组，持平 {len(rows)-pos-neg} 组。')
lines += ['',
          '净化秩在鲁棒表现与 Clean 保真之间存在经验取舍，具体方向以上述当前快照的完整条件计数为准；不能将上一版仅9个完整条件的结论直接外推到本次新增条件。所有 Pending 条件仍不参与上述计数。', '',
          '因此，现有实验支持按数据集、骨干、攻击和净化秩报告条件性结果，不支持仅凭并列主表宣称普遍优于普通 AT。论文主实验仍需统一样本范围；机制消融仍需专门控制变量实验；秩选择应依据独立验证集而不是本报告测试结果。', '',
          '源报告另有 65 条跨攻击 raw Clean 记录差异及部分 EA-forward 异常表现；本报告保留所有种子，未修正或排除这些值。常规攻击结果不替代自适应攻击评估，已有有限 BPDA 范围见源报告。', '',
          '## 文件与复现', '',
          '- [源结果报告](report.md)：原始全指标矩阵、协议及异常说明。',
          '- [专题汇总 CSV](focused_summary.csv)：本文全部单元格的状态、种子数、均值和样本标准差。',
          '- [专题逐种子 CSV](focused_by_seed.csv)：逐种子准确率/差值，统一使用百分数或百分点尺度。',
          '- [生成脚本](build_focused_report.py)：仅从已核验 conditions_long.csv 派生本报告，无第三方依赖。',
          '- [来源校验清单](focused_manifest.json)：输入哈希与派生统计口径。', '',
          '运行：`python docs/EXP031_results_20260911_142412/build_focused_report.py`。运行会重新生成本专题报告及其派生 CSV/清单，不改变源 CSV、训练、攻击、日志或 checkpoint。', '']

for filename, rows in [('focused_summary.csv', SUMMARY), ('focused_by_seed.csv', DETAIL)]:
    with (ROOT / filename).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
(ROOT / 'focused_report.md').write_text('\n'.join(lines), encoding='utf-8')
(ROOT / 'focused_manifest.json').write_text(json.dumps({
    'source_snapshot': manifest['snapshot_started_at'],
    'source_sha256': hashlib.sha256((ROOT / 'conditions_long.csv').read_bytes()).hexdigest(),
    'attack_sets': ATTACK_SETS,
    'seed_requirement': sorted(SEEDS),
    'std_ddof': 1,
    'aggregation': 'within-seed equal attack mean, then five-seed mean/sample std',
    'value_unit': 'accuracy percent or paired percentage points',
    'summary_rows': len(SUMMARY),
    'detail_rows': len(DETAIL),
    'new_experiments_run': False,
}, ensure_ascii=False, indent=2), encoding='utf-8')
print('Generated', ROOT / 'focused_report.md')
print('\n'.join(line for line in lines if line.startswith('- ') and '组' in line))
