"""只读验收 EXP-031 现有产物，向全新目录输出结果快照。"""
import csv
import gc
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch
torch.set_num_threads(1)
from rpcf.summarize_exp031 import read_tasks, validate_attack, validate_rpcf_history

ROOT = Path('logs/exp031/exp031_full_20260729_174215')
SNAPSHOT = datetime.now()
OUT = Path('docs') / ('EXP031_results_' + SNAPSHOT.strftime('%Y%m%d_%H%M%S'))
OUT.mkdir(exist_ok=False)
TASKS = read_tasks(ROOT / 'planned_tasks.csv')
assert len(TASKS) == len({t['task_id'] for t in TASKS}) == 3073
FIELDS = ('dataset', 'model', 'seed', 'method', 'attack')
METHODS = ('madry', 'rpcf_at', 'trades', 'fbf', 'ea_forward')
MODELS = ('eegnet', 'deepconvnet', 'tsception', 'atcnet', 'conformer', 'tcnet')
DATASETS = ('thubenchmark', 'seediv', 'bciciv2a')
ATTACKS = ('autoattack', 'fgsm', 'pgd', 'cw')
errors, warnings, audit, values, bpda = [], [], [], [], []
complete = set()

def write_csv(name, rows):
    if not rows:
        (OUT / name).write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def load(path):
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)

def key(t):
    return tuple(t[x] for x in FIELDS)

def validate_tnp_sampled(t, p, a):
    """全量核对协议、索引、标签与输入形状；不读取输入信号内容。"""
    required = {'clean','adversarial','clean_pur_by_rank','adv_pur_by_rank',
                'labels','source_indices','ranks','metrics','raw_subset_metrics','meta'}
    assert not required-set(p), 'missing TNP fields'
    assert list(p['ranks']) == [25,30]
    for k in ('dataset','model','seed'):
        assert p['meta'][k] == t[k], f'TNP {k} mismatch'
    assert p['meta']['fold'] == 0
    positions = torch.as_tensor(p['meta']['selected_positions'],dtype=torch.long)
    assert positions.numel() == len(p['labels']) and len(set(positions.tolist())) == positions.numel()
    assert positions.min() >= 0 and positions.max() < len(a['labels'])
    assert list(p['source_indices']) == [a['source_indices'][i] for i in positions.tolist()]
    assert torch.equal(p['labels'].long(),a['labels'].index_select(0,positions).long()), 'label mismatch'
    assert tuple(p['clean'].shape) == tuple(p['adversarial'].shape)
    # 更新快照仅核对协议、索引、标签和形状，不重读大体积输入信号。
    for m in p['metrics']:
        for k in ('purified_clean_accuracy','purified_adv_accuracy'):
            assert 0 <= float(m[k]) <= 1
    for k in ('standard_accuracy','robust_accuracy'):
        assert 0 <= float(p['raw_subset_metrics'][k]) <= 1

def add(t, scope, rank, metric, value):
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError(f'invalid accuracy: {value}')
    values.append({**{x: t[x] for x in FIELDS}, 'scope': scope, 'rank': rank,
                   'metric': metric, 'value': value, 'artifact': t['output_path']})

for t in TASKS:
    p = ROOT / 'status' / (t['task_id'] + '.json')
    try:
        s = json.loads(p.read_text()) if p.exists() else {}
        state = s.get('status', 'pending')
        exists = Path(t['output_path']).exists()
        if state == 'completed':
            if s.get('returncode') != 0 or not exists:
                raise ValueError('completed task has missing output or nonzero return code')
            complete.add(t['task_id'])
        audit.append({**{k: t[k] for k in ('task_id', 'kind', *FIELDS)},
                      **{k: s.get(k) for k in ('returncode','elapsed_seconds','physical_gpu',
                          'actual_batch_size','actual_cache_attack_batch_size','actual_rpcf_batch_size','actual_rpcf_eval_batch_size')},
                      'status': state, 'output_exists': exists, 'artifact': t['output_path']})
    except Exception as exc:
        errors.append({'task_id': t['task_id'], 'error': str(exc)})

print('OUTPUT', str(OUT.resolve()), flush=True)
print('STATUS', dict(Counter(a['status'] for a in audit)), flush=True)
for t in TASKS:
    if t['kind'] == 'rpcf_at' and t['task_id'] in complete:
        try:
            validate_rpcf_history(t)
            prefix = t['command'][t['command'].index('--history_prefix') + 1]
            history = json.loads(Path(prefix + '.json').read_text())
            assert len(history['history']) == 100, 'not 100 training epochs'
            assert [x['epoch'] for x in history['history']] == list(range(1,101))
            for k in ('dataset','model','seed'):
                assert history[k] == t[k], f'training history {k} mismatch'
        except Exception as exc:
            errors.append({'task_id': t['task_id'], 'error': str(exc)})

attack_tasks = {key(t): t for t in TASKS if t['kind'] == 'attack'}
valid_attack = set()
for i, t in enumerate(t for t in TASKS if t['kind'] == 'attack' and t['task_id'] in complete):
    try:
        p = load(t['output_path'])
        m = validate_attack(t, p)
        add(t, 'full_test', 'raw', 'clean', m['clean_accuracy'])
        add(t, 'full_test', 'raw', 'robust', m['adv_accuracy'])
        valid_attack.add(key(t))
        del p
    except Exception as exc:
        errors.append({'task_id': t['task_id'], 'error': str(exc)})
    if (i + 1) % 300 == 0:
        print('ATTACK_VALIDATED', i + 1, flush=True)

identities = {}
for i, t in enumerate(t for t in TASKS if t['kind'] == 'tnp' and t['task_id'] in complete):
    start = len(values)
    try:
        assert key(t) in valid_attack, 'paired attack not validated'
        p = load(t['output_path'])
        a = load(attack_tasks[key(t)]['output_path'])
        validate_tnp_sampled(t, p, a)
        assert len(p['source_indices']) == min(512, len(a['labels'])), 'wrong TNP sample count'
        positions = torch.as_tensor(p['meta']['selected_positions'], dtype=torch.long)
        assert positions.numel() == len(p['labels']) and len(set(positions.tolist())) == positions.numel()
        shape = (len(p['labels']), 2, *p['clean'].shape[1:])
        assert tuple(p['clean_pur_by_rank'].shape) == shape and tuple(p['adv_pur_by_rank'].shape) == shape
        assert len(p['metrics']) == 2 and {int(x['rank']) for x in p['metrics']} == {25, 30}
        ident_key = (t['dataset'], t['model'], t['seed'], t['attack'])
        identity = (list(p['source_indices']), p['labels'].tolist())
        if ident_key in identities:
            assert identities[ident_key] == identity, 'Madry/RPCF subset mismatch'
        identities[ident_key] = identity
        add(t, 'subset_n512', 'raw', 'clean', p['raw_subset_metrics']['standard_accuracy'])
        add(t, 'subset_n512', 'raw', 'robust', p['raw_subset_metrics']['robust_accuracy'])
        for m in p['metrics']:
            add(t, 'subset_n512', int(m['rank']), 'clean', m['purified_clean_accuracy'])
            add(t, 'subset_n512', int(m['rank']), 'robust', m['purified_adv_accuracy'])
        del p, a
    except Exception as exc:
        del values[start:]
        errors.append({'task_id': t['task_id'], 'error': str(exc)})
    if (i + 1) % 20 == 0:
        gc.collect()
        print('TNP_VALIDATED', i + 1, t['dataset'], t['model'], flush=True)

for t in (t for t in TASKS if t['kind'] == 'bpda' and t['task_id'] in complete):
    try:
        p = load(t['output_path'])
        m = p['meta']
        for k in ('dataset', 'model', 'seed'):
            assert m[k] == t[k], f'BPDA {k} mismatch'
        assert m.get('experiment_id') == 'EXP-031'
        assert m.get('rank') == t['rank'] and m.get('pgd_steps') == 10
        assert abs(float(m.get('pgd_alpha', -1)) - 0.006) < 1e-12
        assert abs(float(m.get('eps', -1)) - 0.03) < 1e-12
        for metric in ('purified_clean_accuracy', 'bpda_purified_adv_accuracy'):
            value = float(p['metrics'][metric])
            assert 0 <= value <= 1
            bpda.append({'dataset': t['dataset'], 'model': t['model'], 'seed': t['seed'],
                         'rank': t['rank'], 'metric': metric, 'value': value,
                         'artifact': t['output_path']})
        del p
    except Exception as exc:
        errors.append({'task_id': t['task_id'], 'error': str(exc)})

groupfields = ('dataset', 'model', 'method', 'attack', 'scope', 'rank', 'metric')
groups = defaultdict(list)
for v in values:
    groups[tuple(v[x] for x in groupfields)].append(v)
aggregate = []
for k, items in groups.items():
    seeds = sorted(x['seed'] for x in items)
    full = seeds == [42, 43, 44, 45, 46]
    row = dict(zip(groupfields, k))
    row.update(seed_count=len(items), seeds=','.join(map(str, seeds)), status='complete' if full else 'pending')
    row['mean'] = statistics.mean(x['value'] for x in items) if full else ''
    row['sample_std'] = statistics.stdev(x['value'] for x in items) if full else ''
    aggregate.append(row)

lookup = {(v['dataset'], v['model'], v['seed'], v['method'], v['attack'], v['scope'], v['rank'], v['metric']): v for v in values}
pairs = []
for k, v in lookup.items():
    d, model, seed, method, attack, scope, rank, metric = k
    if method == 'rpcf_at':
        other = lookup.get((d, model, seed, 'madry', attack, scope, rank, metric))
        if other:
            pairs.append({**{x: v[x] for x in groupfields}, 'seed': seed,
                          'comparison': 'rpcf_minus_madry', 'delta_pp': 100 * (v['value'] - other['value'])})
    if rank in (25, 30):
        other = lookup.get((d, model, seed, method, attack, scope, 'raw', metric))
        if other:
            pairs.append({**{x: v[x] for x in groupfields}, 'seed': seed,
                          'comparison': 'tnp_minus_same_subset_raw', 'delta_pp': 100 * (v['value'] - other['value'])})
pgroups = defaultdict(list)
for p in pairs:
    pgroups[tuple(p[x] for x in ('comparison', *groupfields))].append(p)
pair_summary = []
for k, items in pgroups.items():
    if sorted(p['seed'] for p in items) != [42, 43, 44, 45, 46]:
        continue
    nums = [p['delta_pp'] for p in items]
    pair_summary.append({**dict(zip(('comparison', *groupfields), k)), 'seed_count': 5,
                         'mean_delta_pp': statistics.mean(nums), 'sample_std_pp': statistics.stdev(nums),
                         'positive_seeds': sum(x > 0 for x in nums)})

for d in DATASETS:
    for model in MODELS:
        for method in METHODS:
            for seed in range(42, 47):
                xs = [lookup.get((d, model, seed, method, a, 'full_test', 'raw', 'clean')) for a in ATTACKS]
                if all(xs):
                    drift = 100 * (max(x['value'] for x in xs) - min(x['value'] for x in xs))
                    if drift > 1e-8:
                        warnings.append({'dataset': d, 'model': model, 'method': method,
                                         'seed': seed, 'warning': 'raw_clean_differs_across_attack_tasks', 'range_pp': drift})

ag = {tuple(x[f] for f in groupfields): x for x in aggregate}
def cell(d, model, method, attack, scope, rank, metric):
    g = ag.get((d, model, method, attack, scope, rank, metric))
    if not g or g['status'] != 'complete':
        return f"Pending ({g['seed_count'] if g else 0}/5)"
    return f"{100*g['mean']:.2f} ± {100*g['sample_std']:.2f}"

lines = ['# EXP-031 现有结果报告', '', f'任务状态快照开始时间：{SNAPSHOT.isoformat(timespec="seconds")}；run：`{ROOT.name}`。统计固定使用开始时完成的任务，不混入报告生成期间新增完成项。', '',
         '所有准确率均为五种子（42–46）均值 ± 样本标准差，单位 %，fold0。未满五种子的格子显示 Pending(n/5)，不以部分种子均值替代完整结果。', '',
         '未净化为完整测试集；净化为确定性最多 n512 子集，两者不能直接相减。Clean 列统一取 AutoAttack 产物记录。普通 clean 训练模型及其 TNP 未纳入本实验；TRADES/FBF/EA-forward 的 TNP 也未纳入。', '',
         '常规攻击针对各自分类器生成，不等于对净化—分类链路的自适应攻击。AA/FGSM/PGD 的 Linf epsilon=0.03；PGD200 alpha=2/255、无随机起点。CW 为 L2、200步、c=10000、kappa=1、lr=0.1，不受0.03的Linf约束。', '',
         '本次更新复核已保存产物的协议字段、状态、训练记录、样本索引、标签及张量形状；没有重新核对clean/adv输入信号内容，也未重新执行分类推理或攻击。该检查不等同于独立复现实验；前一快照的信号抽查结论不自动覆盖新增产物。', '',
         '## 完成范围', '', '| 数据集 | 类型 | 完成 | 失败 | 其他/待执行 |', '|---|---|---:|---:|---:|']
for d in DATASETS:
    for kind in ('train_standard','train_ea','rpcf_cache','rpcf_at','attack','tnp','bpda'):
        c = Counter(a['status'] for a in audit if a['dataset'] == d and a['kind'] == kind)
        if c:
            lines.append(f"| {d} | {kind} | {c['completed']} | {c['failed']} | {sum(c.values())-c['completed']-c['failed']} |")
lines += ['', f'产物校验错误：{len(errors)}；raw Clean 跨攻击记录差异：{len(warnings)} 条。具体见 audit_errors.json 与 warnings.csv。', '',
          '## 结果解释边界', '',
          '- 完成状态不等于方法有效；低准确率和大种子方差均保留原值，不剔除失败表现的种子。',
          '- 比较 RPCF 额外收益使用 rpcf_minus_madry；比较净化收益使用同一 n512 子集的 tnp_minus_same_subset_raw，见 paired_summary.csv。',
          '- 当前仅做描述性统计，没有显著性检验或认证鲁棒性结论；完整矩阵未完成前不宣称跨数据集普遍优势。', '',
          '## 完整方法对比表']
for d in DATASETS:
    for model in MODELS:
        lines += ['', f'### {d} / {model}', '', '未净化：完整测试集。', '',
                  '| 方法 | Clean | AutoAttack | FGSM | PGD-200 | CW-L2 |', '|---|---:|---:|---:|---:|---:|']
        for method in METHODS:
            cs = [cell(d,model,method,'autoattack','full_test','raw','clean')] + [cell(d,model,method,a,'full_test','raw','robust') for a in ATTACKS]
            lines.append('| ' + method + ' | ' + ' | '.join(cs) + ' |')
        lines += ['', '净化：确定性 n512 子集。', '', '| 方法 | 秩 | Clean | AutoAttack | FGSM | PGD-200 | CW-L2 |', '|---|---:|---:|---:|---:|---:|---:|']
        for rank in (25,30):
            for method in ('madry','rpcf_at'):
                cs = [cell(d,model,method,'autoattack','subset_n512',rank,'clean')] + [cell(d,model,method,a,'subset_n512',rank,'robust') for a in ATTACKS]
                lines.append(f'| {method}+TNP | {rank} | ' + ' | '.join(cs) + ' |')
        lines.append('| clean-only+TNP | 25/30 | 未纳入 | 未纳入 | 未纳入 | 未纳入 | 未纳入 |')
lines += ['', '## BPDA+PGD-10', '', '仅 THUBenchmark/EEGNet/RPCF_AT，identity-BPDA；无 Madry BPDA 对照，不外推其他条件。', '', '| 秩 | 净化 Clean | BPDA robust |', '|---:|---:|---:|']
for rank in (25,30):
    cells=[]
    for metric in ('purified_clean_accuracy','bpda_purified_adv_accuracy'):
        b=[x for x in bpda if x['rank']==rank and x['metric']==metric]
        cells.append(f"{100*statistics.mean(x['value'] for x in b):.2f} ± {100*statistics.stdev(x['value'] for x in b):.2f}" if sorted(x['seed'] for x in b)==[42,43,44,45,46] else f'Pending ({len(b)}/5)')
    lines.append(f'| {rank} | ' + ' | '.join(cells) + ' |')
write_csv('task_audit.csv', audit)
write_csv('conditions_long.csv', values)
write_csv('five_seed_summary.csv', aggregate)
write_csv('paired_by_seed.csv', pairs)
write_csv('paired_summary.csv', pair_summary)
write_csv('bpda_by_seed.csv', bpda)
write_csv('warnings.csv', warnings)
(OUT/'audit_errors.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
(OUT/'manifest.json').write_text(json.dumps({'run':str(ROOT),'snapshot_started_at':SNAPSHOT.isoformat(),'snapshot_completed_at':datetime.now().isoformat(),
    'task_counts':dict(Counter(a['status'] for a in audit)), 'validated_attack_count':len(valid_attack),
    'accuracy_records':len(values),'errors':len(errors),'warnings':len(warnings),
    'tnp_input_check':'all indices/labels and tensor shapes; no input signal content comparison in this refresh',
    'full_experiment_complete':len(complete)==len(TASKS) and not errors}, indent=2), encoding='utf-8')
print('REPORT_COMPLETE', str(OUT.resolve()), 'errors',len(errors),'warnings',len(warnings),flush=True)
