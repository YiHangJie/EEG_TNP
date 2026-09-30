"""验收新增五seed并只读合并原五seed；仅完整十seed结果标Complete。"""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import Counter
from pathlib import Path

from rpcf.exp033_common import read_json,write_json,file_hash,fingerprint,WEIGHTS
from rpcf.exp033_report import (_condition,_key,_group_rows,_paired_rows,_write_csv,
                                _validate_identity,_validate_row)


def accepted_old(manifest):
    """原v6已验收预测再次复算，原实验文件只读。"""
    root=Path(manifest['old_run'])
    rows=[];identities={}
    tasks=read_json(root/'tasks.json')
    for task in tasks:
        if task['group'] not in ('rank','loss') or task['kind'] not in ('reference','tnp','attack'): continue
        path=Path(task['output_path']);status=read_json(root/'status'/f"{task['task_id']}.json")
        if status['status']!='completed' or file_hash(path)!=status['output_sha256']:
            raise ValueError(f'Old result no longer accepted: {path}')
        data=read_json(path)
        labels=_validate_identity(data,512,task,identities)
        for row in data['rows']:
            value=_validate_row(row,labels,task,512)
            value.update(source_experiment='EXP-033',source_path=str(path),task_id=task['task_id'])
            rows.append(value)
    if len(rows)!=275: raise ValueError('Expected 275 retained sensitivity seed rows')
    official=list(csv.DictReader((root/'summary/metrics_long.csv').open()))
    lookup={(r['task_id'],r['method'],int(r.get('rank') or 0)):r for r in official if r['group'] in ('rank','loss')}
    for r in rows:
        old=lookup[(r['task_id'],r['method'],r['rank'])]
        if any(abs(r[k]-float(old[k]))>1e-12 for k in ('standard_accuracy','robust_accuracy')):
            raise ValueError('Retained prediction-derived metric differs from accepted CSV')
    return rows


def figures(rows,out):
    """完整十seed时生成独立SA/RA图，1倍只引用默认实验。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    out.mkdir(parents=True,exist_ok=True)
    chart_rows=[]
    def point(group,method,rank,metric,param=None,factor=None):
        selected=[r for r in rows if r['group']==group and r['method']==method and r['rank']==rank]
        if group=='loss':
            selected=[r for r in selected if (r.get('variant')=='default' if factor==1 else
                      r.get('scan_parameter')==param and abs(r['scan_value']-WEIGHTS[param]*factor)<1e-12)]
        selected.sort(key=lambda r:r['seed'])
        assert [r['seed'] for r in selected]==list(range(42,52))
        vals=[r[metric]*100 for r in selected]
        mu,sd=st.mean(vals),st.stdev(vals)
        chart_rows.append(dict(group=group,method=method,rank=rank,metric=metric,parameter=param,multiplier=factor,
                               actual_coefficient=WEIGHTS[param]*factor if param else None,mean_percent=mu,sample_sd_percent=sd,
                               **{f'seed{r["seed"]}_percent':r[metric]*100 for r in selected},
                               sources=[r['source_path'] for r in selected]))
        return mu,sd,vals
    metrics=[('standard_accuracy','Clean accuracy (SA, %)'),('robust_accuracy','Robust accuracy (RA, %)')]
    plots=[('rank_sensitivity',None),*[(name,name) for name in WEIGHTS]]
    for stem,param in plots:
        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        fig.subplots_adjust(left=.075,right=.98,bottom=.23,top=.74,wspace=.25)
        fig.suptitle('Rank sensitivity' if param is None else f'Weight sensitivity: {param}',x=.075,ha='left',fontsize=16,fontweight='bold')
        fig.text(.075,.89,'THUBenchmark / EEGNet / PGD-200 L-infinity 0.03 / S512 / 10 seeds (42-51)',color='#475569')
        for ax,(metric,label) in zip(axes,metrics):
            if param is None:
                xs=[15,20,25,30,35,40]
                ps=[point('rank','trp_caf',r,metric) for r in xs]
                ax.errorbar(xs,[p[0] for p in ps],yerr=[p[1] for p in ps],marker='o',capsize=3,label='TRP+CAF')
                for x,(_,_,vals) in zip(xs,ps): ax.scatter([x+(i-4.5)*.07 for i in range(10)],vals,s=9,alpha=.35,color='#2479B5')
                mu,sd,_=point('rank','caf',0,metric)
                ax.axhline(mu,color='gray',linestyle='--',label='CAF raw reference');ax.axhspan(mu-sd,mu+sd,color='gray',alpha=.1)
                ax.set_xticks(xs);ax.set_xlabel('TRP rank')
            else:
                xs=[0,.5,1,2]
                for vi,(method,rank,name,color) in enumerate([('caf',0,'CAF raw','#6B7280'),('trp_caf',25,'TRP+CAF r25','#2479B5'),('trp_caf',30,'TRP+CAF r30','#D67923')]):
                    ps=[point('loss',method,rank,metric,param,f) for f in xs]
                    ax.errorbar(xs,[p[0] for p in ps],yerr=[p[1] for p in ps],marker='o',capsize=3,label=name,color=color)
                    for x,(_,_,vals) in zip(xs,ps): ax.scatter([x+(i-4.5)*.003+(vi-1)*.012 for i in range(10)],vals,s=8,alpha=.3,color=color)
                ax.set_xticks(xs,[f'{x:g}x\n({x*WEIGHTS[param]:g})' for x in xs]);ax.set_xlabel('Multiplier (actual coefficient)')
                ax.axvline(1,color='gray',linestyle=':',linewidth=.8)
            ax.set_ylabel(label);ax.grid(axis='y',alpha=.25)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=3,loc='upper left',bbox_to_anchor=(.065,.845),frameon=False)
        fig.text(.075,.045,'Mean +/- sample SD (n=10); dots: individual seeds. Weight 1x reuses default runs. Y-axes use data ranges.',fontsize=8.5)
        fig.savefig(out/f'{stem}.png',dpi=180);fig.savefig(out/f'{stem}.pdf');plt.close(fig)
    _write_csv(out/'chart_data.csv',chart_rows)


def summarize(root,strict=False):
    from rpcf.exp035 import verify_plan
    from rpcf.exp033_worker import verify_artifacts
    m,tasks=verify_plan(root)
    old=accepted_old(m)
    new=[];errors=[];pending=[];states=[];identities={}
    for task in tasks:
        tid=task['task_id'];sp=root/'status'/f'{tid}.json'
        state=read_json(sp) if sp.exists() else dict(status='Pending')
        states.append(dict(task_id=tid,kind=task['kind'],seed=task['seed'],status=state['status']))
        if state['status']!='completed': pending.append(tid);continue
        try:
            path=Path(task['output_path'])
            if file_hash(path)!=state['output_sha256']: raise ValueError('Output hash changed')
            data=read_json(path)
            if data['task_fingerprint']!=fingerprint(task) or data['experiment_id']!='EXP-035' or data['smoke']!=m['smoke']:
                raise ValueError('Output provenance differs')
            verify_artifacts(data)
            labels=_validate_identity(data,m['sample_num'],task,identities)
            expected=3 if task['kind']=='reference' else 1 if task['kind']=='attack' else len(task['ranks']) if task['kind']=='tnp' else 0
            if len(data.get('rows',[]))!=expected: raise ValueError('Unexpected metric row count')
            for row in data['rows']:
                value=_validate_row(row,labels,task,m['sample_num'])
                value.update(source_experiment='EXP-035',source_path=str(path),task_id=tid)
                new.append(value)
        except Exception as exc: errors.append(dict(task_id=tid,error=str(exc)))
    rows=new if m['smoke'] else old+new
    expected_seeds=m['seeds'] if m['smoke'] else m['combined_seeds']
    grouped,group_errors=_group_rows(rows,expected_seeds);errors.extend(group_errors)
    expected_conditions={_key(_condition(r)) for r in old}
    if not m['smoke'] and {_key(_condition(r)) for r in rows}!=expected_conditions:
        errors.append('Condition set differs from the retained 55-condition sensitivity protocol')
    complete=not pending and not errors and len(new)==m['expected_new_rows'] and all(g['complete'] for g in grouped)
    if not m['smoke']: complete=complete and len(rows)==550 and len(grouped)==55
    out=root/'summary';out.mkdir(exist_ok=True)
    _write_csv(out/'metrics_new_seeds.csv',new)
    _write_csv(out/'metrics_long.csv',rows)
    _write_csv(out/'metrics_grouped.csv',grouped)
    paired=_paired_rows(rows);_write_csv(out/'paired_differences.csv',paired)
    pg,pe=_group_rows(paired,expected_seeds);errors.extend(pe);_write_csv(out/'paired_grouped.csv',pg)
    if pe: complete=False
    _write_csv(out/'task_states.csv',states)
    report=dict(experiment_id='EXP-035',smoke=m['smoke'],complete=complete,status='Complete' if complete else 'Pending',
                task_count=len(tasks),completed_task_count=sum(x['status']=='completed' for x in states),
                new_seeds=m['seeds'],combined_seeds=expected_seeds,retained_rows=0 if m['smoke'] else len(old),
                new_metric_rows=len(new),expected_new_metric_rows=m['expected_new_rows'],
                metric_rows=len(rows),condition_count=len(grouped),errors=errors,pending_tasks=pending,
                scope='parameter_sensitivity_only',statistic='arithmetic_mean_and_sample_SD_ddof1')
    write_json(out/'report.json',report)
    (out/'README.md').write_text(f'# EXP-035 参数敏感性\n\n状态：{report["status"]}；smoke={m["smoke"]}。\n\n'
        f'新增种子 {m["seeds"]}，新增指标 {len(new)}/{m["expected_new_rows"]}；保留EXP-033的42–46，正式合并目标55条件×10seed。\n'
        'PGD-200 L∞ 0.03；THUBenchmark / EEGNet / fold0 / S512。均值和样本标准差，不跨参数混合。\n'
        'metrics_new_seeds.csv为新增结果；metrics_long.csv为合并结果，source_path保留逐任务来源；未完成不标Complete。\n'
        '完成后figures内有rank和五个CE/KL权重双面板PNG/PDF及绘图数据；本实验不自动改写已交付Excel。\n')
    if complete and not m['smoke']: figures(rows,root/'figures')
    print(json.dumps({k:v for k,v in report.items() if k!='pending_tasks'},ensure_ascii=False),flush=True)
    if strict and not complete: raise RuntimeError('EXP-035 strict acceptance failed')
    return report
