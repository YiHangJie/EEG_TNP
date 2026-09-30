"""EXP-035 新种子依赖准备与原 EXP-033 科学实现的隔离适配。"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from rpcf.exp033_common import (read_json, write_json, file_hash, fingerprint,
                               set_option, config_for_rank, WEIGHTS, ATTACK_PROTOCOL)


def result(root, tid):
    return read_json(Path(root)/'metrics'/f'{tid}.json')


def paths(manifest,seed):
    """仅列出敏感性依赖，不加载无关的 AP、clean-only 或结构对照。"""
    root=Path(manifest['run_dir']); base=root/'prerequisites'/f'seed{seed}'
    def checkpoint(tid):
        p=root/'metrics'/f'{tid}.json'
        return read_json(p)['checkpoint_path'] if p.exists() else str(base/f'{tid}.pending.pth')
    return dict(checkpoint_madry=checkpoint(f'madry_seed{seed}'),
                checkpoint_rpcf_at=checkpoint(f'train_seed{seed}_default'),
                cache=str(base/'cache.pth'), attack_madry=str(base/'madry_pgd.pth'),
                attack_madry_aa=str(base/'madry_autoattack.pth'),
                attack_rpcf_at=str(base/'caf_default_pgd.pth'),
                tnp_rpcf_at=str(base/'caf_default_tnp.pth'),
                canonical_clean_tnp=str(base/'canonical_tnp.pth'))


def configs(manifest,ranks):
    return [config_for_rank(manifest,rank) if manifest['smoke'] else
            f'PTR3d_8_2048_rank{rank}_3d_interpolate.yaml' for rank in ranks]


def command(manifest,seed,name,**overrides):
    """沿用实际命令；只改 seed/路径，隐含有效攻击 batch 由适配器保留。"""
    values=list(manifest['templates'][name]['command'])
    values=set_option(values,'--seed',seed)
    for key,value in overrides.items(): values=set_option(values,'--'+key,value)
    return values


def training_arguments(manifest,task,attempt):
    p=paths(manifest,task['seed'])
    values=command(manifest,task['seed'],'caf',cache_path=p['cache'],checkpoint_path=p['checkpoint_madry'],
                   output_checkpoint=Path(attempt)/'checkpoint.pth',history_prefix=Path(attempt)/'history')[1:]
    for key,value in task['weights'].items(): values=set_option(values,'--'+key,value)
    if manifest['smoke']:
        for key,value in dict(epochs=1,batch_size=2,eval_batch_size=16,online_at_batch_size=2,
                              online_train_sample_num=2,max_cache_batches=1).items():
            values=set_option(values,'--'+key,value)
    return values


def cache_arguments(manifest,seed):
    p=paths(manifest,seed)
    values=command(manifest,seed,'cache',checkpoint_path=p['checkpoint_madry'],output_path=p['cache'],
                   tag=f"{manifest['run_id']}_seed{seed}",sample_num=manifest['sample_num'],
                   configs=','.join(configs(manifest,manifest['ranks'])))
    if manifest['smoke']: values=set_option(values,'--attack_batch_size',2)
    return values[1:]


def legacy_attack(manifest,seed,name,checkpoint,output):
    """新目录仍用原有效batch和同一S512留存顺序，保留完整test评估。"""
    import rpcf.evaluate_attack as engine
    original_resolve=engine.resolve_exp031_attack_batch
    original_compact=engine.compact_exp031_attack_artifact
    argv=sys.argv
    values=command(manifest,seed,name,checkpoint_path=checkpoint,output_path=output)[1:]
    if manifest['smoke']: values=set_option(values,'--sample_num',2)
    # 此逻辑路径仅触发旧函数的S512分支；真实输出仍由output限定在新run目录。
    def compact(clean,adv,labels,indices,path,seed,fold,*extras):
        return original_compact(clean,adv,labels,indices,'ad_data/exp031/exp035_storage_adapter/attack.pth',seed,fold,*extras)
    try:
        engine.resolve_exp031_attack_batch=lambda path,dataset,model,requested: min(requested,manifest['templates']['effective_attack_batch'])
        engine.compact_exp031_attack_artifact=compact
        sys.argv=['rpcf.evaluate_attack',*values]
        engine.main()
    finally:
        engine.resolve_exp031_attack_batch=original_resolve
        engine.compact_exp031_attack_artifact=original_compact
        sys.argv=argv
    return dict(rows=[],attack_path=output,command=['rpcf.evaluate_attack',*values],
                effective_batch_size=manifest['templates']['effective_attack_batch'],
                storage_adapter='same_original_S512_compaction')


def legacy_tnp(manifest,seed,name,checkpoint,attack,output,shared=None):
    import rpcf.evaluate_purification as engine
    values=command(manifest,seed,name,checkpoint_path=checkpoint,attack_path=attack,output_path=output,
                   sample_num=manifest['sample_num'],configs=','.join(configs(manifest,[25,30])))[1:]
    if shared: values=set_option(values,'--shared_clean_path',shared)
    import torch
    from rpcf.exp032_tnp import rng_state,restore_rng
    argv=sys.argv
    original_load,original_save=torch.load,engine.atomic_torch_save
    work=Path(str(output)+'.work').resolve()
    identity=fingerprint(dict(command=values,checkpoint=file_hash(checkpoint),attack=file_hash(attack),
                              shared=file_hash(shared) if shared else None))
    def load_stateful(path,*args,**kwargs):
        value=original_load(path,*args,**kwargs)
        if isinstance(path,(str,Path)) and Path(path).resolve().parent==work:
            if value.get('_exp035_identity')!=identity or '_exp035_rng' not in value:
                raise ValueError('TNP shard lacks matching RNG identity; preserve and use a new run')
            restore_rng(value['_exp035_rng'])
        return value
    def save_stateful(value,path):
        if Path(path).resolve().parent==work:
            value={**value,'_exp035_identity':identity,'_exp035_rng':rng_state()}
        return original_save(value,path)
    try:
        torch.load=load_stateful
        engine.atomic_torch_save=save_stateful
        sys.argv=['rpcf.evaluate_purification',*values,'--keep_work_dir']
        engine.main()
    finally:
        torch.load=original_load
        engine.atomic_torch_save=original_save
        sys.argv=argv
    return dict(rows=[],payload_path=output,command=['rpcf.evaluate_purification',*values])


def audit_sources(manifest,task,ctx,root,device):
    import torch
    from rpcf.exp032_common import load_payload,align_payload
    from rpcf.exp033_common import load_attack
    from rpcf.exp033_worker import tensor_hash
    p=paths(manifest,task['seed'])
    norms={}
    for method in ('madry','rpcf_at'):
        _,norms[method]=load_attack(p[f'attack_{method}'],ctx,p[f'checkpoint_{method}'])
    aa=load_payload(p['attack_madry_aa'])
    for key,value in dict(dataset='thubenchmark',model='eegnet',seed=task['seed'],fold=0,attack='autoattack').items():
        if aa['meta'].get(key)!=value: raise ValueError(f'Canonical AA identity differs: {key}')
    if Path(aa['meta']['checkpoint_path']).resolve()!=Path(p['checkpoint_madry']).resolve():
        raise ValueError('Canonical AA classifier differs')
    if aa['meta']['attack_protocol'].get('eps')!=.03 or aa['meta']['attack_protocol'].get('norm')!='Linf':
        raise ValueError('Canonical AA norm differs')
    aa_aligned=align_payload(aa,ctx.indices,ctx.labels,ctx.clean)
    for key in ('canonical_clean_tnp','tnp_rpcf_at'):
        payload=load_payload(p[key])
        if payload['ranks'] != [25,30]: raise ValueError('Reference ranks differ')
        aligned=align_payload(payload,ctx.indices,ctx.labels,ctx.clean)
        reference='madry' if key=='canonical_clean_tnp' else 'rpcf_at'
        attack_key='attack_madry_aa' if reference=='madry' else 'attack_rpcf_at'
        if (Path(payload['meta']['checkpoint_path']).resolve()!=Path(p[f'checkpoint_{reference}']).resolve()
                or Path(payload['meta']['attack_path']).resolve()!=Path(p[attack_key]).resolve()):
            raise ValueError('TNP classifier/attack provenance differs')
        if key=='canonical_clean_tnp' and not torch.equal(aligned['adversarial'],aa_aligned['adversarial']):
            raise ValueError('Canonical TNP adversarial inputs differ')
        if key=='tnp_rpcf_at':
            adv,_=load_attack(p['attack_rpcf_at'],ctx,p['checkpoint_rpcf_at'])
            if not torch.equal(aligned['adversarial'],adv): raise ValueError('Default TNP attack differs')
    trained=result(root,f"train_seed{task['seed']}_default")
    history=read_json(trained['history_path'])
    if not all(history[k] for k in ('all_layers','static_rank_weights','online_madry_at')) or history['cached_adv_loss_enabled']:
        raise ValueError('Default CAF configuration differs')
    names=dict(clean_ce_weight='clean_ce',pur_ce_weight='pur_ce',adv_pur_ce_weight='adv_pur_ce',
               lambda_pur='pur_kl',lambda_adv_pur='adv_pur_kl')
    if any(history['loss_weights'][names[k]]!=v for k,v in WEIGHTS.items()): raise ValueError('Default weights differ')
    files={}
    for key,path in p.items():
        q=Path(path);stat=q.stat()
        files[key]=dict(path=str(q),size=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=file_hash(q))
    return dict(rows=[],files=files,checks=norms,clean_sha256=tensor_hash(ctx.clean),split=ctx.split)


def execute(root,tid):
    from rpcf.exp035 import verify_plan
    from rpcf.exp033_common import context
    from rpcf.exp033_worker import artifact_records,verify_artifacts,tensor_hash
    import rpcf.exp033_worker as legacy
    import torch
    from utils.reproducibility import seed_everything
    m,tasks=verify_plan(root)
    t=next(t for t in tasks if t['task_id']==tid)
    output=Path(t['output_path'])
    if output.exists():
        old=read_json(output)
        if old['task_fingerprint']!=fingerprint(t): raise ValueError('Existing output identity differs')
        verify_artifacts(old)
        return
    for dep in t['dependencies']: verify_artifacts(result(root,dep))
    torch.set_num_threads(2);seed_everything(t['seed'])
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('EXP-035 workers require a CUDA GPU')
    torch.cuda.set_device(device)  # 首次分配前初始化该worker的CUDA上下文。
    torch.cuda.reset_peak_memory_stats(device)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    seed=t['seed'];kind=t['kind'];p=paths(m,seed)
    base=Path(p['cache']).parent;base.mkdir(parents=True,exist_ok=True)
    legacy.source_paths=paths
    legacy.training_arguments=training_arguments
    ctx=None
    if kind=='prepare':
        ctx=context(m,seed)
        r=dict(rows=[],clean_sha256=tensor_hash(ctx.clean),split=ctx.split)
    elif kind=='madry':
        folder=root/'training'/tid;folder.mkdir(parents=True,exist_ok=True)
        attempt=folder/f"attempt{len(list(folder.glob('attempt*')))+1:03d}";attempt.mkdir()
        tag=f"{m['run_id']}_madry_{attempt.name}"
        values=command(m,seed,'madry',checkpoint_tag=tag)
        if m['smoke']:
            for key,value in dict(epochs=1,patience=1,train_sample_num=2).items(): values=set_option(values,'--'+key,value)
        cmd=[sys.executable,'-u','-m',*values]
        write_json(attempt/'command.json',cmd)
        subprocess.run(cmd,check=True)
        from utils.experiment_artifacts import build_checkpoint_path
        cp=build_checkpoint_path('thubenchmark','eegnet','train_only_subject_no_ea_subject_split','madry',.03,seed,0,tag=tag)
        r=dict(rows=[],checkpoint_path=cp,checkpoint_sha256=file_hash(cp),command=cmd)
    elif kind in ('cache_base','cache_rank','cache_merge'):
        from rpcf.exp035_cache import run_cache
        values=cache_arguments(m,seed)
        mode={'cache_base':'base','cache_rank':'rank','cache_merge':'merge'}[kind]
        if mode=='rank':
            values=set_option(values,'--ranks',t['rank'])
            values=set_option(values,'--configs',configs(m,[t['rank']])[0])
        values.append({'base':'--base_only','rank':'--rank_shard_only','merge':'--finalize_only'}[mode])
        run_cache(values,mode,root/'worker_logs'/f'{tid}.purification.log')
        artifact=Path(p['cache']) if mode=='merge' else Path(p['cache']+'.work')/('base.pth' if mode=='base' else f"rank{t['rank']}.pth")
        r=dict(rows=[],payload_path=str(artifact),command=['rpcf.generate_cache',*values])
    elif kind=='cache_streams':
        from rpcf.exp035_cache import plan_streams
        files=plan_streams(p['cache'],configs(m,m['ranks']),m['ranks'])
        r=dict(rows=[],rng_paths=files)
    elif kind in ('madry_aa','madry_pgd','default_pgd'):
        classifier=p['checkpoint_rpcf_at'] if kind=='default_pgd' else p['checkpoint_madry']
        ap=p[{'madry_aa':'attack_madry_aa','madry_pgd':'attack_madry','default_pgd':'attack_rpcf_at'}[kind]]
        r=legacy_attack(m,seed,kind,classifier,ap)
    elif kind in ('canonical','default_tnp'):
        if kind=='canonical': r=legacy_tnp(m,seed,kind,p['checkpoint_madry'],p['attack_madry_aa'],p['canonical_clean_tnp'])
        else: r=legacy_tnp(m,seed,kind,p['checkpoint_rpcf_at'],p['attack_rpcf_at'],p['tnp_rpcf_at'],p['canonical_clean_tnp'])
    elif kind=='train':
        # 原子任务内原 finetune 原样执行，不在父进程另做shuffle/模型初始化。
        r=legacy.train(m,t,None,root,device)
    else:
        ctx=context(m,seed)
        if kind not in ('sources','attack'): legacy.frozen_sources(root,t,ctx)
        if kind=='sources': r=audit_sources(m,t,ctx,root,device)
        elif kind=='attack':
            # 只修正新产物的实验标签；逐样本PGD和partial RNG仍调用原实现。
            import rpcf.exp031_artifacts as artifacts
            save=artifacts.atomic_torch_save
            def save_tagged(payload,path):
                if isinstance(payload,dict) and payload.get('meta',{}).get('experiment_id')=='EXP-033':
                    payload={**payload,'meta':{**payload['meta'],'experiment_id':'EXP-035'}}
                return save(payload,path)
            try:
                artifacts.atomic_torch_save=save_tagged
                r=legacy.attack(m,t,ctx,root,device)
            finally: artifacts.atomic_torch_save=save
        elif kind=='reference': r=legacy.reference(m,t,ctx,root,device)
        elif kind=='tnp': r=legacy.tnp(m,t,ctx,root,device)
        else: raise ValueError(kind)
    identity=result(root,f'prepare_seed{seed}') if ctx is None else dict(source_indices=ctx.indices,labels=ctx.labels.tolist())
    r.update(experiment_id='EXP-035',task_id=tid,task_fingerprint=fingerprint(t),smoke=m['smoke'],
             dataset='thubenchmark',model='eegnet',seed=seed,fold=0,
             source_indices=identity['source_indices'],labels=identity['labels'],
             peak_memory_bytes=torch.cuda.max_memory_allocated(device))
    r['artifacts']=artifact_records(r)
    extra=[r['payload_path']] if r.get('payload_path') else r.get('rng_paths',[])
    for path in extra:
        q=Path(path);stat=q.stat()
        r['artifacts'].append(dict(path=str(q),size=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=file_hash(q)))
    write_json(output,r)
    print(f'COMPLETE {tid} rows={len(r.get("rows",[]))}',flush=True)
