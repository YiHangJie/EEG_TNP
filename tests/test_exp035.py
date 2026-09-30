"""EXP-035 的协议、任务覆盖和资源隔离回归；真实CUDA验证另行运行。"""
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from rpcf.exp035 import build_tasks,templates
from rpcf.exp035_worker import training_arguments,cache_arguments
from rpcf.exp035_scheduler import choose
from rpcf.exp033_common import loss_variants


class Exp035Tests(unittest.TestCase):
    def test_new_seed_scope_and_dependency_coverage(self):
        tasks=build_tasks(dict(run_dir='/tmp/exp035_unit',seeds=list(range(47,52)),smoke=False))
        self.assertEqual(len(tasks),345)
        self.assertEqual({t['seed'] for t in tasks},set(range(47,52)))
        counts=Counter(t['kind'] for t in tasks)
        self.assertEqual((counts['train'],counts['attack'],counts['tnp'],counts['cache_rank']),(80,75,95,30))
        by_id={t['task_id']:t for t in tasks}
        self.assertEqual(len(by_id),len(tasks))
        for seed in range(47,52):
            self.assertEqual(len([t for t in tasks if t['seed']==seed and t['kind']=='train']),16)
            self.assertEqual(by_id[f'cache_merge_seed{seed}']['dependencies'],[f'cache_rank_seed{seed}_{r}' for r in (15,20,25,30,35,40)])
            self.assertEqual(by_id[f'train_seed{seed}_default']['dependencies'],[f'cache_merge_seed{seed}'])
        resolved=set()
        while len(resolved)<len(tasks):
            expanded=resolved|{t['task_id'] for t in tasks if set(t['dependencies'])<=resolved}
            self.assertNotEqual(expanded,resolved,'cycle/deadlock in experiment DAG')
            resolved=expanded

    def test_training_keeps_original_scientific_settings(self):
        with tempfile.TemporaryDirectory() as root:
            manifest=dict(run_dir=root,run_id='unit',smoke=False,sample_num=512,ranks=[15,20,25,30,35,40],templates=templates())
            def arg(xs,key): return xs[xs.index('--'+key)+1]
            for variant in loss_variants():
                args=training_arguments(manifest,dict(seed=47,**variant),Path(root)/'attempt001')
                for key,value in dict(seed=47,epochs=100,batch_size=64,eval_batch_size=128,online_at_batch_size=128,
                                      online_at_pgd_steps=10,online_at_step_size=.006,lr=.0001,weight_decay=.0001).items():
                    self.assertEqual(float(arg(args,key)),value)
                for flag in ('--online_madry_at','--all_layers','--static_rank_weights'): self.assertIn(flag,args)
                for key,value in variant['weights'].items(): self.assertEqual(float(arg(args,key)),value)
                self.assertNotIn('--online_train_sample_num',args)
            cache=cache_arguments(manifest,47)
            self.assertEqual(arg(cache,'attack'),'autoattack')
            self.assertEqual(arg(cache,'attack_batch_size'),'16')
            self.assertEqual(arg(cache,'sample_num'),'512')
            self.assertEqual(arg(cache,'ranks'),'15,20,25,30,35,40')

    def test_resource_isolation_and_all_gpu_training(self):
        policy=dict(gpu_ids=list(range(8)),max_workers=24,light_per_gpu=3,min_ram_gib=24,
                    ram_reserve_gib={'train':10,'light':4},gpu_reserve_mib={'train':8192,'light':2048},gpu_margin_mib=1024)
        gpus={g:dict(unknown=g!=7,free_mib=10000,utilization=0) for g in range(8)}
        self.assertEqual(choose(dict(heavy=True),{},gpus,100,policy),7)
        peers={str(i):dict(gpu=7,task=dict(heavy=False)) for i in range(2)}
        self.assertEqual(choose(dict(heavy=False),peers,gpus,100,policy),7)
        self.assertIsNone(choose(dict(heavy=True),peers,gpus,100,policy))
        peers['2']=dict(gpu=7,task=dict(heavy=False))
        self.assertIsNone(choose(dict(heavy=False),peers,gpus,100,policy))
        self.assertIsNone(choose(dict(heavy=False),{},gpus,27,policy))
        self.assertIsNone(choose(dict(heavy=False),{},gpus,100,policy,[7]))
        gpus[7]['unknown']=True
        self.assertIsNone(choose(dict(heavy=False),{},gpus,100,policy))

    def test_ten_seed_statistics_do_not_pool_conditions(self):
        from rpcf.exp033_report import _group_rows
        rows=[dict(group='rank',method='trp_caf',rank=r,seed=s,standard_accuracy=s/100)
              for r in (25,30) for s in range(42,52)]
        groups,errors=_group_rows(rows,list(range(42,52)))
        self.assertFalse(errors);self.assertEqual(len(groups),2)
        self.assertTrue(all(x['n']==10 and x['complete'] for x in groups))
        groups,errors=_group_rows(rows+[rows[0]],list(range(42,52)))
        self.assertTrue(errors);self.assertFalse(all(x['complete'] for x in groups))


if __name__=='__main__': unittest.main()
