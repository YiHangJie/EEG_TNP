"""EXP-034 的科学协议边界：矩阵覆盖、TR闭环预算、实际分解与EA的PGD-10。"""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rpcf.exp034 import build_tasks, expected_keys, DATASETS, MODELS, SEEDS
from rpcf.exp034_structures import candidates, feasible_ranks, parameter_count, decompose


class ScopeTest(unittest.TestCase):
    def test_requested_scope_and_dependencies(self):
        m=dict(run_dir='/tmp/exp034_scope',datasets=list(DATASETS),models=list(MODELS),seeds=list(SEEDS))
        tasks=build_tasks(m)
        self.assertEqual(len(tasks),140)
        self.assertEqual(sum(len(expected_keys(t)) for t in tasks),275)
        external=[t for t in tasks if t['kind']=='external']
        self.assertEqual(len(external),90)
        self.assertEqual(sum(t['adaptive'] for t in external),5)
        for t in external:
            self.assertEqual(t['classifier_method'],'madry')
            if (t['dataset'],t['model'])!=('thubenchmark','eegnet'):
                self.assertEqual(t['attacks'],['pgd'])
                self.assertFalse(t['adaptive'])
        ids={t['task_id'] for t in tasks}
        self.assertEqual(len(ids),140)
        self.assertTrue(all(set(t['dependencies'])<=ids for t in tasks))
        self.assertTrue(all(t['dataset']=='thubenchmark' and t['model']=='eegnet' for t in tasks if t['kind']!='external'))


class TimeTRTest(unittest.TestCase):
    def test_budgets_and_non_degenerate_ring(self):
        for budget in (25,30):
            specs=candidates(budget)
            self.assertTrue(specs)
            for spec in specs:
                r=spec['rank']
                self.assertEqual(len(r),14)
                self.assertEqual(r[0],r[-1])
                self.assertGreaterEqual(min(r),2)
                self.assertLessEqual(r[0]*r[1],10)
                self.assertLessEqual(abs(spec['budget_deviation']),.05)
                self.assertEqual(parameter_count(r),spec['parameters'])
            changed=candidates(budget);changed[0]['rank'][0]=999
            self.assertNotEqual(candidates(budget)[0]['rank'][0],999)

    def test_actual_small_tr_and_backend_restoration(self):
        import tensorly as tl
        shape=(4,5,8);dims=(4,5,2,2,2)
        rank=feasible_ranks(dims,2,2,4)
        spec=dict(method='tr_time',shape=list(shape),rank=rank,parameters=parameter_count(rank,shape))
        x=torch.arange(160,dtype=torch.float64).reshape(shape)/160
        before=tl.get_backend()
        a,diag=decompose(x,spec)
        b,_=decompose(x,spec)
        self.assertEqual(tl.get_backend(),before)
        self.assertEqual(a.shape,x.shape)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(diag['actual_rank'],rank)
        self.assertEqual(diag['actual_parameters'],spec['parameters'])
        wrong=copy.deepcopy(spec);wrong['parameters']+=1
        with self.assertRaises(ValueError):decompose(x,wrong)


class AdaptiveTest(unittest.TestCase):
    def test_subject_binding_and_partial_resume(self):
        """两个subject给相反梯度；断点读回必须不重算也不交换subject。"""
        from rpcf.exp034_worker import adaptive_predictions
        class SubjectModel(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.));self.subject=None;self.calls=[]
            def set_subject_ids(self,ids):self.subject=ids
            def forward(self,x):
                self.calls.append(self.subject.item())
                sign=self.subject.float()*2-1
                z=x.flatten(1).sum(1)*sign*self.weight
                return torch.stack([z,-z],1)
        model=SubjectModel()
        ctx=SimpleNamespace(clean=torch.zeros(2,1,2,2),labels=torch.tensor([0,0]),indices=[11,29])
        task=dict(seed=42,task_id='raw_subject_test')
        with tempfile.TemporaryDirectory() as tmp:
            ap,diag=adaptive_predictions({},task,ctx,model,Path(tmp),'identity','ea',torch.tensor([0,1]))
            self.assertEqual(model.calls[:11],[0]*11)
            self.assertEqual(model.calls[11:],[1]*11)
            self.assertEqual([d['source_index'] for d in diag],[11,29])
            self.assertTrue(all(abs(d['linf_max']-.03)<1e-6 for d in diag))
            count=len(model.calls)
            ap2,diag2=adaptive_predictions({},task,ctx,model,Path(tmp),'identity','ea',torch.tensor([0,1]))
            self.assertEqual(len(model.calls),count)
            self.assertEqual(ap2,ap);self.assertEqual(diag2,diag)
            with self.assertRaises(ValueError):
                adaptive_predictions({},task,ctx,model,Path(tmp),'different-source','ea',torch.tensor([0,1]))


if __name__=='__main__':unittest.main()
