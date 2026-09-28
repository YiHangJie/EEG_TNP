"""EXP-033 并行资源隔离、接管身份和可靠退出码验证。"""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from rpcf.parallel_exp033 import (choose_gpu, capture, proc_info, AdoptedProcess,
                                  resources, recorded_exit, finalize, retire, identity_alive)
from rpcf.exp033_common import fingerprint, write_json

POLICY = dict(gpu_ids=list(range(8)), training_gpus=[0,1,2], max_workers=24,
              light_per_gpu=3, min_ram_gib=24, gpu_margin_mib=1024,
              gpu_reserve_mib={'light':2048,'train':8192}, ram_reserve_gib={'light':4,'train':10})


def record(kind='tnp', gpu=3, pid=123, **kw):
    return dict(task=dict(task_id='example', kind=kind),gpu=gpu,pid=pid,**kw)


def devices(free=9861):
    return {g:dict(free_mib=free, utilization=0, unknown=False, usage={}) for g in range(8)}


class ResourceTest(unittest.TestCase):
    def choose(self, kind='tnp', running=None, gpu=None, ram=100, policy=None, **kw):
        return choose_gpu(dict(kind=kind), running or {}, gpu or devices(), ram, policy or POLICY, **kw)

    def test_balances_light_work_and_caps_three_per_gpu(self):
        p={**POLICY, 'gpu_ids':[3,4], 'training_gpus':[3]}
        self.assertEqual(self.choose(running={1:record(gpu=3)},policy=p),4)
        self.assertEqual(self.choose(running={1:record(gpu=3),2:record(gpu=4)},policy=p),3)
        full={i:record(gpu=3 if i<3 else 4) for i in range(6)}
        self.assertIsNone(self.choose(running=full,policy=p))

    def test_train_and_timing_exclusive_both_directions(self):
        p={**POLICY,'gpu_ids':[0],'training_gpus':[0]}
        for heavy in ('train','timing'):
            self.assertIsNone(self.choose(heavy,running={1:record(gpu=0)},policy=p))
            self.assertIsNone(self.choose(running={1:record(heavy,gpu=0)},policy=p))
            self.assertEqual(self.choose(heavy,policy=p),0)

    def test_existing_training_outside_pool_counts_towards_limit(self):
        running={i:record('train',i+3) for i in range(3)}
        self.assertIsNone(self.choose('train',running=running))

    def test_reserved_training_pool_drains_without_preemption(self):
        self.assertEqual(self.choose(training_ready=True),3)
        self.assertEqual(self.choose('train',running={1:record(gpu=0)},training_ready=True),1)
        self.assertEqual(self.choose('timing',training_ready=True),0)

    def test_oom_retry_isolated_in_both_directions(self):
        p={**POLICY,'gpu_ids':[3],'training_gpus':[3]}
        self.assertIsNone(self.choose(running={1:record(gpu=3)},policy=p,isolated=True))
        self.assertIsNone(self.choose(running={1:record(gpu=3,isolated=True)},policy=p))

    def test_cpu_ram_gpu_memory_and_unknown_context_guards(self):
        self.assertIsNone(self.choose(ram=27.9))
        self.assertIsNone(self.choose('train',ram=33.9))
        self.assertIsNone(self.choose(gpu=devices(3071)))
        g=devices()
        for d in g.values():d['unknown']=True
        self.assertIsNone(self.choose(gpu=g))
        self.assertIsNone(self.choose(running={i:record(gpu=i%8) for i in range(24)}))

    def test_high_utilization_stops_new_sharing(self):
        p={**POLICY,'gpu_ids':[3],'training_gpus':[3]}
        g=devices();g[3]['utilization']=95
        self.assertIsNone(self.choose(gpu=g,policy=p,running={1:record(gpu=3)}))

    def test_unallocated_startup_resources_are_reserved(self):
        running={'first':record(gpu=3,pid=123)}
        answers=['3, GPU-test, 9800, 0\n','GPU-test, 123, 400\n']
        def read(path,*args,**kw):
            name=str(path)
            if name=='/proc/123/status':return 'VmRSS: 1048576 kB\n'
            if name=='/proc/meminfo':return 'MemAvailable: 104857600 kB\n'
            raise AssertionError(name)
        with patch('rpcf.parallel_exp033.owned_pids',return_value={123:'first'}), patch('pathlib.Path.read_text',read), patch('rpcf.parallel_exp033.subprocess.check_output',side_effect=answers):
            g,ram=resources(running,POLICY)
        self.assertEqual(g[3]['free_mib'],9800-(2048-400))
        self.assertEqual(ram,97)
        self.assertFalse(g[3]['unknown'])


class HandoffTest(unittest.TestCase):
    def test_wrong_controller_identity_never_receives_signal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';root.mkdir();folder=root/'parallel_v1';folder.mkdir()
            fake=dict(start_ticks=3,command=[b'python',b'unrelated'],pid=77)
            with patch('rpcf.parallel_exp033.proc_info',return_value=fake),patch('rpcf.parallel_exp033.os.kill') as kill:
                with self.assertRaises(ValueError):capture(root,folder,[],77,3,[0])
                kill.assert_not_called()

    def test_capture_race_resumes_only_old_controller(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';root.mkdir();(root/'status').mkdir();folder=root/'parallel_v1';folder.mkdir()
            original=Path.read_text
            def read(path,*args,**kwargs):
                if str(path)=='/proc/77/wchan':return 'hrtimer_nanosleep'
                if str(path)=='/proc/77/task/77/children':return '88'
                return original(path,*args,**kwargs)
            fake=dict(pid=77,start_ticks=3,command=[b'python',b'rpcf.exp033',b'run'],state='T',pgid=77)
            with patch('rpcf.parallel_exp033.proc_info',return_value=fake),patch('pathlib.Path.read_text',read),patch('rpcf.parallel_exp033.os.kill') as kill:
                with self.assertRaises(RuntimeError):capture(root,folder,[],77,3,[0])
                self.assertEqual([c.args for c in kill.call_args_list],[(77,signal.SIGSTOP),(77,signal.SIGCONT)])
            self.assertFalse((folder/'handoff.json').exists())

    def test_actual_adopted_exit_code_and_pid_reuse_guard(self):
        program="import os,signal,subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.2);raise SystemExit(7)']);print(p.pid,flush=True);os.kill(os.getpid(),signal.SIGSTOP);p.wait()"
        parent=subprocess.Popen([sys.executable,'-u','-c',program],stdout=subprocess.PIPE,text=True)
        try:
            pid=int(parent.stdout.readline());ticks=proc_info(pid)['start_ticks'];adopted=AdoptedProcess(pid,ticks)
            deadline=time.monotonic()+5
            while adopted.poll() is None and time.monotonic()<deadline:time.sleep(.03)
            self.assertEqual(adopted.returncode,7)
            with self.assertRaises(RuntimeError):AdoptedProcess(pid,ticks+1).poll()
            self.assertEqual(proc_info(parent.pid)['state'],'T')
        finally:
            os.kill(parent.pid,signal.SIGTERM);os.kill(parent.pid,signal.SIGCONT);parent.wait(timeout=5);parent.stdout.close()

    def test_retire_acquires_lock_without_signalling_unrelated_pid(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'parallel_v1';folder.mkdir()
            h=dict(controller=dict(pid=999999999,start_ticks=1))
            with (root/'controller.lock').open('a') as lock,patch('rpcf.parallel_exp033.os.kill') as kill:
                retire(h,root,folder,lock);kill.assert_not_called()
                with (root/'controller.lock').open('a') as other:
                    import fcntl
                    with self.assertRaises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.assertTrue((folder/'handoff_complete.json').exists())


class ExitTest(unittest.TestCase):
    def test_signed_worker_exit_and_receipt_identity(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'receipt.json';task=dict(task_id='test',kind='tnp')
            r=dict(task=task,receipt=str(path))
            write_json(path,dict(task_fingerprint=fingerprint(task),returncode=-9))
            self.assertEqual(recorded_exit(r,247),-9)
            write_json(path,dict(task_fingerprint='wrong',returncode=0))
            with self.assertRaises(ValueError):recorded_exit(r,0)

    def test_no_success_without_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            r=dict(task=dict(task_id='x'),receipt=str(Path(d)/'absent'))
            with self.assertRaises(RuntimeError):recorded_exit(r,0)
            self.assertEqual(recorded_exit(r,1),1)
            self.assertEqual(recorded_exit(dict(adopted=True),7),7)

    def test_output_existence_cannot_hide_worker_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'parallel_v1';folder.mkdir()
            task=dict(task_id='test',kind='tnp',output_path=str(root/'result.json'))
            write_json(task['output_path'],dict(task_fingerprint=fingerprint(task)))
            r=dict(task=task,started=time.time(),gpu=0,attempt=1,pid=77,start_ticks=3)
            self.assertFalse(finalize(root,folder,r,7))
            self.assertEqual(json.loads((root/'status/test.json').read_text())['status'],'failed')



class PipelineTest(unittest.TestCase):
    def test_wrapper_preserves_command_and_records_signed_exit(self):
        from rpcf.parallel_exp033 import execute_worker
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);task=dict(task_id='tnp_seed42_rank15',kind='tnp')
            from unittest.mock import Mock
            process=Mock(pid=123);process.wait.return_value=-9
            receipt=root/'receipt.json'
            with patch('rpcf.parallel_exp033.verify_plan',return_value=({},[task])),patch('rpcf.parallel_exp033.subprocess.Popen',return_value=process) as start:
                self.assertEqual(execute_worker(root,task['task_id'],receipt),-9)
                self.assertEqual(start.call_args.args[0],[sys.executable,'-u','-m','rpcf.exp033_worker','--run-dir',str(root),'--task-id',task['task_id']])
            self.assertEqual(json.loads(receipt.read_text())['returncode'],-9)

    def test_real_child_pipeline_finishes_compute_before_timing(self):
        from rpcf.parallel_exp033 import run
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'parallel_v1';folder.mkdir()
            write_json(folder/'handoff.json',dict(controller=dict(pid=999999999,start_ticks=1),workers=[]))
            write_json(folder/'handoff_complete.json',{})
            tasks=[]
            def task(tid,kind,deps=()):
                t=dict(task_id=tid,kind=kind,seed=42,dependencies=list(deps),output_path=str(root/'metrics'/f'{tid}.json'))
                tasks.append(t);return tid
            a=task('train','train');b=task('attack','attack',[a]);task('tnp','tnp',[b]);c=task('structure','structure');task('timing','timing',[c])
            events=[]
            def fake_launch(root,folder,t,gpu,isolated):
                receipt=folder/'receipts'/f"{t['task_id']}.json"
                result=dict(task_fingerprint=fingerprint(t),artifacts=[])
                script=('import json,pathlib,time;time.sleep(.05);'
                        f"p=pathlib.Path({t['output_path']!r});p.parent.mkdir(parents=True,exist_ok=True);p.write_text({json.dumps(result)!r});"
                        f"p=pathlib.Path({str(receipt)!r});p.parent.mkdir(parents=True,exist_ok=True);p.write_text({json.dumps(dict(task_fingerprint=fingerprint(t),returncode=0))!r})")
                process=subprocess.Popen([sys.executable,'-c',script])
                events.append((t['kind'],time.time()))
                return dict(task=t,pid=process.pid,start_ticks=proc_info(process.pid)['start_ticks'],process=process,
                            gpu=gpu,attempt=1,started=time.time(),adopted=False,isolated=isolated,receipt=str(receipt))
            original_sleep=time.sleep
            policy={**POLICY,'gpu_ids':[0,1],'training_gpus':[0],'max_starts_per_tick':2}
            with patch('rpcf.parallel_exp033.launch',side_effect=fake_launch),patch('rpcf.parallel_exp033.resources',return_value=(devices(),100)),patch('rpcf.parallel_exp033.time.sleep',side_effect=lambda s:original_sleep(min(s,.01))),patch('rpcf.exp033_report.summarize') as summary:
                run(root,folder,tasks,policy)
                summary.assert_called_once_with(root,strict=True)
            states={t['task_id']:json.loads((root/'status'/f"{t['task_id']}.json").read_text()) for t in tasks}
            self.assertTrue(all(s['status']=='completed' for s in states.values()))
            timing_start=next(stamp for kind,stamp in events if kind=='timing')
            self.assertGreater(timing_start,max(s['at_epoch'] for tid,s in states.items() if tid!='timing'))
            self.assertEqual(json.loads((root/'runtime.json').read_text())['completed'],5)

    def test_capture_keeps_workers_running_and_preserves_snapshot(self):
        from rpcf.parallel_exp033 import capture
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';folder=root/'parallel_v1';folder.mkdir(parents=True)
            task=dict(task_id='tnp_seed42_rank15',kind='tnp')
            expected=[sys.executable,'-u','-m','rpcf.exp033_worker','--run-dir',str(root),'--task-id',task['task_id']]
            write_json(root/'status'/f"{task['task_id']}.json",dict(status='running',pid=88,gpu=3,attempt=1,command=expected))
            write_json(root/'runtime.json',dict(completed=12,running=[dict(pid=88)]))
            old=dict(pid=77,start_ticks=3,command=[b'python',b'rpcf.exp033',b'run'],state='T',pgid=77)
            child=dict(pid=88,ppid=77,start_ticks=4,command=[x.encode() for x in expected],state='R')
            original=Path.read_text
            def read(path,*args,**kwargs):
                if str(path)=='/proc/77/wchan':return 'hrtimer_nanosleep'
                if str(path)=='/proc/77/task/77/children':return '88'
                return original(path,*args,**kwargs)
            with patch('rpcf.parallel_exp033.proc_info',side_effect=lambda pid:old if pid==77 else child),patch('pathlib.Path.read_text',read),patch('rpcf.parallel_exp033.os.kill') as kill:
                result=capture(root,folder,[task],77,3,[3])
                kill.assert_called_once_with(77,signal.SIGSTOP)
                self.assertEqual(result['workers'][0]['pid'],88)
                self.assertEqual(result['previous_runtime']['completed'],12)


if __name__ == '__main__':
    unittest.main()
