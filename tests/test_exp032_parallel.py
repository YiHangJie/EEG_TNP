"""并行调度约束、真实子进程退出码及 CPU 诊断设备隔离检查。"""

import os
import fcntl
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import signal
import subprocess
import sys
import time
import unittest

from rpcf.exp031 import Task
from rpcf.parallel_exp032 import AdoptedProcess, choose_gpu, normalise_command, proc_info, acquire_controller_lock, retire_controller, valid_worker_gpu, paused_peer_matches, parse_paused_peers, adopt_records


def record(kind, gpu=0):
    return {"task": Task(kind, 0, kind), "gpu": gpu}


class ParallelSchedulingTest(unittest.TestCase):
    def choose(self, kind, running, free=9000, capacity=2):
        return choose_gpu(Task(kind, 0, kind), dict(enumerate(running)), [0],
                          {0: (free, 10240)}, capacity, 4096)

    def test_conda_shebang_normalisation_preserves_arguments(self):
        planned = ["conda", "run", "-n", "torch", "python", "--seed", "42"]
        actual = ["/env/bin/python", "/env/condabin/conda", *planned[1:]]
        self.assertEqual(normalise_command(actual), normalise_command(planned))
        self.assertNotEqual(normalise_command([*actual[:-1], "43"]), normalise_command(planned))

    def test_capture_accepts_cpu_workers_but_rejects_unauthorized_gpu(self):
        self.assertTrue(valid_worker_gpu(Task("toy", 0, "toy"), -1, [0, 6, 7]))
        self.assertFalse(valid_worker_gpu(Task("attack", 0, "attack"), -1, [0, 6, 7]))
        self.assertFalse(valid_worker_gpu(Task("attack", 0, "attack"), 5, [0, 6, 7]))
        self.assertTrue(valid_worker_gpu(Task("attack", 0, "attack"), 7, [0, 6, 7]))

    def test_shared_gpu_requires_verified_pause_and_free_memory(self):
        kwargs = dict(shared_gpu_ids=[1], shared_ready=[1], shared_min_free_mb=6144)
        def choose(kind, running=None, free=8400):
            return choose_gpu(Task(kind, 0, kind), running or {}, [1], {1: (free, 10240)}, 2, 4096, **kwargs)
        for kind in ("attack", "external_eval", "tnp"):
            self.assertEqual(choose(kind), 1)
            self.assertIsNone(choose(kind, free=6143))
            self.assertIsNone(choose(kind, {1: record("tnp", 1)}))
        self.assertIsNone(choose("train_standard"))
        self.assertIsNone(choose("purifier_train"))
        kwargs["shared_ready"] = []
        self.assertIsNone(choose("attack"))

    def test_shared_peer_resume_pid_reuse_unknown_apps_block_dispatch(self):
        expected = {"pid": 123, "start_ticks": 77}
        stopped = {"start_ticks": 77, "state": "T"}
        self.assertTrue(paused_peer_matches(expected, stopped, {123}, 0))
        self.assertFalse(paused_peer_matches(expected, dict(stopped, state="S"), {123}, 0))
        self.assertFalse(paused_peer_matches(expected, dict(stopped, start_ticks=78), {123}, 0))
        self.assertFalse(paused_peer_matches(expected, stopped, {123, 456}, 0))
        self.assertFalse(paused_peer_matches(expected, stopped, {123}, 10))
        self.assertFalse(paused_peer_matches(expected, None, {456}, 0))
        self.assertTrue(paused_peer_matches(expected, None, set(), 0))
        self.assertEqual(parse_paused_peers("1:123:77", [0, 1]), {1: expected})
        with self.assertRaises(ValueError):
            parse_paused_peers("2:123:77", [0, 1])

    def test_adoption_preserves_existing_double_tnp_oom_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tasks").mkdir()
            task_map = {name: Task(name, 0, "tnp") for name in ("first", "second")}
            handoff = {"workers": [{"task_id": name, "pid": i, "start_ticks": i, "gpu": 0}
                                    for i, name in enumerate(task_map, 1)]}
            with patch("rpcf.parallel_exp032.task_complete", return_value=False):
                running = adopt_records(handoff, task_map, root)
            try:
                self.assertEqual([r["tnp_capacity_at_start"] for r in running.values()], [2, 2])
            finally:
                for r in running.values():
                    r["handle"].close()

    def test_tnp_prefers_idle_authorized_shared_gpu(self):
        chosen = choose_gpu(Task("tnp", 0, "tnp"), {1: record("tnp", 0)}, [0, 1],
                            {0: (9000, 10240), 1: (8400, 10240)}, 2, 4096,
                            shared_gpu_ids=[1], shared_ready=[1])
        self.assertEqual(chosen, 1)

    def test_cpu_toy_does_not_reserve_gpu(self):
        self.assertEqual(self.choose("attack", [record("toy")], free=10222), 0)

    def test_only_two_tnp_may_share(self):
        self.assertEqual(self.choose("tnp", [record("tnp")]), 0)
        self.assertIsNone(self.choose("tnp", [record("tnp"), record("tnp")]))
        self.assertIsNone(self.choose("tnp", [record("tnp")], capacity=1))
        self.assertIsNone(self.choose("tnp", [record("tnp")], free=3000))

    def test_gpu_heavy_tasks_remain_exclusive(self):
        for kind in ("train_standard", "attack", "external_eval", "purifier_train"):
            self.assertIsNone(self.choose(kind, [record("tnp")]))
            self.assertIsNone(self.choose("tnp", [record(kind)]))
        self.assertIsNone(self.choose("attack", [], free=6000))

    def test_real_exit_code_with_paused_parent(self):
        program = """
import os,signal,subprocess,sys
p=subprocess.Popen([sys.executable,'-c','import time,sys; time.sleep(0.5); sys.exit(7)'])
print(p.pid,flush=True)
os.kill(os.getpid(),signal.SIGSTOP)
p.wait()
"""
        parent = subprocess.Popen([sys.executable, "-u", "-c", program], stdout=subprocess.PIPE, text=True)
        try:
            child = int(parent.stdout.readline())
            identity = proc_info(child)
            process = AdoptedProcess(child, identity["start_ticks"])
            wrong = AdoptedProcess(child, identity["start_ticks"] + 1)
            with self.assertRaises(RuntimeError):
                wrong.poll()
            deadline = time.monotonic() + 5
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertEqual(process.returncode, 7)
            self.assertEqual(proc_info(parent.pid)["state"], "T")
        finally:
            os.kill(parent.pid, signal.SIGTERM)
            os.kill(parent.pid, signal.SIGCONT)
            parent.wait(timeout=5)
            parent.stdout.close()

    def test_controller_lock_waits_for_delayed_release(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.lock"
            program = "import fcntl,sys,time; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX); print('locked',flush=True); time.sleep(0.5); f.close()"
            holder = subprocess.Popen([sys.executable, "-u", "-c", program, str(path)], stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(holder.stdout.readline().strip(), "locked")
                with path.open("a") as handle:
                    began = time.monotonic()
                    acquire_controller_lock(handle, timeout=3)
                    self.assertGreater(time.monotonic() - began, 0.2)
            finally:
                holder.wait(timeout=5)
                holder.stdout.close()

    def test_controller_lock_timeout_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.lock"
            with path.open("a") as owner, path.open("a") as contender:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(TimeoutError):
                    acquire_controller_lock(contender, timeout=0.05)

    def test_handoff_recovers_after_old_controller_exited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handoff = {"controller": {"pid": 999999999, "start_ticks": 1}}
            with (root / "controller.lock").open("a") as handle:
                with patch("rpcf.parallel_exp032.reservation_active", return_value=False), patch("rpcf.parallel_exp032.os.kill") as kill:
                    retire_controller(handoff, handle, root)
                    kill.assert_not_called()
                result = json.loads((root / "parallel_handoff_completed_v5.json").read_text())
                self.assertEqual(result["old_controller"], 999999999)
                with (root / "controller.lock").open("a") as second:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_toy_cpu_output_unchanged_when_cuda_hidden(self):
        program = """
import hashlib,torch
from types import SimpleNamespace
from rpcf.core import seed_everything
from rpcf.exp032_toy import statistics
from purify import interpolate
torch.set_num_threads(2)
seed_everything(42)
x=torch.randn(1,64,32)
a=SimpleNamespace(dataset='thubenchmark',config='PTR3d_8_2048_rank25_3d_interpolate.yaml')
t=interpolate(a,x,250)
r,e,energy=statistics(t)
assert t.device.type=='cpu'
print(hashlib.sha256(repr((r,e,energy)).encode()).hexdigest())
"""
        outputs = []
        for visible in ("0", "-1"):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            result = subprocess.run([sys.executable, "-u", "-c", program], env=env, check=True,
                                    capture_output=True, text=True, timeout=60)
            outputs.append(result.stdout.strip().splitlines()[-1])
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
