"""使用模拟进程验证调度，不启动 GPU 实验。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from rpcf.exp031 import Task, run_parallel


class SchedulerTests(unittest.TestCase):
    def exercise_failure(self, recoverable):
        tasks = [Task(name, 6, 'tnp') for name in ('bad', 'slow', 'next')]
        done, events, attempts = set(), [], {}
        ticks = [0]

        def start(task, run_dir, gpu):
            attempts[task.task_id] = attempts.get(task.task_id, 0) + 1
            events.append(('start', task.task_id, ticks[0]))
            process = SimpleNamespace(pid=len(events), poll=lambda: (
                None if task.task_id == 'slow' and ticks[0] < 3 else 0))
            return dict(task=task, gpu=gpu, process=process)

        def finish(record, run_dir):
            name = record['task'].task_id
            events.append(('end', name, ticks[0]))
            failed = name == 'bad' and attempts[name] == 1
            if not failed:
                done.add(name)
            return not failed, None, None, None, failed and recoverable

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tnp_single_process.flag').touch()
            with patch('rpcf.exp031.task_complete', side_effect=lambda r, t: t.task_id in done), \
                 patch('rpcf.exp031.start_task', side_effect=start), \
                 patch('rpcf.exp031.finalize_task', side_effect=finish), \
                 patch('rpcf.exp031.gpu_memory_used', return_value=0), \
                 patch('rpcf.exp031.time.sleep', side_effect=lambda _: ticks.__setitem__(0, ticks[0] + 1)):
                if recoverable:
                    run_parallel(tasks, tasks, root, [0, 1])
                else:
                    with self.assertRaisesRegex(RuntimeError, 'Tasks failed'):
                        run_parallel(tasks, tasks, root, [0, 1])
        self.assertLess(events.index(next(e for e in events if e[:2] == ('start', 'next'))),
                        events.index(next(e for e in events if e[:2] == ('end', 'slow'))))
        self.assertEqual(attempts['bad'], 2 if recoverable else 1)

    def test_oom_requeues_without_draining(self):
        self.exercise_failure(True)

    def test_permanent_failure_continues_independent_work(self):
        self.exercise_failure(False)

    def test_handoff_defers_until_old_controller_finishes(self):
        task = Task('old', 6, 'tnp')
        ticks = [0]
        with tempfile.TemporaryDirectory() as directory, \
             patch('rpcf.exp031.task_complete', side_effect=lambda r, t: ticks[0] > 0), \
             patch('rpcf.exp031.reservation_active', side_effect=lambda r: ticks[0] == 0), \
             patch('rpcf.exp031.start_task') as start, \
             patch('rpcf.exp031.time.sleep', side_effect=lambda _: ticks.__setitem__(0, ticks[0] + 1)):
            run_parallel([task], [task], Path(directory), [0], deferred_tasks={'old': (123, 456)})
            start.assert_not_called()

    def test_handoff_reserves_old_gpu_but_uses_free_gpu(self):
        old, new = Task('old', 6, 'tnp'), Task('new', 6, 'tnp')
        ticks, done = [0], set()
        def finish(record, root):
            done.add('new')
            return True, None, None, None, False
        with tempfile.TemporaryDirectory() as directory, \
             patch('rpcf.exp031.task_complete', side_effect=lambda r, t: t.task_id in done or (t.task_id == 'old' and ticks[0] > 1)), \
             patch('rpcf.exp031.reservation_active', return_value=True), \
             patch('rpcf.exp031.gpu_memory_used', return_value=0), \
             patch('rpcf.exp031.finalize_task', side_effect=finish), \
             patch('rpcf.exp031.start_task', return_value=dict(task=new, gpu=1, process=SimpleNamespace(pid=1, poll=lambda: 0))) as start, \
             patch('rpcf.exp031.time.sleep', side_effect=lambda _: ticks.__setitem__(0, ticks[0] + 1)):
            root = Path(directory)
            run_parallel([old, new], [old, new], root, [0, 1], deferred_tasks={'old': (123, 456, 0)})
            start.assert_called_once_with(new, root, 1)


if __name__ == '__main__':
    unittest.main()
