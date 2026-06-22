#!/usr/bin/env python3
"""只读实验状态面板。

该脚本只读取 logs 下已有的 run 目录、日志和 summary 产物，不启动、
停止或修改任何实验。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ERROR_KEYWORDS = (
    "Traceback",
    "RuntimeError",
    "Exception",
    "out of memory",
    "failed",
    "Error:",
)
FINISHED_KEYWORDS = ("pipeline finished", "rerun finished", "runner complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读查询 logs/*/<run_id> 实验运行状态。"
    )
    parser.add_argument("--logs-root", default="logs", help="日志根目录，默认 logs。")
    parser.add_argument("--group", help="日志组名，例如 exp019、exp018、rpcf。")
    parser.add_argument("--run", default="latest", help="run id，默认 latest。")
    parser.add_argument("--log-root", help="直接指定 run 目录。")
    parser.add_argument("--all", action="store_true", help="列出匹配范围内全部 run。")
    parser.add_argument("--json", action="store_true", help="输出 JSON。")
    parser.add_argument(
        "--stale-minutes",
        type=float,
        default=30.0,
        help="running 进程超过多少分钟无日志更新后标记为 stalled。",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=10,
        help="human 输出中 controller/stage 日志尾部行数。",
    )
    parser.add_argument(
        "--scan-tail-lines",
        type=int,
        default=200,
        help="扫描错误关键词时读取每个日志文件的尾部行数。",
    )
    return parser.parse_args()


def read_tail(path: Path, max_lines: int) -> list[str]:
    if max_lines <= 0 or not path.is_file():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= max_lines:
                step = min(block, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
            text = data.decode("utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-max_lines:]


def parse_run_config(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    config: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    except OSError:
        return {}
    return config


def file_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def format_time(ts: float | None) -> str:
    if ts is None:
        return "unknown"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def age_seconds(ts: float | None) -> float | None:
    if ts is None:
        return None
    return max(0.0, time.time() - ts)


def format_age(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m ago"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f}d ago"


def latest_file_mtime(root: Path) -> float | None:
    latest: float | None = None
    if not root.is_dir():
        return None
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        mtime = file_mtime(path)
        if mtime is not None and (latest is None or mtime > latest):
            latest = mtime
    return latest


def pid_state(pid_path: Path) -> dict[str, Any]:
    state: dict[str, Any] = {
        "path": str(pid_path),
        "exists": pid_path.is_file(),
        "pid": None,
        "running": False,
        "error": None,
    }
    if not pid_path.is_file():
        return state
    try:
        raw = pid_path.read_text(encoding="utf-8", errors="replace").strip()
        pid = int(raw)
        state["pid"] = pid
    except (OSError, ValueError) as exc:
        state["error"] = f"invalid pid file: {exc}"
        return state
    if pid <= 2:
        state["error"] = "pid is too small to be treated as an experiment controller"
        return state
    try:
        os.kill(pid, 0)
        state["running"] = True
    except ProcessLookupError:
        state["running"] = False
    except PermissionError:
        state["running"] = False
        state["error"] = "permission denied while checking pid"
    except OSError as exc:
        state["running"] = False
        state["error"] = str(exc)
    return state


def find_summaries(root: Path) -> list[Path]:
    candidates = [
        root / "summary.json",
        root / "summary.csv",
        root / "five_methods" / "five_methods_summary.json",
    ]
    summaries: list[Path] = []
    for path in candidates:
        if path.is_file():
            summaries.append(path)
    # 兼容 comparison_* 子目录，但避免把任意深层大目录全部扫出来。
    for path in sorted(root.glob("*/summary.json")):
        if path not in summaries:
            summaries.append(path)
    for path in sorted(root.glob("*/summary.csv")):
        if path not in summaries:
            summaries.append(path)
    return summaries


def find_completion_summaries(root: Path) -> list[Path]:
    """只返回能证明整个 run 已完成的最终汇总产物。"""
    candidates = [
        root / "summary.json",
        root / "summary.csv",
        root / "five_methods" / "five_methods_summary.json",
    ]
    return [path for path in candidates if path.is_file()]


def latest_by_mtime(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda path: file_mtime(path) or 0.0)


def find_stage_logs(root: Path) -> tuple[list[Path], str]:
    stage_logs = sorted(root.glob("stage*.log"))
    if stage_logs:
        return stage_logs, "stage"
    phase_logs = sorted(root.glob("phase*.log"))
    if phase_logs:
        return phase_logs, "phase"
    generic = sorted(
        path
        for path in root.glob("*.log")
        if path.name != "controller.log" and path.is_file()
    )
    return generic, "log"


def stage_name(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.stem


def scan_errors(paths: list[Path], max_lines: int) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.is_file() or path in seen:
            continue
        seen.add(path)
        matched_line = None
        for line in read_tail(path, max_lines):
            lower_line = line.lower()
            if any(keyword.lower() in lower_line for keyword in ERROR_KEYWORDS):
                matched_line = line.strip()
        if matched_line:
            errors.append({"path": str(path), "line": matched_line})
    return errors


def has_finished_marker(
    controller_log: Path, run_id: str, max_lines: int
) -> bool:
    for line in read_tail(controller_log, max_lines):
        lower_line = line.lower()
        if "runner complete" in lower_line:
            return True
        if (
            run_id.lower() in lower_line
            and any(keyword in lower_line for keyword in FINISHED_KEYWORDS)
        ):
            return True
    return False


def compact_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path)


def analyze_run(root: Path, stale_minutes: float, scan_tail_lines: int) -> dict[str, Any]:
    root = root.resolve()
    run_config_path = root / "run_config.txt"
    controller_log = root / "controller.log"
    pid_path = root / "controller.pid"
    config = parse_run_config(run_config_path)
    pid = pid_state(pid_path)
    summaries = find_summaries(root)
    completion_summaries = find_completion_summaries(root)
    stage_logs, stage_kind = find_stage_logs(root)
    latest_stage = latest_by_mtime(stage_logs)
    latest_log_candidates = []
    if controller_log.is_file():
        latest_log_candidates.append(controller_log)
    latest_log_candidates.extend(stage_logs)
    latest_log = latest_by_mtime(latest_log_candidates)
    latest_mtime = latest_file_mtime(root)
    latest_log_mtime = file_mtime(latest_log) if latest_log else latest_mtime
    finished_marker = has_finished_marker(
        controller_log, root.name, scan_tail_lines
    )
    error_paths = []
    if controller_log.is_file():
        error_paths.append(controller_log)
    if latest_stage is not None:
        error_paths.append(latest_stage)
    error_paths.extend(stage_logs[-5:])
    errors = scan_errors(error_paths, scan_tail_lines)

    status = "unknown"
    latest_age = age_seconds(latest_log_mtime)
    log_is_active = latest_age is not None and latest_age <= stale_minutes * 60.0
    if finished_marker:
        status = "completed"
    elif pid["running"]:
        if pid["running"] and not log_is_active:
            status = "stalled"
        else:
            status = "running"
    elif completion_summaries:
        status = "completed"
    elif log_is_active and not errors:
        status = "running"
    elif errors:
        status = "failed"

    planned_tasks = root / "planned_tasks.csv"
    result: dict[str, Any] = {
        "run_id": root.name,
        "log_root": str(root),
        "group": root.parent.name,
        "status": status,
        "run_config": str(run_config_path) if run_config_path.is_file() else None,
        "config": config,
        "pid": pid,
        "controller_log": str(controller_log) if controller_log.is_file() else None,
        "latest_update_time": format_time(latest_mtime),
        "latest_update_age": format_age(age_seconds(latest_mtime)),
        "latest_log": compact_path(latest_log),
        "latest_log_update_time": format_time(latest_log_mtime),
        "latest_log_update_age": format_age(age_seconds(latest_log_mtime)),
        "latest_stage_kind": stage_kind,
        "latest_stage": stage_name(latest_stage),
        "latest_stage_log": compact_path(latest_stage),
        "summaries": [str(path) for path in summaries],
        "completion_summaries": [str(path) for path in completion_summaries],
        "planned_tasks": str(planned_tasks) if planned_tasks.is_file() else None,
        "errors": errors,
        "finished_marker": finished_marker,
    }
    return result


def run_dirs_for_group(logs_root: Path, group: str | None) -> list[Path]:
    if group:
        group_dir = logs_root / group
        if not group_dir.is_dir():
            return []
        return sorted(path for path in group_dir.iterdir() if path.is_dir())
    run_dirs: list[Path] = []
    if not logs_root.is_dir():
        return []
    for group_dir in sorted(path for path in logs_root.iterdir() if path.is_dir()):
        run_dirs.extend(sorted(path for path in group_dir.iterdir() if path.is_dir()))
    return run_dirs


def non_empty_run_dirs(paths: list[Path]) -> list[Path]:
    result = []
    for path in paths:
        if latest_file_mtime(path) is not None:
            result.append(path)
    return result


def resolve_one_run(args: argparse.Namespace) -> Path:
    logs_root = Path(args.logs_root)
    if args.log_root:
        root = Path(args.log_root)
        if not root.is_dir():
            raise FileNotFoundError(f"log root not found: {root}")
        return root
    run = args.run or "latest"
    if run == "latest":
        runs = non_empty_run_dirs(run_dirs_for_group(logs_root, args.group))
        if not runs:
            group_text = f" group={args.group}" if args.group else ""
            raise FileNotFoundError(f"no runs found under {logs_root}{group_text}")
        return max(runs, key=lambda path: latest_file_mtime(path) or 0.0)
    if args.group:
        root = logs_root / args.group / run
        if not root.is_dir():
            raise FileNotFoundError(f"run not found: {root}")
        return root

    matches = [path for path in run_dirs_for_group(logs_root, None) if path.name == run]
    if not matches:
        raise FileNotFoundError(f"run not found under {logs_root}: {run}")
    if len(matches) > 1:
        choices = ", ".join(str(path) for path in matches)
        raise ValueError(f"ambiguous run id; specify --group. Matches: {choices}")
    return matches[0]


def resolve_all_runs(args: argparse.Namespace) -> list[Path]:
    logs_root = Path(args.logs_root)
    runs = non_empty_run_dirs(run_dirs_for_group(logs_root, args.group))
    return sorted(runs, key=lambda path: latest_file_mtime(path) or 0.0, reverse=True)


def print_run_human(result: dict[str, Any], tail_lines: int) -> None:
    print(f"Run: {result['run_id']}")
    print(f"Group: {result['group']}")
    print(f"Status: {result['status']}")
    print(f"Log root: {result['log_root']}")
    pid = result["pid"]
    if pid["exists"]:
        pid_text = f"{pid['pid']} ({'running' if pid['running'] else 'exited'})"
        if pid.get("error"):
            pid_text += f"; {pid['error']}"
    else:
        pid_text = "none"
    print(f"PID: {pid_text}")
    print(
        "Latest update: "
        f"{result['latest_update_time']} ({result['latest_update_age']})"
    )
    print(
        "Current/last stage: "
        f"{result['latest_stage'] or 'unknown'}"
        f" [{result['latest_stage_kind']}]"
    )
    print(f"Latest stage log: {result['latest_stage_log'] or 'none'}")
    print(f"Controller log: {result['controller_log'] or 'none'}")
    if result["summaries"]:
        print("Summaries:")
        for path in result["summaries"]:
            print(f"  - {path}")
    else:
        print("Summaries: none")
    if result["planned_tasks"]:
        print(f"Planned tasks: {result['planned_tasks']}")
    if result["errors"]:
        print("Recent errors:")
        for error in result["errors"]:
            print(f"  - {error['path']}: {error['line']}")
    else:
        print("Recent errors: none")

    log_paths = []
    if result["controller_log"]:
        log_paths.append(Path(result["controller_log"]))
    if result["latest_stage_log"]:
        stage_path = Path(result["latest_stage_log"])
        if stage_path not in log_paths:
            log_paths.append(stage_path)
    for path in log_paths:
        print(f"\nLast {tail_lines} lines: {path}")
        tail = read_tail(path, tail_lines)
        if not tail:
            print("  <empty or unreadable>")
            continue
        for line in tail:
            print(f"  {line}")


def print_all_human(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No runs found.")
        return
    headers = ("status", "group", "run_id", "latest_update", "stage")
    rows = []
    for result in results:
        rows.append(
            (
                result["status"],
                result["group"],
                result["run_id"],
                result["latest_update_age"],
                result["latest_stage"] or "-",
            )
        )
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    header_line = "  ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    )
    print(header_line)
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row))
        )


def main() -> int:
    args = parse_args()
    try:
        if args.all:
            roots = resolve_all_runs(args)
            results = [
                analyze_run(root, args.stale_minutes, args.scan_tail_lines)
                for root in roots
            ]
            if args.json:
                print(json.dumps(results, ensure_ascii=False, indent=2))
            else:
                print_all_human(results)
            return 0

        root = resolve_one_run(args)
        result = analyze_run(root, args.stale_minutes, args.scan_tail_lines)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print_run_human(result, args.tail_lines)
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"exp_status: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
