"""Parse purification logs, export Excel, and draw matplotlib PDF comparisons.

Run:
    python collect_log.py --log-dir log_purify --output purify_summary.xlsx \
        --plot-prefix plots/purify

Features:
- Scan ``log-dir`` for ``*.log`` files.
- Extract metrics (before/after accuracies, losses, MSE, compression_rate, etc.).
- Infer TN 架构 (PTR, PTR3d_interpolate, PTR3d, PTRtfs) from config 前缀，并按架构分 sheet 写入 Excel。
- 可选生成两张架构对比折线图 (compression_rate 作为横轴)，输出 PDF：
- 可选生成四张架构对比折线图 (compression_rate 作为横轴)，输出 PDF：
 - 可选生成六张架构对比折线图 (compression_rate 作为横轴)，输出 PDF：
    1) clean_acc_after vs compression_rate
    2) adv_acc_after vs compression_rate
    3) clean_acc_after 与 adv_acc_after 同图
    4) mean_mse_clean vs compression_rate
    5) mean_mse_adv vs compression_rate
    6) clean/adv 的 acc 与 mse 同图 (双纵轴)
Excel 依赖标准库；绘图需 matplotlib（可选）。
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape
import zipfile

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


FILENAME_RE = re.compile(r"^purify_([^_]+)_([^_]+)_([^_]+)_([^_]+)_(.+)\.log$")

BASE_ADV_RE = re.compile(
    r"Adversarial data accuracy:\s*([0-9.+eE-]+),\s*loss:\s*([0-9.+eE-]+)"
)
BASE_CLEAN_RE = re.compile(
    r"Clean data accuracy:\s*([0-9.+eE-]+),\s*loss:\s*([0-9.+eE-]+)"
)
PURIFIED_ADV_RE = re.compile(
    r"Purified adversarial data accuracy:\s*([0-9.+eE-]+),\s*loss:\s*([0-9.+eE-]+)"
)
PURIFIED_CLEAN_RE = re.compile(
    r"Purified clean data accuracy:\s*([0-9.+eE-]+),\s*loss:\s*([0-9.+eE-]+)"
)
MSE_ADV_RE = re.compile(r"Mean mse of purified adversarial data:\s*([0-9.+eE-]+)")
MSE_CLEAN_RE = re.compile(r"Mean mse of purified clean data:\s*([0-9.+eE-]+)")
COMPRESSION_RE = re.compile(r"compression rate:\s*([0-9.+eE-]+)")
SAMPLE_NUM_RE = re.compile(r"sample_num=(\d+)")
CONFIG_IN_LOG_RE = re.compile(r"config='([^']+)'")


@dataclass
class LogRecord:
    log_file: str
    dataset: Optional[str] = None
    model: Optional[str] = None
    attack: Optional[str] = None
    seed: Optional[int] = None
    config: Optional[str] = None
    arch: Optional[str] = None
    sample_num: Optional[int] = None
    compression_rate: Optional[float] = None
    adv_acc_before: Optional[float] = None
    adv_loss_before: Optional[float] = None
    clean_acc_before: Optional[float] = None
    clean_loss_before: Optional[float] = None
    adv_acc_after: Optional[float] = None
    adv_loss_after: Optional[float] = None
    clean_acc_after: Optional[float] = None
    clean_loss_after: Optional[float] = None
    mean_mse_adv: Optional[float] = None
    mean_mse_clean: Optional[float] = None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().rstrip(",")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_pair(pattern: re.Pattern[str], text: str) -> tuple[Optional[float], Optional[float]]:
    match = pattern.search(text)
    if not match:
        return None, None
    return _to_float(match.group(1)), _to_float(match.group(2))


def _extract_single(pattern: re.Pattern[str], text: str) -> Optional[float]:
    match = pattern.search(text)
    return _to_float(match.group(1)) if match else None


def parse_filename(name: str) -> Dict[str, Any]:
    """Parse dataset/model/attack/seed/config from the log file name."""
    parsed: Dict[str, Any] = {}
    match = FILENAME_RE.match(name)
    if not match:
        return parsed

    dataset, model, attack, seed, config = match.groups()
    parsed.update(
        {
            "dataset": dataset,
            "model": model,
            "attack": attack,
            "seed": int(seed) if seed.isdigit() else seed,
            "config": config,
        }
    )
    return parsed


def detect_arch(config_name: Optional[str]) -> Optional[str]:
    """Return the TN architecture (PTR, PTR3d_interpolate, PTR3d, PTRtfs) inferred from config."""
    if not config_name:
        return None
    lower = config_name.lower()
    if "3d_interpolate" in lower or lower.startswith("ptr3d_interpolate"):
        return "PTR3d_interpolate"
    for arch in ("PTR3d", "PTRtfs", "PTR"):
        if lower.startswith(arch.lower()):
            return arch
    return None


def parse_log_file(path: Path) -> LogRecord:
    """Extract metrics from a single log file."""
    content = path.read_text(encoding="utf-8", errors="ignore")

    metadata = parse_filename(path.name)
    adv_before, adv_loss_before = _extract_pair(BASE_ADV_RE, content)
    clean_before, clean_loss_before = _extract_pair(BASE_CLEAN_RE, content)
    adv_after, adv_loss_after = _extract_pair(PURIFIED_ADV_RE, content)
    clean_after, clean_loss_after = _extract_pair(PURIFIED_CLEAN_RE, content)

    record = LogRecord(
        log_file=path.name,
        dataset=metadata.get("dataset"),
        model=metadata.get("model"),
        attack=metadata.get("attack"),
        seed=metadata.get("seed"),
        config=metadata.get("config"),
        sample_num=_extract_single(SAMPLE_NUM_RE, content),
        compression_rate=_extract_single(COMPRESSION_RE, content),
        adv_acc_before=adv_before,
        adv_loss_before=adv_loss_before,
        clean_acc_before=clean_before,
        clean_loss_before=clean_loss_before,
        adv_acc_after=adv_after,
        adv_loss_after=adv_loss_after,
        clean_acc_after=clean_after,
        clean_loss_after=clean_loss_after,
        mean_mse_adv=_extract_single(MSE_ADV_RE, content),
        mean_mse_clean=_extract_single(MSE_CLEAN_RE, content),
    )

    # If config is missing from filename, fall back to the value in the log.
    if record.config is None:
        cfg_match = CONFIG_IN_LOG_RE.search(content)
        if cfg_match:
            record.config = cfg_match.group(1)

    # Infer architecture from config.
    record.arch = detect_arch(record.config)

    return record


# --- Minimal XLSX writer (no external dependencies) ----------------------------------

def _column_letter(idx: int) -> str:
    """Convert a 1-based column index into an Excel column label."""
    label = ""
    while idx:
        idx, rem = divmod(idx - 1, 26)
        label = chr(65 + rem) + label
    return label or "A"


def _render_sheet_rows(rows: Sequence[Sequence[Any]]) -> str:
    max_cols = max((len(r) for r in rows), default=1)
    max_rows = len(rows) if rows else 1
    dimension = f"A1:{_column_letter(max_cols)}{max_rows}"

    row_xml: List[str] = []
    for r_idx, row in enumerate(rows, start=1):
        cells: List[str] = []
        for c_idx, value in enumerate(row, start=1):
            cell_ref = f"{_column_letter(c_idx)}{r_idx}"
            if value is None:
                cells.append(f'<c r="{cell_ref}"/>')
                continue
            if isinstance(value, (int, float)):
                cells.append(f'<c r="{cell_ref}"><v>{value}</v></c>')
            else:
                cells.append(
                    f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'
                )
        row_xml.append(f'<row r="{r_idx}">' + "".join(cells) + "</row>")

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="{dimension}"/>'
        "<sheetData>"
        + "".join(row_xml)
        + "</sheetData>"
        "</worksheet>"
    )
    return sheet_xml


def _render_workbook(sheets: List[Tuple[str, Sequence[Sequence[Any]]]]) -> Dict[str, str]:
    """Build XML parts for a workbook with multiple sheets."""
    workbook_sheets_xml: List[str] = []
    workbook_rels_xml: List[str] = []
    parts: Dict[str, str] = {}

    for idx, (sheet_name, rows) in enumerate(sheets, start=1):
        safe_name = escape(sheet_name[:31]) or f"Sheet{idx}"
        rel_id = f"rId{idx}"
        sheet_file = f"worksheets/sheet{idx}.xml"

        workbook_sheets_xml.append(
            f'<sheet name="{safe_name}" sheetId="{idx}" r:id="{rel_id}"/>'
        )
        workbook_rels_xml.append(
            f'<Relationship Id="{rel_id}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="{sheet_file}"/>'
        )

        parts[f"xl/{sheet_file}"] = _render_sheet_rows(rows)

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        + "".join(workbook_sheets_xml)
        + "</sheets>"
        "</workbook>"
    )

    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        + "".join(workbook_rels_xml)
        + "</Relationships>"
    )

    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        + "".join(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for idx in range(1, len(sheets) + 1)
        )
        + "</Types>"
    )

    parts.update(
        {
            "[Content_Types].xml": content_types,
            "_rels/.rels": rels_xml,
            "xl/_rels/workbook.xml.rels": workbook_rels,
            "xl/workbook.xml": workbook_xml,
        }
    )
    return parts


def write_xlsx(path: Path, sheet_rows: Dict[str, Sequence[Sequence[Any]]]) -> None:
    """Write a minimal XLSX workbook with multiple sheets."""
    sheets = [(name, sheet_rows[name]) for name in sorted(sheet_rows.keys())]
    parts = _render_workbook(sheets)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in parts.items():
            zf.writestr(name, data)


# --- Data prep ----------------------------------------------------------------

def collect_records(log_dir: Path) -> List[LogRecord]:
    records: List[LogRecord] = []
    for path in sorted(log_dir.glob("*.log")):
        records.append(parse_log_file(path))
    return records


def build_rows(records: Iterable[LogRecord]) -> List[List[Any]]:
    columns = [
        "log_file",
        "dataset",
        "model",
        "attack",
        "seed",
        "config",
        "arch",
        "sample_num",
        "compression_rate",
        "adv_acc_before",
        "adv_loss_before",
        "clean_acc_before",
        "clean_loss_before",
        "adv_acc_after",
        "adv_loss_after",
        "clean_acc_after",
        "clean_loss_after",
        "mean_mse_adv",
        "mean_mse_clean",
    ]

    rows = [columns]
    for record in records:
        data = asdict(record)
        rows.append([data.get(col) for col in columns])
    return rows


def build_rows_by_arch(records: Iterable[LogRecord]) -> Dict[str, List[List[Any]]]:
    grouped: Dict[str, List[LogRecord]] = {}
    for record in records:
        key = record.arch or "Unknown"
        grouped.setdefault(key, []).append(record)

    rows_by_arch: Dict[str, List[List[Any]]] = {}
    for arch, recs in grouped.items():
        rows_by_arch[arch] = build_rows(recs)
    return rows_by_arch


# --- Plotting (matplotlib, optional) ----------------------------------------


def _build_series(records: Iterable[LogRecord]) -> Dict[str, List[Dict[str, Any]]]:
    series: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        if r.compression_rate is None:
            continue
        arch = r.arch or "Unknown"
        point = {
            "compression_rate": r.compression_rate,
            "clean_acc_after": r.clean_acc_after,
            "adv_acc_after": r.adv_acc_after,
            "mean_mse_clean": r.mean_mse_clean,
            "mean_mse_adv": r.mean_mse_adv,
        }
        if not any(v is not None for v in point.values()):
            continue
        series.setdefault(arch, []).append(point)

    for arch in list(series.keys()):
        series[arch] = sorted(series[arch], key=lambda p: p["compression_rate"])
    return {k: series[k] for k in sorted(series.keys())}


def _plot_metric_matplotlib(
    series: Dict[str, List[Dict[str, Any]]],
    metric: str,
    outfile: Path,
    y_label: str,
    x_label: str = "compression_rate",
) -> None:
    if plt is None:
        print(f"Skip plotting {metric}: matplotlib not available. pip install matplotlib to enable plotting.")
        return

    has_data = False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arch, points in series.items():
        xs: List[float] = []
        ys: List[float] = []
        for p in points:
            if p.get(metric) is None:
                continue
            xs.append(p["compression_rate"])
            ys.append(p[metric])
        if not xs or not ys:
            continue
        has_data = True
        ax.plot(xs, ys, marker="o", label=arch)

    if not has_data:
        plt.close(fig)
        print(f"Skip plotting {metric}: no data.")
        return

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(title="arch")
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, format="pdf")
    plt.close(fig)
    print(f"Plot saved: {outfile}")


def _plot_acc_pair_matplotlib(series: Dict[str, List[Dict[str, Any]]], outfile: Path) -> None:
    """Plot clean_acc_after and adv_acc_after together with per-arch colors."""
    if plt is None:
        print("Skip plotting acc pair: matplotlib not available. pip install matplotlib to enable plotting.")
        return

    palette = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    colors = {arch: palette[idx % len(palette)] for idx, arch in enumerate(series.keys())}

    has_data = False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arch, points in series.items():
        xs_clean = [p["compression_rate"] for p in points if p.get("clean_acc_after") is not None]
        ys_clean = [p["clean_acc_after"] for p in points if p.get("clean_acc_after") is not None]
        xs_adv = [p["compression_rate"] for p in points if p.get("adv_acc_after") is not None]
        ys_adv = [p["adv_acc_after"] for p in points if p.get("adv_acc_after") is not None]

        if xs_clean and ys_clean:
            has_data = True
            ax.plot(xs_clean, ys_clean, marker="o", linestyle="-", color=colors[arch], label=f"{arch} clean")
        if xs_adv and ys_adv:
            has_data = True
            ax.plot(xs_adv, ys_adv, marker="o", linestyle="--", color=colors[arch], label=f"{arch} adv")

    if not has_data:
        plt.close(fig)
        print("Skip plotting acc pair: no data.")
        return

    ax.set_xlabel("compression_rate")
    ax.set_ylabel("acc_after")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(title="arch / metric")
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, format="pdf")
    plt.close(fig)
    print(f"Plot saved: {outfile}")


def _plot_acc_mse_combo_matplotlib(series: Dict[str, List[Dict[str, Any]]], outfile: Path) -> None:
    """Plot clean/adv acc and mse in two subplots sharing the x-axis."""
    if plt is None:
        print("Skip plotting acc/mse combo: matplotlib not available. pip install matplotlib to enable plotting.")
        return

    palette = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    colors = {arch: palette[idx % len(palette)] for idx, arch in enumerate(series.keys())}

    has_acc = False
    has_mse = False
    fig, (ax_acc, ax_mse) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for arch, points in series.items():
        color = colors[arch]
        xs_clean_acc = [p["compression_rate"] for p in points if p.get("clean_acc_after") is not None]
        ys_clean_acc = [p["clean_acc_after"] for p in points if p.get("clean_acc_after") is not None]
        xs_adv_acc = [p["compression_rate"] for p in points if p.get("adv_acc_after") is not None]
        ys_adv_acc = [p["adv_acc_after"] for p in points if p.get("adv_acc_after") is not None]

        xs_clean_mse = [p["compression_rate"] for p in points if p.get("mean_mse_clean") is not None]
        ys_clean_mse = [p["mean_mse_clean"] for p in points if p.get("mean_mse_clean") is not None]
        xs_adv_mse = [p["compression_rate"] for p in points if p.get("mean_mse_adv") is not None]
        ys_adv_mse = [p["mean_mse_adv"] for p in points if p.get("mean_mse_adv") is not None]

        if xs_clean_acc and ys_clean_acc:
            has_acc = True
            ax_acc.plot(xs_clean_acc, ys_clean_acc, marker="o", linestyle="-", color=color, label=f"{arch} clean acc")
        if xs_adv_acc and ys_adv_acc:
            has_acc = True
            ax_acc.plot(xs_adv_acc, ys_adv_acc, marker="o", linestyle="--", color=color, label=f"{arch} adv acc")

        if xs_clean_mse and ys_clean_mse:
            has_mse = True
            ax_mse.plot(xs_clean_mse, ys_clean_mse, marker="s", linestyle=":", color=color, label=f"{arch} clean mse")
        if xs_adv_mse and ys_adv_mse:
            has_mse = True
            ax_mse.plot(xs_adv_mse, ys_adv_mse, marker="s", linestyle="-.", color=color, label=f"{arch} adv mse")

    if not (has_acc or has_mse):
        plt.close(fig)
        print("Skip plotting acc/mse combo: no data.")
        return

    ax_acc.set_ylabel("acc_after")
    ax_mse.set_ylabel("mean_mse")
    ax_mse.set_xlabel("compression_rate")
    ax_acc.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax_mse.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    legend_opts = {"fontsize": 8, "title_fontsize": 8}
    handles_acc, labels_acc = ax_acc.get_legend_handles_labels()
    handles_mse, labels_mse = ax_mse.get_legend_handles_labels()
    if handles_acc:
        ax_acc.legend(handles_acc, labels_acc, title="arch / acc", loc="lower right", **legend_opts)
    if handles_mse:
        ax_mse.legend(handles_mse, labels_mse, title="arch / mse", loc="upper right", **legend_opts)

    fig.tight_layout(h_pad=1.2)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, format="pdf")
    plt.close(fig)
    print(f"Plot saved: {outfile}")


def generate_plots(records: Iterable[LogRecord], prefix: Path) -> None:
    series = _build_series(records)
    if not series:
        print("No series data to plot.")
        return
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix_no_ext = prefix.with_suffix("")
    _plot_metric_matplotlib(
        series,
        "clean_acc_after",
        prefix_no_ext.with_name(prefix_no_ext.name + "_clean_acc_after.pdf"),
        "clean_acc_after",
    )
    _plot_metric_matplotlib(
        series,
        "adv_acc_after",
        prefix_no_ext.with_name(prefix_no_ext.name + "_adv_acc_after.pdf"),
        "adv_acc_after",
    )
    _plot_metric_matplotlib(
        series,
        "mean_mse_clean",
        prefix_no_ext.with_name(prefix_no_ext.name + "_mean_mse_clean.pdf"),
        "mean_mse_clean",
    )
    _plot_metric_matplotlib(
        series,
        "mean_mse_adv",
        prefix_no_ext.with_name(prefix_no_ext.name + "_mean_mse_adv.pdf"),
        "mean_mse_adv",
    )
    _plot_acc_pair_matplotlib(
        series,
        prefix_no_ext.with_name(prefix_no_ext.name + "_acc_pair.pdf"),
    )
    _plot_acc_mse_combo_matplotlib(
        series,
        prefix_no_ext.with_name(prefix_no_ext.name + "_acc_mse_combo.pdf"),
    )


# --- CLI --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect purification log metrics into Excel.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("log_purify"),
        help="Directory containing log files (default: log_purify)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("purify_summary.xlsx"),
        help="Path to the output Excel file (default: purify_summary.xlsx)",
    )
    parser.add_argument(
        "--plot-prefix",
        type=Path,
        default=Path("plots/purify"),
        help="If set, export matplotlib PDF plots using this prefix (e.g., plots/purify).",
    )
    args = parser.parse_args()

    if not args.log_dir.exists():
        raise SystemExit(f"Log directory not found: {args.log_dir}")

    records = collect_records(args.log_dir)
    if not records:
        raise SystemExit(f"No log files found in {args.log_dir}")

    rows_by_arch = build_rows_by_arch(records)
    write_xlsx(args.output, rows_by_arch)
    sheet_names = ", ".join(sorted(rows_by_arch.keys()))
    print(f"Wrote {len(records)} records across sheets [{sheet_names}] to {args.output}")

    if args.plot_prefix:
        generate_plots(records, args.plot_prefix)


if __name__ == "__main__":
    main()
