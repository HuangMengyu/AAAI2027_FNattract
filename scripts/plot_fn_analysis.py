#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import defaultdict


plt = None
Line2D = None

METRICS = ["precision", "recall", "false_attraction_rate"]
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "false_attraction_rate": "False attraction rate",
}
SIGNAL_ORDER = ["binary", "prior", "combined"]
SIGNAL_COLORS = {
    "binary": "#2563eb",
    "prior": "#16a34a",
    "combined": "#dc2626",
}
SIGNAL_STYLES = {
    "binary": (0, (5, 3)),
    "prior": (0, (2, 2)),
    "combined": "solid",
}
SIGNAL_MARKERS = {
    "binary": "o",
    "prior": "^",
    "combined": "s",
}
SIGNAL_WIDTHS = {
    "binary": 1.8,
    "prior": 1.8,
    "combined": 3.0,
}
MODEL_COLORS = {
    "with prior": "#dc2626",
    "no prior": "#2563eb",
}
MODEL_STYLES = {
    "with prior": "solid",
    "no prior": (0, (5, 3)),
}
MODEL_MARKERS = {
    "with prior": "o",
    "no prior": "x",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot fn_attraction_epoch_summary.csv files with matplotlib. "
            "Inputs can be CSV files or directories containing fn_attraction_epoch_summary.csv."
        )
    )
    parser.add_argument("--with-prior", nargs="+", required=True)
    parser.add_argument("--no-prior", nargs="*", default=None)
    parser.add_argument("--out-dir", default="fn_analysis_plots")
    parser.add_argument("--with-prior-signal", default="binary", choices=["binary"], help="Signal used for with-prior false-attraction comparison; fixed to binary")
    parser.add_argument("--no-prior-signal", default="binary", choices=["binary"], help="Signal used for no-prior false-attraction comparison; fixed to binary")
    parser.add_argument("--title-prefix", default="FN attraction analysis")
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf", "svg"], help="Output formats. PNG is the default; add pdf/svg only when vector output is needed.")
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument(
        "--use_average",
        action="store_true",
        help=(
            "Average branch variants into branch families before plotting. "
            "For example, inter:*, intra:*, and temporal:* become inter, intra, and temporal."
        ),
    )
    return parser.parse_args()


def import_matplotlib():
    global plt, Line2D
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    from matplotlib.lines import Line2D as MatplotlibLine2D
    plt = pyplot
    Line2D = MatplotlibLine2D


def find_csvs(paths):
    csvs = []
    for path in paths or []:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                if "fn_attraction_epoch_summary.csv" in files:
                    csvs.append(os.path.join(root, "fn_attraction_epoch_summary.csv"))
        elif os.path.isfile(path):
            csvs.append(path)
        else:
            raise FileNotFoundError(path)
    return sorted(set(csvs))


def to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_rows(paths):
    rows = []
    for path in find_csvs(paths):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    row["epoch"] = int(float(row["epoch"]))
                except (KeyError, ValueError):
                    continue
                row["_source"] = path
                for metric in METRICS:
                    row[metric] = to_float(row.get(metric))
                rows.append(row)
    return rows


def branch_family(branch):
    if not branch:
        return branch
    return str(branch).split(":", 1)[0]


def use_branch_family_average(rows):
    averaged_rows = []
    for row in rows:
        new_row = dict(row)
        new_row["original_branch"] = row.get("branch")
        new_row["branch"] = branch_family(row.get("branch"))
        averaged_rows.append(new_row)
    return averaged_rows


def sanitize_name(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def unique_values(rows, field):
    return sorted({row[field] for row in rows if row.get(field) not in (None, "")})


def average_rows(rows, fields, metric):
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is None or math.isnan(value):
            continue
        key = tuple(row[field] for field in fields)
        grouped[key].append(value)
    averaged = []
    for key, values in grouped.items():
        out = {field: item for field, item in zip(fields, key)}
        out[metric] = sum(values) / len(values)
        averaged.append(out)
    return sorted(averaged, key=lambda row: tuple(row[field] for field in fields))


def smooth_points(points, window):
    if window <= 1 or len(points) <= 1:
        return points
    out = []
    for idx, (epoch, _) in enumerate(points):
        left = max(0, idx - window + 1)
        values = [value for _, value in points[left:idx + 1]]
        out.append((epoch, sum(values) / len(values)))
    return out


def points_for(rows, metric, branch=None, signal=None, model=None, smooth_window=1):
    filtered = []
    for row in rows:
        if branch is not None and row.get("branch") != branch:
            continue
        if signal is not None and row.get("signal") != signal:
            continue
        if model is not None and row.get("model") != model:
            continue
        filtered.append(row)
    fields = ["epoch"]
    averaged = average_rows(filtered, fields, metric)
    points = [(row["epoch"], row[metric]) for row in averaged]
    return smooth_points(points, smooth_window)


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e5e7eb",
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save_figure(fig, out_dir, stem, formats, dpi):
    os.makedirs(out_dir, exist_ok=True)
    for fmt in formats:
        fig.savefig(os.path.join(out_dir, f"{stem}.{fmt}"), dpi=dpi if fmt == "png" else None)
    plt.close(fig)


def signal_handles():
    return [
        Line2D(
            [0],
            [0],
            color=SIGNAL_COLORS[signal],
            linestyle=SIGNAL_STYLES[signal],
            marker=SIGNAL_MARKERS[signal],
            markersize=4.5,
            linewidth=SIGNAL_WIDTHS[signal],
            label=signal,
        )
        for signal in SIGNAL_ORDER
    ]


def signal_style_handles():
    return [
        Line2D(
            [0],
            [0],
            color="#111827",
            linestyle=SIGNAL_STYLES[signal],
            marker=SIGNAL_MARKERS[signal],
            markersize=4.5,
            linewidth=SIGNAL_WIDTHS[signal],
            label=signal,
        )
        for signal in SIGNAL_ORDER
    ]


def branch_handles(branches, colors):
    return [
        Line2D([0], [0], color=colors[branch], linewidth=2.2, label=branch)
        for branch in branches
    ]


def model_handles():
    return [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[name],
            linestyle=MODEL_STYLES[name],
            marker=MODEL_MARKERS[name],
            markersize=4.5,
            linewidth=2.6,
            label=name,
        )
        for name in ["with prior", "no prior"]
    ]


def model_style_handles():
    return [
        Line2D(
            [0],
            [0],
            color="#111827",
            linestyle=MODEL_STYLES[name],
            marker=MODEL_MARKERS[name],
            markersize=4.5,
            linewidth=2.6,
            label=name,
        )
        for name in ["with prior", "no prior"]
    ]


def plot_metric_by_signal(ax, rows, metric, branch=None, smooth_window=1):
    for signal in SIGNAL_ORDER:
        points = points_for(rows, metric, branch=branch, signal=signal, smooth_window=smooth_window)
        if not points:
            continue
        epochs, values = zip(*points)
        ax.plot(
            epochs,
            values,
            color=SIGNAL_COLORS[signal],
            linestyle=SIGNAL_STYLES[signal],
            marker=SIGNAL_MARKERS[signal],
            markersize=3.5,
            markevery=max(1, len(epochs) // 8),
            linewidth=SIGNAL_WIDTHS[signal],
            label=signal,
        )
    ax.set_title(METRIC_LABELS[metric])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_ylim(bottom=0)


def plot_exp1_one_branch(rows, branch, out_dir, title_prefix, formats, dpi, smooth_window):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharex=True)
    for ax, metric in zip(axes, METRICS):
        plot_metric_by_signal(ax, rows, metric, branch=branch, smooth_window=smooth_window)
    axes[0].legend(handles=signal_handles(), frameon=False, loc="best")
    fig.suptitle(f"{title_prefix}: {branch}")
    fig.tight_layout()
    save_figure(fig, out_dir, f"exp1_branch_{sanitize_name(branch)}_metrics", formats, dpi)


def plot_exp1_all_branches(rows, out_dir, title_prefix, formats, dpi, smooth_window):
    branches = unique_values(rows, "branch")
    fig, axes = plt.subplots(
        len(METRICS),
        len(branches),
        figsize=(4.0 * len(branches), 3.2 * len(METRICS)),
        sharex=False,
    )
    if len(branches) == 1:
        axes = [[axes[idx]] for idx in range(len(METRICS))]
    for row_idx, metric in enumerate(METRICS):
        for col_idx, branch in enumerate(branches):
            ax = axes[row_idx][col_idx] if len(branches) > 1 else axes[row_idx][0]
            plot_metric_by_signal(ax, rows, metric, branch=branch, smooth_window=smooth_window)
            ax.set_title(branch if row_idx == 0 else "")
            if col_idx != 0:
                ax.set_ylabel("")
            if row_idx != len(METRICS) - 1:
                ax.set_xlabel("")
    fig.legend(handles=signal_handles(), frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{title_prefix}: all branches", y=1.055)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, out_dir, "exp1_all_branches_metrics", formats, dpi)


def plot_exp1_all_branches_overlay(rows, out_dir, title_prefix, formats, dpi, smooth_window):
    branches = unique_values(rows, "branch")
    cmap = plt.get_cmap("tab10")
    branch_colors = {branch: cmap(idx % 10) for idx, branch in enumerate(branches)}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharex=True)
    for ax, metric in zip(axes, METRICS):
        averaged = average_rows(rows, ["epoch", "branch", "signal"], metric)
        for branch in branches:
            for signal in SIGNAL_ORDER:
                points = [
                    (row["epoch"], row[metric])
                    for row in averaged
                    if row["branch"] == branch and row["signal"] == signal
                ]
                points = smooth_points(sorted(points), smooth_window)
                if not points:
                    continue
                epochs, values = zip(*points)
                ax.plot(
                    epochs,
                    values,
                    color=branch_colors[branch],
                    linestyle=SIGNAL_STYLES[signal],
                    marker=SIGNAL_MARKERS[signal],
                    markersize=3.2,
                    markevery=max(1, len(epochs) // 8),
                    linewidth=SIGNAL_WIDTHS[signal],
                    alpha=0.72 if signal == "combined" else 0.48,
                )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_ylim(bottom=0)
    fig.legend(
        handles=branch_handles(branches, branch_colors),
        frameon=False,
        loc="lower center",
        ncol=min(len(branches), 5),
        bbox_to_anchor=(0.5, 0.07),
    )
    fig.legend(
        handles=signal_style_handles(),
        frameon=False,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(f"{title_prefix}: all branches overlay", y=0.995)
    fig.tight_layout(rect=(0, 0.18, 1, 0.93))
    save_figure(fig, out_dir, "exp1_all_branches_overlay_metrics", formats, dpi)


def prepare_model_comparison(with_rows, no_rows, with_signal, no_signal):
    compare_rows = []
    for row in with_rows:
        if row.get("signal") == with_signal:
            new_row = dict(row)
            new_row["model"] = "with prior"
            compare_rows.append(new_row)
    for row in no_rows:
        if row.get("signal") == no_signal:
            new_row = dict(row)
            new_row["model"] = "no prior"
            compare_rows.append(new_row)
    return compare_rows


def plot_exp2_one_branch(rows, branch, out_dir, title_prefix, formats, dpi, smooth_window):
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for model_name in ["with prior", "no prior"]:
        points = points_for(
            rows,
            "false_attraction_rate",
            branch=branch,
            model=model_name,
            smooth_window=smooth_window,
        )
        if not points:
            continue
        epochs, values = zip(*points)
        ax.plot(
            epochs,
            values,
            color=MODEL_COLORS[model_name],
            linestyle=MODEL_STYLES[model_name],
            marker=MODEL_MARKERS[model_name],
            markersize=4.0,
            markevery=max(1, len(epochs) // 8),
            linewidth=2.8 if model_name == "with prior" else 2.2,
            label=model_name,
        )
    ax.set_title("Binary false attraction rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("False attraction rate")
    ax.set_ylim(bottom=0)
    ax.legend(handles=model_handles(), frameon=False)
    fig.suptitle(f"{title_prefix}: {branch} (binary signal)")
    fig.tight_layout()
    save_figure(fig, out_dir, f"exp2_branch_{sanitize_name(branch)}_false_attraction_with_vs_no_prior", formats, dpi)


def plot_exp2_all_branches(rows, out_dir, title_prefix, formats, dpi, smooth_window):
    branches = unique_values(rows, "branch")
    cols = min(3, max(1, len(branches)))
    grid_rows = math.ceil(len(branches) / cols)
    fig, axes = plt.subplots(grid_rows, cols, figsize=(5.4 * cols, 3.7 * grid_rows), sharex=False, sharey=True)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, branch in zip(axes, branches):
        for model_name in ["with prior", "no prior"]:
            points = points_for(
                rows,
                "false_attraction_rate",
                branch=branch,
                model=model_name,
                smooth_window=smooth_window,
            )
            if not points:
                continue
            epochs, values = zip(*points)
            ax.plot(
                epochs,
                values,
                color=MODEL_COLORS[model_name],
                linestyle=MODEL_STYLES[model_name],
                marker=MODEL_MARKERS[model_name],
                markersize=3.5,
                markevery=max(1, len(epochs) // 8),
                linewidth=2.6 if model_name == "with prior" else 2.0,
            )
        ax.set_title(branch)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary false attraction rate")
        ax.set_ylim(bottom=0)
    for ax in axes[len(branches):]:
        ax.axis("off")
    fig.legend(handles=model_handles(), frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{title_prefix}: binary false attraction with prior vs no prior", y=1.055)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, out_dir, "exp2_all_branches_false_attraction_with_vs_no_prior", formats, dpi)


def plot_exp2_all_branches_overlay(rows, out_dir, title_prefix, formats, dpi, smooth_window):
    branches = unique_values(rows, "branch")
    cmap = plt.get_cmap("tab10")
    branch_colors = {branch: cmap(idx % 10) for idx, branch in enumerate(branches)}
    averaged = average_rows(rows, ["epoch", "branch", "model"], "false_attraction_rate")
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    for branch in branches:
        for model_name in ["with prior", "no prior"]:
            points = [
                (row["epoch"], row["false_attraction_rate"])
                for row in averaged
                if row["branch"] == branch and row["model"] == model_name
            ]
            points = smooth_points(sorted(points), smooth_window)
            if not points:
                continue
            epochs, values = zip(*points)
            ax.plot(
                epochs,
                values,
                color=branch_colors[branch],
                linestyle=MODEL_STYLES[model_name],
                marker=MODEL_MARKERS[model_name],
                markersize=3.5,
                markevery=max(1, len(epochs) // 8),
                linewidth=2.7 if model_name == "with prior" else 1.9,
                alpha=0.72 if model_name == "with prior" else 0.46,
            )
    ax.set_title("Binary false attraction rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary false attraction rate")
    ax.set_ylim(bottom=0)
    fig.legend(
        handles=branch_handles(branches, branch_colors),
        frameon=False,
        loc="lower center",
        ncol=min(len(branches), 5),
        bbox_to_anchor=(0.5, 0.07),
    )
    fig.legend(
        handles=model_style_handles(),
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(f"{title_prefix}: all branches overlay (binary signal)", y=0.995)
    fig.tight_layout(rect=(0, 0.18, 1, 0.93))
    save_figure(fig, out_dir, "exp2_all_branches_overlay_false_attraction_with_vs_no_prior", formats, dpi)


def main():
    args = parse_args()
    import_matplotlib()
    setup_style()
    with_rows = read_rows(args.with_prior)
    if not with_rows:
        raise ValueError("No with-prior rows found")
    if args.use_average:
        with_rows = use_branch_family_average(with_rows)
    os.makedirs(args.out_dir, exist_ok=True)

    plot_exp1_all_branches(with_rows, args.out_dir, args.title_prefix, args.formats, args.dpi, args.smooth_window)
    plot_exp1_all_branches_overlay(with_rows, args.out_dir, args.title_prefix, args.formats, args.dpi, args.smooth_window)
    for branch in unique_values(with_rows, "branch"):
        plot_exp1_one_branch(with_rows, branch, args.out_dir, args.title_prefix, args.formats, args.dpi, args.smooth_window)

    if args.no_prior:
        no_rows = read_rows(args.no_prior)
        if no_rows:
            if args.use_average:
                no_rows = use_branch_family_average(no_rows)
            compare_rows = prepare_model_comparison(
                with_rows,
                no_rows,
                args.with_prior_signal,
                args.no_prior_signal,
            )
            plot_exp2_all_branches(compare_rows, args.out_dir, args.title_prefix, args.formats, args.dpi, args.smooth_window)
            plot_exp2_all_branches_overlay(compare_rows, args.out_dir, args.title_prefix, args.formats, args.dpi, args.smooth_window)
            for branch in unique_values(compare_rows, "branch"):
                plot_exp2_one_branch(compare_rows, branch, args.out_dir, args.title_prefix, args.formats, args.dpi, args.smooth_window)

    print(f"Saved plots to {args.out_dir}")


if __name__ == "__main__":
    main()
