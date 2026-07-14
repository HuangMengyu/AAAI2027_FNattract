#!/usr/bin/env python3
import argparse
import csv
import math
import os
import statistics
from collections import defaultdict


plt = None
Line2D = None

ATTRACTION_CSV = "fn_attraction_epoch_summary.csv"
OVERLAP_CSV = "fn_binary_prior_overlap.csv"

METRICS = ["precision", "recall", "false_attraction_rate"]
METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "false_attraction_rate": "False attraction rate",
}
SIGNAL_ORDER = ["binary", "prior", "combined"]
SIGNAL_COLORS = {
    "binary": "#1F77B4",
    "prior": "#2CA02C",
    "combined": "#D62728",
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
    "binary": 1.5,
    "prior": 1.5,
    "combined": 1.5,
}
PLOT_BACKGROUND_COLOR = "white"
MODEL_ORDER = ["with prior", "no prior"]
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
BRANCH_ORDER = ["inter", "intra", "temporal"]
BRANCH_COLORS = {
    "inter": "#1F77B4",
    "intra": "#2CA02C",
    "temporal": "#D62728",
}
OVERLAP_GROUP_ORDER = [
    "binary_no_prior_no",
    "binary_no_prior_yes",
    "binary_yes_prior_no",
    "binary_yes_prior_yes",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot all-fold fn_analysis results with branch-family averaging. "
            "Inputs can be CSV files or directories containing per-fold fn_analysis subfolders."
        )
    )
    parser.add_argument("--with-prior", nargs="+", required=True)
    parser.add_argument("--no-prior", nargs="*", default=None)
    parser.add_argument("--out-dir", default="fn_analysis_allfolds_plots")
    parser.add_argument("--with-prior-signal", default="binary", choices=["binary"])
    parser.add_argument("--no-prior-signal", default="binary", choices=["binary"])
    parser.add_argument("--title-prefix", default="FN attraction analysis")
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
        help="Output plot formats. PNG is the default.",
    )
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args()


def import_matplotlib():
    global plt, Line2D
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    from matplotlib.lines import Line2D as MatplotlibLine2D

    plt = pyplot
    Line2D = MatplotlibLine2D


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": "white",
        "grid.linewidth": 1.0,
        "grid.alpha": 1.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "savefig.facecolor": PLOT_BACKGROUND_COLOR,
        "savefig.edgecolor": PLOT_BACKGROUND_COLOR,
        "savefig.transparent": False,
        "figure.facecolor": PLOT_BACKGROUND_COLOR,
        "axes.facecolor": PLOT_BACKGROUND_COLOR,
    })


def find_named_csvs(paths, filename):
    csvs = []
    for path in paths or []:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                if filename in files:
                    csvs.append(os.path.join(root, filename))
        elif os.path.isfile(path):
            if os.path.basename(path) == filename:
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


def to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def read_rows(paths, filename, numeric_fields):
    rows = []
    csv_paths = find_named_csvs(paths, filename)
    for path in csv_paths:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                epoch = to_int(row.get("epoch"))
                if epoch is None:
                    continue
                row["epoch"] = epoch
                row["_source"] = path
                row["_run_id"] = make_run_id(row)
                for field in numeric_fields:
                    row[field] = to_float(row.get(field))
                rows.append(row)
    return rows, csv_paths


def make_run_id(row):
    return "|".join([
        str(row.get("dataset_name", "")),
        str(row.get("fold", "")),
        str(row.get("seed", "")),
        str(row.get("_source", "")),
    ])


def branch_family(branch):
    if not branch:
        return branch
    return str(branch).split(":", 1)[0]


def mean_std(values):
    clean = [value for value in values if value is not None and not math.isnan(value)]
    if not clean:
        return None, None, 0
    mean = statistics.mean(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return mean, std, len(clean)


def order_index(value, order):
    try:
        return order.index(value)
    except ValueError:
        return len(order)


def unique_ordered(values, order=None):
    values = sorted(set(values))
    if order is None:
        return values
    return sorted(values, key=lambda value: (order_index(value, order), value))


def aggregate_rows_by_branch_family(rows, key_fields, metric_fields):
    grouped = defaultdict(lambda: defaultdict(list))
    metadata = {}
    for row in rows:
        branch = branch_family(row.get("branch"))
        key_items = []
        for field in key_fields:
            key_items.append(branch if field == "branch" else row.get(field))
        key = tuple(key_items)
        metadata.setdefault(key, {
            field: value for field, value in zip(key_fields, key)
        })
        metadata[key].update({
            "dataset_name": row.get("dataset_name", ""),
            "fold": row.get("fold", ""),
            "seed": row.get("seed", ""),
            "_source": row.get("_source", ""),
            "_run_id": row.get("_run_id", ""),
        })
        for field in metric_fields:
            value = row.get(field)
            if value is not None and not math.isnan(value):
                grouped[key][field].append(value)

    averaged = []
    for key in sorted(grouped):
        out = dict(metadata[key])
        for field in metric_fields:
            values = grouped[key].get(field, [])
            out[field] = statistics.mean(values) if values else None
        averaged.append(out)
    return averaged


def aggregate_attraction_rows(rows):
    return aggregate_rows_by_branch_family(
        rows,
        ["_run_id", "dataset_name", "fold", "seed", "epoch", "branch", "signal"],
        METRICS,
    )


def aggregate_overlap_rows(rows):
    return aggregate_rows_by_branch_family(
        rows,
        ["_run_id", "dataset_name", "fold", "seed", "epoch", "branch", "group"],
        ["same_class_rate", "pair_percentage"],
    )


def final_epoch_rows_per_run(rows):
    max_epoch_by_run = {}
    for row in rows:
        run_id = row.get("_run_id")
        epoch = row.get("epoch")
        if run_id is None or epoch is None:
            continue
        max_epoch_by_run[run_id] = max(epoch, max_epoch_by_run.get(run_id, epoch))
    return [
        row for row in rows
        if row.get("epoch") == max_epoch_by_run.get(row.get("_run_id"))
    ]


def summarize_rows(rows, group_fields, metric_fields):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        grouped[key].append(row)

    summaries = []
    for key in sorted(grouped):
        rows_for_key = grouped[key]
        out = {field: value for field, value in zip(group_fields, key)}
        run_ids = sorted({row.get("_run_id", "") for row in rows_for_key})
        epochs = sorted({str(row.get("epoch")) for row in rows_for_key})
        out["n_runs"] = len(run_ids)
        out["epochs_used"] = ";".join(epochs)
        for field in metric_fields:
            mean, std, n = mean_std([row.get(field) for row in rows_for_key])
            out[f"{field}_mean"] = mean
            out[f"{field}_std"] = std
            out[f"{field}_n"] = n
        summaries.append(out)
    return summaries


def summary_lookup(summaries, key_fields):
    lookup = {}
    for row in summaries:
        key = tuple(row.get(field) for field in key_fields)
        lookup[key] = row
    return lookup


def smooth_series(epochs, means, stds, window):
    if window <= 1 or len(epochs) <= 1:
        return epochs, means, stds
    out_means = []
    out_stds = []
    for idx in range(len(epochs)):
        left = max(0, idx - window + 1)
        out_means.append(statistics.mean(means[left:idx + 1]))
        out_stds.append(statistics.mean(stds[left:idx + 1]))
    return epochs, out_means, out_stds


def save_figure(fig, out_dir, stem, formats, dpi):
    os.makedirs(out_dir, exist_ok=True)
    fig.patch.set_facecolor(PLOT_BACKGROUND_COLOR)
    for ax in fig.axes:
        ax.set_facecolor(PLOT_BACKGROUND_COLOR)
    for fmt in formats:
        fig.savefig(
            os.path.join(out_dir, f"{stem}.{fmt}"),
            dpi=dpi if fmt == "png" else None,
            facecolor=PLOT_BACKGROUND_COLOR,
            edgecolor=PLOT_BACKGROUND_COLOR,
            transparent=False,
        )
    plt.close(fig)


def signal_handles(style_color="#111827"):
    return [
        Line2D(
            [0],
            [0],
            color=style_color if style_color else SIGNAL_COLORS[signal],
            linestyle=SIGNAL_STYLES[signal],
            marker=SIGNAL_MARKERS[signal],
            markersize=4.5,
            linewidth=SIGNAL_WIDTHS[signal],
            label=signal,
        )
        for signal in SIGNAL_ORDER
    ]


def branch_handles(branches):
    return [
        Line2D([0], [0], color=BRANCH_COLORS.get(branch, "#6b7280"), linewidth=2.6, label=branch)
        for branch in branches
    ]


def model_handles(style_color=None):
    return [
        Line2D(
            [0],
            [0],
            color=style_color if style_color else MODEL_COLORS[model],
            linestyle=MODEL_STYLES[model],
            marker=MODEL_MARKERS[model],
            markersize=4.5,
            linewidth=2.6,
            label=model,
        )
        for model in MODEL_ORDER
    ]


def plot_summary_line(ax, summaries, key_fields, key_values, metric, color, linestyle, marker,
                      linewidth, alpha_fill=0.16, smooth_window=1):
    rows = [
        row for row in summaries
        if all(row.get(field) == value for field, value in zip(key_fields, key_values))
    ]
    rows = sorted(rows, key=lambda row: row["epoch"])
    if not rows:
        return False
    epochs = [row["epoch"] for row in rows]
    means = [row[f"{metric}_mean"] for row in rows]
    stds = [row[f"{metric}_std"] or 0.0 for row in rows]
    if any(value is None for value in means):
        return False
    epochs, means, stds = smooth_series(epochs, means, stds, smooth_window)
    lower = [max(0.0, mean - std) for mean, std in zip(means, stds)]
    upper = [mean + std for mean, std in zip(means, stds)]
    ax.fill_between(epochs, lower, upper, color=color, alpha=alpha_fill, linewidth=0)
    ax.plot(
        epochs,
        means,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markersize=3.5,
        markevery=max(1, len(epochs) // 8),
        linewidth=linewidth,
    )
    return True


def plot_exp1_grid(plot_rows, out_dir, title_prefix, formats, dpi, smooth_window):
    summaries = summarize_rows(plot_rows, ["epoch", "branch", "signal"], METRICS)
    branches = unique_ordered([row["branch"] for row in summaries], BRANCH_ORDER)
    fig, axes = plt.subplots(
        len(METRICS),
        len(branches),
        figsize=(4.2 * len(branches), 3.25 * len(METRICS)),
        sharex=False,
    )
    if len(branches) == 1:
        axes = [[axes[idx]] for idx in range(len(METRICS))]

    for row_idx, metric in enumerate(METRICS):
        for col_idx, branch in enumerate(branches):
            ax = axes[row_idx][col_idx] if len(branches) > 1 else axes[row_idx][0]
            for signal in SIGNAL_ORDER:
                plot_summary_line(
                    ax,
                    summaries,
                    ["branch", "signal"],
                    [branch, signal],
                    metric,
                    SIGNAL_COLORS[signal],
                    SIGNAL_STYLES[signal],
                    SIGNAL_MARKERS[signal],
                    SIGNAL_WIDTHS[signal],
                    alpha_fill=0.15 if signal == "combined" else 0.11,
                    smooth_window=smooth_window,
                )
            ax.set_title(branch if row_idx == 0 else METRIC_LABELS[metric])
            ax.set_xlabel("Epoch" if row_idx == len(METRICS) - 1 else "")
            ax.set_ylabel(METRIC_LABELS[metric] if col_idx == 0 else "")
            ax.set_ylim(bottom=0)

    fig.legend(
        handles=signal_handles(style_color=None),
        frameon=False,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(f"{title_prefix}: branch-average mean +/- std", y=1.055)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, out_dir, "exp1_branch_average_metrics_mean_std", formats, dpi)


def plot_exp1_overlay(plot_rows, out_dir, title_prefix, formats, dpi, smooth_window):
    summaries = summarize_rows(plot_rows, ["epoch", "branch", "signal"], METRICS)
    branches = unique_ordered([row["branch"] for row in summaries], BRANCH_ORDER)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharex=True)

    for ax, metric in zip(axes, METRICS):
        for branch in branches:
            for signal in SIGNAL_ORDER:
                plot_summary_line(
                    ax,
                    summaries,
                    ["branch", "signal"],
                    [branch, signal],
                    metric,
                    BRANCH_COLORS.get(branch, "#6b7280"),
                    SIGNAL_STYLES[signal],
                    SIGNAL_MARKERS[signal],
                    SIGNAL_WIDTHS[signal],
                    alpha_fill=0.10 if signal == "combined" else 0.07,
                    smooth_window=smooth_window,
                )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_ylim(bottom=0)

    fig.legend(
        handles=branch_handles(branches),
        frameon=False,
        loc="lower center",
        ncol=len(branches),
        bbox_to_anchor=(0.5, 0.1),
    )
    fig.legend(
        handles=signal_handles(style_color="#111827"),
        frameon=False,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.07),
    )
    # fig.suptitle(f"{title_prefix}: branch-average overlay", y=0.995)
    fig.tight_layout(rect=(0, 0.18, 1, 0.93))
    save_figure(fig, out_dir, "exp1_branch_average_overlay_mean_std", formats, dpi)


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


def plot_exp2_grid(compare_rows, out_dir, title_prefix, formats, dpi, smooth_window):
    summaries = summarize_rows(compare_rows, ["epoch", "branch", "model"], ["false_attraction_rate"])
    branches = unique_ordered([row["branch"] for row in summaries], BRANCH_ORDER)
    fig, axes = plt.subplots(1, len(branches), figsize=(5.2 * len(branches), 4.2), sharex=False, sharey=True)
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, branch in zip(axes, branches):
        for model in MODEL_ORDER:
            plot_summary_line(
                ax,
                summaries,
                ["branch", "model"],
                [branch, model],
                "false_attraction_rate",
                MODEL_COLORS[model],
                MODEL_STYLES[model],
                MODEL_MARKERS[model],
                2.7 if model == "with prior" else 2.1,
                alpha_fill=0.14 if model == "with prior" else 0.10,
                smooth_window=smooth_window,
            )
        ax.set_title(branch)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("False attraction rate")
        ax.set_ylim(bottom=0)

    fig.legend(
        handles=model_handles(),
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.0),
    )
    # fig.suptitle(f"{title_prefix}: binary false attraction mean +/- std", y=1.055)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, out_dir, "exp2_branch_average_binary_far_mean_std", formats, dpi)


def plot_exp2_overlay(compare_rows, out_dir, title_prefix, formats, dpi, smooth_window):
    summaries = summarize_rows(compare_rows, ["epoch", "branch", "model"], ["false_attraction_rate"])
    branches = unique_ordered([row["branch"] for row in summaries], BRANCH_ORDER)
    fig, ax = plt.subplots(figsize=(8, 6))

    for branch in branches:
        for model in MODEL_ORDER:
            plot_summary_line(
                ax,
                summaries,
                ["branch", "model"],
                [branch, model],
                "false_attraction_rate",
                BRANCH_COLORS.get(branch, "#6b7280"),
                MODEL_STYLES[model],
                MODEL_MARKERS[model],
                2.7 if model == "with prior" else 2.0,
                alpha_fill=0.10 if model == "with prior" else 0.06,
                smooth_window=smooth_window,
            )
    # ax.set_title("False attraction rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("False attraction rate")
    ax.set_ylim(bottom=0)
    fig.legend(
        handles=branch_handles(branches),
        frameon=False,
        loc="lower center",
        ncol=len(branches),
        bbox_to_anchor=(0.5, 0.09),
    )
    fig.legend(
        handles=model_handles(style_color="#111827"),
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.03),
    )
    # fig.suptitle(f"{title_prefix}", y=0.995)
    fig.tight_layout(rect=(0, 0.18, 1, 0.93))
    save_figure(fig, out_dir, "exp2_branch_average_overlay_binary_far_mean_std", formats, dpi)


def format_float(value):
    if value is None:
        return ""
    return f"{value:.3f}"


def format_mean_std(row, metric):
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if mean is None:
        return ""
    return f"{mean:.3f} +/- {(std or 0.0):.3f}"


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def attraction_table_rows(final_rows):
    summaries = summarize_rows(final_rows, ["dataset_name", "branch", "signal"], METRICS)
    rows = []
    for row in sorted(
        summaries,
        key=lambda r: (
            r.get("dataset_name", ""),
            order_index(r.get("branch"), BRANCH_ORDER),
            order_index(r.get("signal"), SIGNAL_ORDER),
        ),
    ):
        out = {
            "dataset_name": row.get("dataset_name", ""),
            "branch": row.get("branch", ""),
            "signal": row.get("signal", ""),
            "precision": format_mean_std(row, "precision"),
            "recall": format_mean_std(row, "recall"),
            "false_attraction_rate": format_mean_std(row, "false_attraction_rate"),
            "precision_mean": format_float(row.get("precision_mean")),
            "precision_std": format_float(row.get("precision_std")),
            "recall_mean": format_float(row.get("recall_mean")),
            "recall_std": format_float(row.get("recall_std")),
            "false_attraction_rate_mean": format_float(row.get("false_attraction_rate_mean")),
            "false_attraction_rate_std": format_float(row.get("false_attraction_rate_std")),
            "n_runs": row.get("n_runs", ""),
            "epochs_used": row.get("epochs_used", ""),
        }
        out["_summary"] = row
        rows.append(out)
    return rows


def overlap_table_rows(final_rows):
    summaries = summarize_rows(final_rows, ["dataset_name", "branch", "group"], ["same_class_rate", "pair_percentage"])
    rows = []
    for row in sorted(
        summaries,
        key=lambda r: (
            r.get("dataset_name", ""),
            order_index(r.get("branch"), BRANCH_ORDER),
            order_index(r.get("group"), OVERLAP_GROUP_ORDER),
        ),
    ):
        out = {
            "dataset_name": row.get("dataset_name", ""),
            "branch": row.get("branch", ""),
            "quadrant": row.get("group", ""),
            "same_class_rate": format_mean_std(row, "same_class_rate"),
            "same_class_rate_mean": format_float(row.get("same_class_rate_mean")),
            "same_class_rate_std": format_float(row.get("same_class_rate_std")),
            "pair_percentage_mean": format_float(row.get("pair_percentage_mean")),
            "pair_percentage_std": format_float(row.get("pair_percentage_std")),
            "n_runs": row.get("n_runs", ""),
            "epochs_used": row.get("epochs_used", ""),
        }
        out["_summary"] = row
        rows.append(out)
    return rows


def write_attraction_table_csv(out_dir, rows):
    fieldnames = [
        "dataset_name",
        "branch",
        "signal",
        "precision",
        "recall",
        "false_attraction_rate",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "false_attraction_rate_mean",
        "false_attraction_rate_std",
        "n_runs",
        "epochs_used",
    ]
    clean_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    write_csv(os.path.join(out_dir, "fn_attraction_last_epoch_branch_average.csv"), fieldnames, clean_rows)


def write_overlap_table_csv(out_dir, rows):
    fieldnames = [
        "dataset_name",
        "branch",
        "quadrant",
        "same_class_rate",
        "same_class_rate_mean",
        "same_class_rate_std",
        "pair_percentage_mean",
        "pair_percentage_std",
        "n_runs",
        "epochs_used",
    ]
    clean_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    write_csv(os.path.join(out_dir, "fn_binary_prior_overlap_last_epoch_branch_average.csv"), fieldnames, clean_rows)


def best_cells_attraction(rows):
    best = set()
    grouped = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[(row["dataset_name"], row["branch"])].append((idx, row))
    for _, group_rows in grouped.items():
        for metric, maximize, col in [
            ("precision", True, 3),
            ("recall", True, 4),
            ("false_attraction_rate", False, 5),
        ]:
            values = [
                (idx, row["_summary"].get(f"{metric}_mean"))
                for idx, row in group_rows
                if row["_summary"].get(f"{metric}_mean") is not None
            ]
            if not values:
                continue
            best_value = max(value for _, value in values) if maximize else min(value for _, value in values)
            for idx, value in values:
                if value == best_value:
                    best.add((idx + 1, col))
    return best


def best_cells_overlap(rows):
    best = set()
    grouped = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[(row["dataset_name"], row["branch"])].append((idx, row))
    for _, group_rows in grouped.items():
        values = [
            (idx, row["_summary"].get("same_class_rate_mean"))
            for idx, row in group_rows
            if row["_summary"].get("same_class_rate_mean") is not None
        ]
        if not values:
            continue
        best_value = max(value for _, value in values)
        for idx, value in values:
            if value == best_value:
                best.add((idx + 1, 3))
    return best


def display_rows(rows, columns):
    displayed = []
    prev_dataset = None
    prev_branch = None
    for row in rows:
        out = []
        for col in columns:
            value = row.get(col, "")
            if col == "dataset_name":
                value = "" if value == prev_dataset else value
            if col == "branch":
                value = "" if value == prev_branch and row.get("dataset_name") == prev_dataset else value
            out.append(value)
        displayed.append(out)
        prev_dataset = row.get("dataset_name")
        prev_branch = row.get("branch")
    return displayed


def save_png_table(path, rows, columns, title, best_cells):
    if not rows:
        return
    header_labels = [col.replace("_", " ") for col in columns]
    cell_text = display_rows(rows, columns)
    height = max(2.4, 0.36 * (len(rows) + 2))
    width = max(7.0, 1.55 * len(columns))
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(PLOT_BACKGROUND_COLOR)
    ax.set_facecolor(PLOT_BACKGROUND_COLOR)
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=header_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.25)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#38bdf8")
        cell.set_linewidth(0.75)
        if row_idx == 0:
            cell.set_facecolor("#e0f2fe")
            cell.get_text().set_weight("bold")
        elif (row_idx, col_idx) in best_cells:
            cell.set_facecolor("#dbeafe")
            cell.get_text().set_weight("bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f8fafc")
    ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=300,
        facecolor=PLOT_BACKGROUND_COLOR,
        edgecolor=PLOT_BACKGROUND_COLOR,
        transparent=False,
    )
    plt.close(fig)


def save_tables(out_dir, with_attraction_rows, with_overlap_rows):
    attraction_final = final_epoch_rows_per_run(with_attraction_rows)
    attraction_rows = attraction_table_rows(attraction_final)
    write_attraction_table_csv(out_dir, attraction_rows)
    save_png_table(
        os.path.join(out_dir, "fn_attraction_last_epoch_branch_average_table.png"),
        attraction_rows,
        ["dataset_name", "branch", "signal", "precision", "recall", "false_attraction_rate"],
        "Last-epoch attraction statistics",
        best_cells_attraction(attraction_rows),
    )

    if with_overlap_rows:
        overlap_final = final_epoch_rows_per_run(with_overlap_rows)
        overlap_rows = overlap_table_rows(overlap_final)
        write_overlap_table_csv(out_dir, overlap_rows)
        save_png_table(
            os.path.join(out_dir, "fn_binary_prior_overlap_last_epoch_branch_average_table.png"),
            overlap_rows,
            ["dataset_name", "branch", "quadrant", "same_class_rate"],
            "Last-epoch binary-prior overlap",
            best_cells_overlap(overlap_rows),
        )
    else:
        print("No with-prior overlap CSV rows found; skipped overlap table.")


def main():
    args = parse_args()
    import_matplotlib()
    setup_style()
    os.makedirs(args.out_dir, exist_ok=True)

    with_raw, with_files = read_rows(args.with_prior, ATTRACTION_CSV, METRICS)
    if not with_raw:
        raise ValueError("No with-prior attraction rows found")
    with_rows = aggregate_attraction_rows(with_raw)
    with_overlap_raw, with_overlap_files = read_rows(
        args.with_prior,
        OVERLAP_CSV,
        ["same_class_rate", "pair_percentage"],
    )
    with_overlap_rows = aggregate_overlap_rows(with_overlap_raw)

    print(f"Read {len(with_files)} with-prior attraction CSV file(s)")
    print(f"Read {len(with_overlap_files)} with-prior overlap CSV file(s)")

    plot_exp1_grid(with_rows, args.out_dir, args.title_prefix, args.formats, args.dpi, args.smooth_window)
    plot_exp1_overlay(with_rows, args.out_dir, args.title_prefix, args.formats, args.dpi, args.smooth_window)
    save_tables(args.out_dir, with_rows, with_overlap_rows)

    if args.no_prior:
        no_raw, no_files = read_rows(args.no_prior, ATTRACTION_CSV, METRICS)
        if no_raw:
            no_rows = aggregate_attraction_rows(no_raw)
            compare_rows = prepare_model_comparison(
                with_rows,
                no_rows,
                args.with_prior_signal,
                args.no_prior_signal,
            )
            print(f"Read {len(no_files)} no-prior attraction CSV file(s)")
            plot_exp2_grid(compare_rows, args.out_dir, args.title_prefix, args.formats, args.dpi, args.smooth_window)
            plot_exp2_overlay(compare_rows, args.out_dir, args.title_prefix, args.formats, args.dpi, args.smooth_window)
        else:
            print("No no-prior attraction rows found; skipped Exp2 plots.")

    print(f"Saved all-fold analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
