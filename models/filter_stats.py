import csv
import json
import os
import re

import torch


FN_ANALYSIS_KEY = "__fn_analysis__"


def format_branch_label(branch):
    """Convert internal modality branch keys into human-readable sensor labels."""
    if branch is None:
        return branch

    label = str(branch)
    label = re.sub(r"\bmod(\d+)\b", r"sensor\1", label)
    label = re.sub(
        r"inter:pair_sensor(\d+)_sensor(\d+)",
        r"inter:sensor\1 to sensor\2",
        label,
    )
    label = re.sub(
        r"inter:anchor_sensor(\d+)_sensor(\d+)",
        r"inter:anchor sensor\1 to sensor\2",
        label,
    )
    label = re.sub(
        r"inter:anchor_sensor(\d+)_(.+)",
        lambda match: f"inter:anchor sensor{match.group(1)} to {match.group(2).replace('_', ', ')}",
        label,
    )
    return label


def init_fn_analysis_stats(stats, epoch, threshold=0.5):
    if stats is None:
        return
    stats[FN_ANALYSIS_KEY] = {
        "epoch": epoch,
        "threshold": float(threshold),
        "branches": {},
    }


def fn_analysis_enabled(stats):
    return isinstance(stats, dict) and FN_ANALYSIS_KEY in stats


def _safe_div(numerator, denominator):
    if denominator == 0:
        return ""
    return numerator / denominator


def _safe_f1(precision, recall):
    if precision == "" or recall == "" or precision + recall == 0:
        return ""
    return 2 * precision * recall / (precision + recall)


def _default_signal_counts():
    return {
        "total_negative_pairs": 0,
        "true_false_negative_pairs": 0,
        "different_class_pairs": 0,
        "attracted_pairs": 0,
        "true_positive_attractions": 0,
        "false_positive_attractions": 0,
        "missed_true_false_negatives": 0,
    }


def _default_overlap_counts():
    return {
        "pair_count": 0,
        "same_class_count": 0,
        "binary_prob_sum": 0.0,
        "prior_prob_sum": 0.0,
    }


def _get_fn_branch(stats, branch):
    analysis = stats[FN_ANALYSIS_KEY]
    branches = analysis["branches"]
    if branch not in branches:
        branches[branch] = {
            "signals": {},
            "overlap": {},
        }
    return branches[branch]


def _update_signal_counts(branch_stats, signal, decision, true_same, non_diag_mask):
    decision = decision.bool() & non_diag_mask
    true_same = true_same.bool() & non_diag_mask
    different_class = (~true_same) & non_diag_mask

    counts = branch_stats["signals"].setdefault(signal, _default_signal_counts())
    counts["total_negative_pairs"] += int(non_diag_mask.sum().item())
    counts["true_false_negative_pairs"] += int(true_same.sum().item())
    counts["different_class_pairs"] += int(different_class.sum().item())
    counts["attracted_pairs"] += int(decision.sum().item())
    counts["true_positive_attractions"] += int((decision & true_same).sum().item())
    counts["false_positive_attractions"] += int((decision & different_class).sum().item())
    counts["missed_true_false_negatives"] += int(((~decision) & true_same).sum().item())


def _update_overlap_group(branch_stats, group, mask, true_same, binary_output, prior_prob):
    group_mask = mask.bool()
    count = int(group_mask.sum().item())
    counts = branch_stats["overlap"].setdefault(group, _default_overlap_counts())
    counts["pair_count"] += count
    if count == 0:
        return
    counts["same_class_count"] += int((group_mask & true_same).sum().item())
    counts["binary_prob_sum"] += float(binary_output[group_mask].detach().sum().item())
    counts["prior_prob_sum"] += float(prior_prob[group_mask].detach().sum().item())


def update_fn_analysis_stats(
    stats,
    branch,
    labels,
    diag_mask,
    binary_output=None,
    prior_prob=None,
    prior_decision=None,
    threshold=None,
):
    if not fn_analysis_enabled(stats) or labels is None:
        return

    with torch.no_grad():
        labels = labels.reshape(-1)
        non_diag_mask = ~diag_mask
        if int(non_diag_mask.sum().item()) == 0:
            return

        true_same = (labels.unsqueeze(0) == labels.unsqueeze(1)) & non_diag_mask
        branch_stats = _get_fn_branch(stats, branch)
        threshold = stats[FN_ANALYSIS_KEY]["threshold"] if threshold is None else float(threshold)

        binary_decision = None
        if binary_output is not None:
            binary_decision = binary_output > threshold
            _update_signal_counts(branch_stats, "binary", binary_decision, true_same, non_diag_mask)

        if prior_decision is None and prior_prob is not None:
            prior_decision = prior_prob > threshold
        if prior_decision is not None:
            _update_signal_counts(branch_stats, "prior", prior_decision, true_same, non_diag_mask)

        if binary_decision is not None and prior_decision is not None:
            combined_decision = binary_decision & prior_decision
            _update_signal_counts(branch_stats, "combined", combined_decision, true_same, non_diag_mask)

        if binary_decision is not None and prior_prob is not None:
            prior_overlap_decision = prior_decision if prior_decision is not None else prior_prob > threshold
            binary_yes = binary_decision & non_diag_mask
            binary_no = (~binary_decision) & non_diag_mask
            prior_yes = prior_overlap_decision.bool() & non_diag_mask
            prior_no = (~prior_overlap_decision.bool()) & non_diag_mask
            _update_overlap_group(
                branch_stats,
                "binary_yes_prior_yes",
                binary_yes & prior_yes,
                true_same,
                binary_output,
                prior_prob,
            )
            _update_overlap_group(
                branch_stats,
                "binary_yes_prior_no",
                binary_yes & prior_no,
                true_same,
                binary_output,
                prior_prob,
            )
            _update_overlap_group(
                branch_stats,
                "binary_no_prior_yes",
                binary_no & prior_yes,
                true_same,
                binary_output,
                prior_prob,
            )
            _update_overlap_group(
                branch_stats,
                "binary_no_prior_no",
                binary_no & prior_no,
                true_same,
                binary_output,
                prior_prob,
            )


def build_fn_analysis_rows(stats, context=None):
    if not fn_analysis_enabled(stats):
        return [], []
    context = context or {}
    analysis = stats[FN_ANALYSIS_KEY]
    base = {
        "epoch": analysis["epoch"] + 1,
        "threshold": analysis["threshold"],
        **context,
    }
    attraction_rows = []
    overlap_rows = []
    for branch, branch_stats in sorted(analysis["branches"].items()):
        branch_label = format_branch_label(branch)
        for signal, counts in sorted(branch_stats["signals"].items()):
            precision = _safe_div(counts["true_positive_attractions"], counts["attracted_pairs"])
            recall = _safe_div(counts["true_positive_attractions"], counts["true_false_negative_pairs"])
            attraction_rows.append({
                **base,
                "branch": branch_label,
                "signal": signal,
                **counts,
                "attraction_rate": _safe_div(counts["attracted_pairs"], counts["total_negative_pairs"]),
                "precision": precision,
                "recall": recall,
                "f1": _safe_f1(precision, recall),
                "false_attraction_rate": _safe_div(
                    counts["false_positive_attractions"],
                    counts["different_class_pairs"],
                ),
            })

        overlap_total = sum(group["pair_count"] for group in branch_stats["overlap"].values())
        for group, counts in sorted(branch_stats["overlap"].items()):
            overlap_rows.append({
                **base,
                "branch": branch_label,
                "group": group,
                "pair_count": counts["pair_count"],
                "pair_percentage": _safe_div(counts["pair_count"], overlap_total),
                "same_class_count": counts["same_class_count"],
                "same_class_rate": _safe_div(counts["same_class_count"], counts["pair_count"]),
                "mean_binary_probability": _safe_div(counts["binary_prob_sum"], counts["pair_count"]),
                "mean_prior_probability": _safe_div(counts["prior_prob_sum"], counts["pair_count"]),
            })
    return attraction_rows, overlap_rows


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_xlsx(path, rows):
    try:
        import pandas as pd
    except Exception:
        return False
    try:
        pd.DataFrame(rows).to_excel(path, index=False)
    except Exception:
        return False
    return True


def save_fn_analysis_outputs(output_dir, attraction_rows, overlap_rows, metadata=None):
    os.makedirs(output_dir, exist_ok=True)
    attraction_fields = [
        "dataset_name", "fold", "seed", "epoch", "threshold", "branch", "signal",
        "total_negative_pairs", "true_false_negative_pairs", "different_class_pairs",
        "attracted_pairs", "true_positive_attractions", "false_positive_attractions",
        "missed_true_false_negatives", "attraction_rate", "precision", "recall",
        "f1", "false_attraction_rate",
    ]
    overlap_fields = [
        "dataset_name", "fold", "seed", "epoch", "threshold", "branch", "group",
        "pair_count", "pair_percentage", "same_class_count", "same_class_rate",
        "mean_binary_probability", "mean_prior_probability",
    ]
    attraction_csv = os.path.join(output_dir, "fn_attraction_epoch_summary.csv")
    overlap_csv = os.path.join(output_dir, "fn_binary_prior_overlap.csv")
    _write_csv(attraction_csv, attraction_rows, attraction_fields)
    _write_csv(overlap_csv, overlap_rows, overlap_fields)

    _write_xlsx(
        os.path.join(output_dir, "fn_attraction_epoch_summary.xlsx"),
        attraction_rows,
    )
    _write_xlsx(
        os.path.join(output_dir, "fn_binary_prior_overlap.xlsx"),
        overlap_rows,
    )

    if metadata is not None:
        with open(os.path.join(output_dir, "fn_analysis_metadata.json"), "w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)


def update_filter_stats(
    stats,
    branch,
    fn_mask,
    attract_mask,
    # cancel_mask,  # Disabled: final method only uses attraction.
    binary_output,
    diag_mask,
    prior_prob=None,
    prior_candidate_mask=None,
    # prior_hard_neg_mask=None,  # Disabled: final method only uses attraction.
):
    if stats is None:
        return

    with torch.no_grad():
        non_diag_mask = ~diag_mask
        num_pairs = int(non_diag_mask.sum().item())
        if num_pairs == 0:
            return

        if binary_output is None:
            probs = torch.empty(0, device=diag_mask.device)
        else:
            probs = binary_output[non_diag_mask].detach()
        branch_stats = stats.setdefault(branch, {
            "calls": 0,
            "pairs": 0,
            "fn": 0,
            "attract": 0,
            # "cancel": 0,
            "prob_sum": 0.0,
            "prob_sq_sum": 0.0,
            "gt_09": 0,
            "uncertain_045_055": 0,
            "binary_gt_05": 0,
            "prior_candidates": 0,
            "prior_attract": 0,
            # "prior_hard_neg": 0,
            "prior_prob_sum": 0.0,
            "prior_prob_sq_sum": 0.0,
            "prior_prob_count": 0,
            "prior_gt_05": 0,
        })

        branch_stats["calls"] += 1
        branch_stats["pairs"] += num_pairs
        branch_stats["fn"] += int((fn_mask & non_diag_mask).sum().item())
        branch_stats["attract"] += int((attract_mask & non_diag_mask).sum().item())
        # branch_stats["cancel"] += int((cancel_mask & non_diag_mask).sum().item())
        if binary_output is not None:
            branch_stats["prob_sum"] += float(probs.sum().item())
            branch_stats["prob_sq_sum"] += float((probs * probs).sum().item())
            branch_stats["gt_09"] += int((probs > 0.9).sum().item())
            branch_stats["binary_gt_05"] += int((probs > 0.5).sum().item())
            branch_stats["uncertain_045_055"] += int(((probs > 0.45) & (probs < 0.55)).sum().item())
        if prior_prob is not None and prior_candidate_mask is not None:
            candidate_mask = prior_candidate_mask & non_diag_mask
            # hard_neg_mask = prior_hard_neg_mask & non_diag_mask if prior_hard_neg_mask is not None else torch.zeros_like(candidate_mask)
            candidate_count = int(candidate_mask.sum().item())
            branch_stats["prior_candidates"] += candidate_count
            branch_stats["prior_attract"] += int((attract_mask & non_diag_mask).sum().item())
            # branch_stats["prior_hard_neg"] += int(hard_neg_mask.sum().item())
            if candidate_count > 0:
                candidate_probs = prior_prob[candidate_mask].detach()
                branch_stats["prior_prob_sum"] += float(candidate_probs.sum().item())
                branch_stats["prior_prob_sq_sum"] += float((candidate_probs * candidate_probs).sum().item())
                branch_stats["prior_prob_count"] += candidate_count
                branch_stats["prior_gt_05"] += int((candidate_probs > 0.5).sum().item())


def _update_decision_agreement(branch_stats, prefix, decision, true_same, non_diag_mask):
    decision = decision.bool() & non_diag_mask
    correct = decision == true_same
    predicted_positive = decision
    true_positive = true_same

    branch_stats[f"{prefix}_label_pairs"] += int(non_diag_mask.sum().item())
    branch_stats[f"{prefix}_label_correct"] += int((correct & non_diag_mask).sum().item())
    branch_stats[f"{prefix}_pred_pos"] += int((predicted_positive & non_diag_mask).sum().item())
    branch_stats[f"{prefix}_true_pos"] += int((predicted_positive & true_positive & non_diag_mask).sum().item())
    branch_stats[f"{prefix}_false_pos"] += int((predicted_positive & ~true_positive & non_diag_mask).sum().item())
    branch_stats[f"{prefix}_false_neg"] += int((~predicted_positive & true_positive & non_diag_mask).sum().item())


def update_label_agreement_stats(
    stats,
    branch,
    labels,
    diag_mask,
    binary_output=None,
    prior_prob=None,
    binary_decision=None,
    prior_decision=None,
):
    if stats is None or labels is None:
        return

    with torch.no_grad():
        labels = labels.reshape(-1)
        non_diag_mask = ~diag_mask
        num_pairs = int(non_diag_mask.sum().item())
        if num_pairs == 0:
            return

        true_same = (labels.unsqueeze(0) == labels.unsqueeze(1)) & non_diag_mask
        branch_stats = stats.setdefault(branch, {})
        defaults = {
            "label_pairs": 0,
            "label_same": 0,
            "binary_label_correct": 0,
            "binary_label_pairs": 0,
            "binary_pred_pos": 0,
            "binary_true_pos": 0,
            "binary_false_pos": 0,
            "binary_false_neg": 0,
            "prior_label_correct": 0,
            "prior_label_pairs": 0,
            "prior_pred_pos": 0,
            "prior_true_pos": 0,
            "prior_false_pos": 0,
            "prior_false_neg": 0,
            "binary_prior_agree": 0,
            "binary_prior_pairs": 0,
        }
        for key, value in defaults.items():
            branch_stats.setdefault(key, value)

        branch_stats["label_pairs"] += num_pairs
        branch_stats["label_same"] += int(true_same.sum().item())

        if binary_decision is None and binary_output is not None:
            binary_decision = binary_output > 0.5
        if prior_decision is None and prior_prob is not None:
            prior_decision = prior_prob > 0.5

        if binary_decision is not None:
            _update_decision_agreement(branch_stats, "binary", binary_decision, true_same, non_diag_mask)
        if prior_decision is not None:
            _update_decision_agreement(branch_stats, "prior", prior_decision, true_same, non_diag_mask)
        if binary_decision is not None and prior_decision is not None:
            branch_stats["binary_prior_agree"] += int(((binary_decision.bool() == prior_decision.bool()) & non_diag_mask).sum().item())
            branch_stats["binary_prior_pairs"] += num_pairs
