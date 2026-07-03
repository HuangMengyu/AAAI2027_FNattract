import torch


def update_filter_stats(
    stats,
    branch,
    fn_mask,
    attract_mask,
    cancel_mask,
    binary_output,
    diag_mask,
    prior_prob=None,
    prior_candidate_mask=None,
    prior_hard_neg_mask=None,
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
            "cancel": 0,
            "prob_sum": 0.0,
            "prob_sq_sum": 0.0,
            "gt_09": 0,
            "uncertain_045_055": 0,
            "binary_gt_05": 0,
            "prior_candidates": 0,
            "prior_attract": 0,
            "prior_hard_neg": 0,
            "prior_prob_sum": 0.0,
            "prior_prob_sq_sum": 0.0,
            "prior_prob_count": 0,
            "prior_gt_05": 0,
        })

        branch_stats["calls"] += 1
        branch_stats["pairs"] += num_pairs
        branch_stats["fn"] += int((fn_mask & non_diag_mask).sum().item())
        branch_stats["attract"] += int((attract_mask & non_diag_mask).sum().item())
        branch_stats["cancel"] += int((cancel_mask & non_diag_mask).sum().item())
        if binary_output is not None:
            branch_stats["prob_sum"] += float(probs.sum().item())
            branch_stats["prob_sq_sum"] += float((probs * probs).sum().item())
            branch_stats["gt_09"] += int((probs > 0.9).sum().item())
            branch_stats["binary_gt_05"] += int((probs > 0.5).sum().item())
            branch_stats["uncertain_045_055"] += int(((probs > 0.45) & (probs < 0.55)).sum().item())
        if prior_prob is not None and prior_candidate_mask is not None:
            candidate_mask = prior_candidate_mask & non_diag_mask
            hard_neg_mask = prior_hard_neg_mask & non_diag_mask if prior_hard_neg_mask is not None else torch.zeros_like(candidate_mask)
            candidate_count = int(candidate_mask.sum().item())
            branch_stats["prior_candidates"] += candidate_count
            branch_stats["prior_attract"] += int((attract_mask & non_diag_mask).sum().item())
            branch_stats["prior_hard_neg"] += int(hard_neg_mask.sum().item())
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
