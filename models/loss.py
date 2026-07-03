
import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .filter_stats import update_filter_stats, update_label_agreement_stats
from .prior_bmm import prior_pair_value_matrix, prior_positive_probability_matrix, similarity_bmm_probability_matrix


def weighted_logsumexp(logits, weights, dim=1, eps=1e-12):
    weights = torch.clamp(weights, min=0.0)
    neg_inf = torch.full_like(weights, -torch.inf)
    log_weights = torch.where(weights > 0, torch.log(weights.clamp_min(eps)), neg_inf)
    return torch.logsumexp(logits + log_weights, dim=dim)


def prior_sample_distance_matrix(prior_features):
    return prior_pair_value_matrix(prior_features, {"metric": "euclidean"})


class InfoNCE(nn.Module):
    def __init__(self, temperature, device, filter=False, binary_classifier=None, scope_variable=0, filter_stats=None, stats_key=None, prior_features=None, prior_bmm=None, prior_hard_neg_weight=1.0, prior_cancel_weighting=True, labels=None, replace_binary_with_bmm=False, fn_filter_use_binary=True, fn_filter_use_prior=True):
        super(InfoNCE, self).__init__()
        self.device = device
        self.temperature = temperature
        self.filter = filter
        self.binary_classifier = binary_classifier
        self.scope_variable = scope_variable
        self.filter_stats = filter_stats
        self.stats_key = stats_key
        self.prior_features = prior_features
        self.prior_bmm = prior_bmm
        self.prior_hard_neg_weight = prior_hard_neg_weight
        self.prior_cancel_weighting = prior_cancel_weighting
        self.labels = labels
        self.replace_binary_with_bmm = replace_binary_with_bmm
        self.fn_filter_use_binary = fn_filter_use_binary
        self.fn_filter_use_prior = fn_filter_use_prior
        self.tau = temperature # scaling factor for masks


    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, out_1, out_2):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        bsz = out_1.shape[0]

        raw_out_1 = out_1
        raw_out_2 = out_2
        out_1 = F.normalize(out_1, dim=1)
        out_2 = F.normalize(out_2, dim=1)


        pos_sim = torch.sum(out_1 * out_2, dim=-1)
        neg_sim = torch.mm(out_1, out_2.t())

        use_binary_signal = self.fn_filter_use_binary and (
            self.replace_binary_with_bmm or self.binary_classifier is not None
        )
        use_prior_signal = (
            self.fn_filter_use_prior
            and self.prior_features is not None
            and self.prior_bmm is not None
        )

        if self.filter and (use_binary_signal or use_prior_signal):
            diag_mask = torch.eye(bsz, dtype=torch.bool, device=self.device)
            neg_mask = ~diag_mask
            FN_mask = neg_mask

            binary_output = None
            if use_binary_signal:
                with torch.no_grad():
                    if self.replace_binary_with_bmm:
                        binary_output = similarity_bmm_probability_matrix(neg_sim)
                    else:
                        binary_input = torch.cat((
                            raw_out_1.unsqueeze(1).expand(-1, bsz, -1).reshape(-1, raw_out_1.shape[1]),
                            raw_out_2.unsqueeze(0).expand(bsz, -1, -1).reshape(-1, raw_out_2.shape[1]),
                        ), dim=1)
                        binary_output = torch.sigmoid(self.binary_classifier(binary_input)).reshape(bsz, bsz)

            neg_weight = neg_mask.float()

            if use_prior_signal:
                prior_prob = prior_positive_probability_matrix(self.prior_features, self.prior_bmm)
                prior_decision = prior_prob > 0.5
                attract_weight = prior_prob
            else:
                prior_prob = None
                attract_weight = binary_output

            if use_binary_signal and use_prior_signal:
                candidate_mask = FN_mask & (binary_output > 0.5)
                attract_mask = candidate_mask & prior_decision
                cancel_mask = FN_mask & ((binary_output < 0.5) & prior_decision)
                hard_neg_mask = FN_mask & ((binary_output > 0.5) & ~prior_decision)

                cancel_weight = torch.ones_like(neg_weight) - prior_prob if self.prior_cancel_weighting else torch.ones_like(neg_weight)
                neg_weight = torch.where(cancel_mask, cancel_weight, neg_weight)
                if isinstance(self.prior_hard_neg_weight, str) and self.prior_hard_neg_weight.lower() == "auto":
                    hard_neg_weight = 1.0 - prior_prob
                else:
                    hard_neg_weight = torch.full_like(neg_weight, float(self.prior_hard_neg_weight))
                neg_weight = torch.where(hard_neg_mask, hard_neg_weight, neg_weight)
            elif use_binary_signal:
                candidate_mask = None
                hard_neg_mask = None
                attract_mask = FN_mask & (binary_output > 0.5)
                cancel_mask = torch.zeros_like(FN_mask)
            else:
                candidate_mask = FN_mask
                hard_neg_mask = torch.zeros_like(FN_mask)
                attract_mask = FN_mask & prior_decision
                cancel_mask = torch.zeros_like(FN_mask)

            neg_weight = neg_weight.masked_fill(attract_mask, 0.0)

            update_filter_stats(
                self.filter_stats,
                self.stats_key,
                FN_mask,
                attract_mask,
                cancel_mask,
                binary_output,
                diag_mask,
                prior_prob=prior_prob,
                prior_candidate_mask=candidate_mask,
                prior_hard_neg_mask=hard_neg_mask,
            )
            update_label_agreement_stats(
                self.filter_stats,
                self.stats_key,
                self.labels,
                diag_mask,
                binary_output=binary_output,
                prior_prob=prior_prob,
            )

            scaled_sim = neg_sim / self.temperature
            pos_weight = diag_mask.float() + attract_mask.float() * attract_weight
            denom_weight = pos_weight + neg_weight
            log_pos = weighted_logsumexp(scaled_sim, pos_weight, dim=1)
            log_den = weighted_logsumexp(scaled_sim, denom_weight, dim=1)
            logits = log_den - log_pos
        else:
            scaled_sim = neg_sim / self.temperature
            logits = torch.logsumexp(scaled_sim, dim=1) - torch.diag(scaled_sim)

        logits_re = logits
        pos_loss = torch.mean(logits_re, dim=0)

        return pos_loss, pos_sim

def loss_ntxent(features, device, filter_neg=False, binary_classifier=None, scope_variable=0, filter_stats=None, stats_key=None, prior_features=None, prior_bmm=None, prior_hard_neg_weight=1.0, prior_cancel_weighting=True, labels=None, replace_binary_with_bmm=False, fn_filter_use_binary=True, fn_filter_use_prior=True):
    fn = InfoNCE(
        temperature=0.2,
        device=device,
        filter=filter_neg,
        binary_classifier=binary_classifier,
        scope_variable=scope_variable,
        filter_stats=filter_stats,
        stats_key=stats_key,
        prior_features=prior_features,
        prior_bmm=prior_bmm,
        prior_hard_neg_weight=prior_hard_neg_weight,
        prior_cancel_weighting=prior_cancel_weighting,
        labels=labels,
        replace_binary_with_bmm=replace_binary_with_bmm,
        fn_filter_use_binary=fn_filter_use_binary,
        fn_filter_use_prior=fn_filter_use_prior,
    ) # topk: the percentage of each batch to be filtered
    pos_loss, pos_sim = fn(features[0], features[1])
    return pos_loss, pos_sim


def loss_ntxent_anchor_vs_modalities(
    anchor_feature,
    positive_features,
    device,
    filter_neg=False,
    branch_configs=None,
    scope_variable=0,
    filter_stats=None,
    prior_hard_neg_weight=1.0,
    prior_cancel_weighting=True,
    labels=None,
    replace_binary_with_bmm=False,
    fn_filter_use_binary=True,
    fn_filter_use_prior=True,
    temperature=0.2,
):
    anchor_raw = anchor_feature
    positive_raw = positive_features
    anchor_feature = F.normalize(anchor_feature, dim=1)
    positive_features = [F.normalize(feature, dim=1) for feature in positive_features]
    branch_configs = branch_configs or [{} for _ in positive_features]

    if len(branch_configs) != len(positive_features):
        raise ValueError(
            f"Expected one branch config per positive modality, got {len(branch_configs)} "
            f"configs for {len(positive_features)} positives"
        )

    batch_size = anchor_feature.shape[0]
    diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    sim_chunks = []
    pos_weight_chunks = []
    denom_weight_chunks = []
    pos_sim_chunks = []

    for branch_idx, (positive_feature, positive_raw_feature, branch_config) in enumerate(
        zip(positive_features, positive_raw, branch_configs)
    ):
        sim_chunk = torch.mm(anchor_feature, positive_feature.t())
        pos_sim = torch.diag(sim_chunk)
        pos_sim_chunks.append(pos_sim)
        neg_mask = ~diag_mask

        pos_weight = diag_mask.float()
        neg_weight = neg_mask.float()

        use_binary_signal = fn_filter_use_binary and (
            replace_binary_with_bmm or branch_config.get("binary_classifier") is not None
        )
        prior_features = branch_config.get("prior_features")
        prior_bmm = branch_config.get("prior_bmm")
        use_prior_signal = (
            fn_filter_use_prior
            and prior_features is not None
            and prior_bmm is not None
        )

        if filter_neg and (use_binary_signal or use_prior_signal):
            fn_mask = neg_mask

            binary_output = None
            if use_binary_signal:
                with torch.no_grad():
                    if replace_binary_with_bmm:
                        binary_output = similarity_bmm_probability_matrix(sim_chunk)
                    else:
                        binary_input = torch.cat((
                            anchor_raw.unsqueeze(1).expand(-1, batch_size, -1).reshape(-1, anchor_raw.shape[1]),
                            positive_raw_feature.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, positive_raw_feature.shape[1]),
                        ), dim=1)
                        binary_classifier = branch_config["binary_classifier"]
                        binary_output = torch.sigmoid(binary_classifier(binary_input)).reshape(batch_size, batch_size)

            if use_prior_signal:
                prior_prob = prior_positive_probability_matrix(prior_features, prior_bmm)
                prior_decision = prior_prob > 0.5
                attract_weight = prior_prob
            else:
                prior_prob = None
                attract_weight = binary_output

            if use_binary_signal and use_prior_signal:
                candidate_mask = fn_mask & (binary_output > 0.5)
                attract_mask = candidate_mask & prior_decision
                cancel_mask = fn_mask & ((binary_output < 0.5) & prior_decision)
                hard_neg_mask = fn_mask & ((binary_output > 0.5) & ~prior_decision)

                cancel_weight = torch.ones_like(neg_weight) - prior_prob if prior_cancel_weighting else torch.ones_like(neg_weight)
                neg_weight = torch.where(cancel_mask, cancel_weight, neg_weight)
                if isinstance(prior_hard_neg_weight, str) and prior_hard_neg_weight.lower() == "auto":
                    hard_neg_weight = 1.0 - prior_prob
                else:
                    hard_neg_weight = torch.full_like(neg_weight, float(prior_hard_neg_weight))
                neg_weight = torch.where(hard_neg_mask, hard_neg_weight, neg_weight)
            elif use_binary_signal:
                candidate_mask = None
                hard_neg_mask = None
                attract_mask = fn_mask & (binary_output > 0.5)
                cancel_mask = torch.zeros_like(fn_mask)
            else:
                candidate_mask = fn_mask
                hard_neg_mask = torch.zeros_like(fn_mask)
                attract_mask = fn_mask & prior_decision
                cancel_mask = torch.zeros_like(fn_mask)

            neg_weight = neg_weight.masked_fill(attract_mask, 0.0)

            update_filter_stats(
                filter_stats,
                branch_config.get("stats_key", f"inter_anchor_branch_{branch_idx}"),
                fn_mask,
                attract_mask,
                cancel_mask,
                binary_output,
                diag_mask,
                prior_prob=prior_prob,
                prior_candidate_mask=candidate_mask,
                prior_hard_neg_mask=hard_neg_mask,
            )
            update_label_agreement_stats(
                filter_stats,
                branch_config.get("stats_key", f"inter_anchor_branch_{branch_idx}"),
                labels,
                diag_mask,
                binary_output=binary_output,
                prior_prob=prior_prob,
            )
            pos_weight = diag_mask.float() + attract_mask.float() * attract_weight

        sim_chunks.append(sim_chunk / temperature)
        pos_weight_chunks.append(pos_weight)
        denom_weight_chunks.append(pos_weight + neg_weight)

    scaled_sim = torch.cat(sim_chunks, dim=1)
    pos_weight = torch.cat(pos_weight_chunks, dim=1)
    denom_weight = torch.cat(denom_weight_chunks, dim=1)
    log_pos = weighted_logsumexp(scaled_sim, pos_weight, dim=1)
    log_den = weighted_logsumexp(scaled_sim, denom_weight, dim=1)
    pos_loss = torch.mean(log_den - log_pos, dim=0)
    pos_sim = torch.stack(pos_sim_chunks, dim=1).mean(dim=1)
    return pos_loss, pos_sim
