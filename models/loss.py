
import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .filter_stats import build_filter_masks, update_filter_stats, update_label_agreement_stats
from .prior_bmm import prior_pair_value_matrix, prior_positive_probability_matrix, similarity_bmm_probability_matrix


def weighted_logsumexp(logits, weights, dim=1, eps=1e-12):
    weights = torch.clamp(weights, min=0.0)
    neg_inf = torch.full_like(weights, -torch.inf)
    log_weights = torch.where(weights > 0, torch.log(weights.clamp_min(eps)), neg_inf)
    return torch.logsumexp(logits + log_weights, dim=dim)


def prior_sample_distance_matrix(prior_features):
    return prior_pair_value_matrix(prior_features, {"metric": "euclidean"})


class InfoNCE(nn.Module):
    def __init__(self, temperature, device, filter=False, binary_classifier=None, scope_variable=0, filter_stats=None, stats_key=None, use_fn_mask=True, adaptive_filter_thresholds=False, prior_features=None, prior_bmm=None, prior_hard_neg_weight=1.0, prior_cancel_weighting=True, labels=None, agreement_prior_features=None, agreement_prior_bmm=None, prior_require_agreement=False, branch_agreement_decision=None, branch_agreement_weight=None, replace_binary_with_bmm=False):
        super(InfoNCE, self).__init__()
        self.device = device
        self.temperature = temperature
        self.filter = filter
        self.binary_classifier = binary_classifier
        self.scope_variable = scope_variable
        self.filter_stats = filter_stats
        self.stats_key = stats_key
        self.use_fn_mask = use_fn_mask
        self.adaptive_filter_thresholds = adaptive_filter_thresholds
        self.prior_features = prior_features
        self.prior_bmm = prior_bmm
        self.prior_hard_neg_weight = prior_hard_neg_weight
        self.prior_cancel_weighting = prior_cancel_weighting
        self.labels = labels
        self.agreement_prior_features = agreement_prior_features
        self.agreement_prior_bmm = agreement_prior_bmm
        self.prior_require_agreement = prior_require_agreement
        self.branch_agreement_decision = branch_agreement_decision
        self.branch_agreement_weight = branch_agreement_weight
        self.replace_binary_with_bmm = replace_binary_with_bmm
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

        if self.filter and (self.replace_binary_with_bmm or self.binary_classifier is not None):
            diag_mask = torch.eye(bsz, dtype=torch.bool, device=self.device)
            neg_mask = ~diag_mask
            if self.use_fn_mask:
                pos_neg_sim = torch.mm(out_2, out_2.t())
                # Previous FN mask:
                # FN_mask = (pos_neg_sim > neg_sim) & neg_mask
                delta = pos_sim.unsqueeze(1) - torch.minimum(neg_sim, pos_neg_sim)
                FN_mask = (delta > -0.1) & (delta < 0.1) & neg_mask
            else:
                FN_mask = neg_mask

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

            if self.prior_features is not None and self.prior_bmm is not None:
                prior_prob = prior_positive_probability_matrix(self.prior_features, self.prior_bmm)
                prior_decision = prior_prob > 0.5
                if (
                    self.prior_require_agreement
                    and self.agreement_prior_features is not None
                    and self.agreement_prior_bmm is not None
                ):
                    agreement_prior_prob = prior_positive_probability_matrix(
                        self.agreement_prior_features,
                        self.agreement_prior_bmm,
                    )
                    agreement_prior_decision = agreement_prior_prob > 0.5
                    prior_decision = prior_decision & agreement_prior_decision
                    prior_prob = torch.minimum(prior_prob, agreement_prior_prob)
                if self.branch_agreement_decision is not None:
                    branch_decision = self.branch_agreement_decision.to(device=self.device, dtype=torch.bool)
                    prior_decision = prior_decision & branch_decision
                    if self.branch_agreement_weight is not None:
                        branch_weight = self.branch_agreement_weight.to(device=self.device, dtype=prior_prob.dtype)
                        prior_prob = torch.minimum(prior_prob, branch_weight)
                candidate_mask = FN_mask & (binary_output > 0.5)
                attract_mask = candidate_mask & prior_decision
                # hard_neg_mask = candidate_mask & ~attract_mask
                cancel_mask = FN_mask & ((binary_output < 0.5) & prior_decision)
                hard_neg_mask = FN_mask & ((binary_output > 0.5) & ~prior_decision)
                threshold_stats = None

                neg_weight = neg_weight.masked_fill(attract_mask, 0.0)
                cancel_weight = torch.ones_like(neg_weight) - prior_prob if self.prior_cancel_weighting else torch.ones_like(neg_weight)
                neg_weight = torch.where(cancel_mask, cancel_weight, neg_weight)
                if isinstance(self.prior_hard_neg_weight, str) and self.prior_hard_neg_weight.lower() == "auto":
                    # hard_neg_weight = 1.0 + 0.001 * binary_output * (1.0 - prior_prob)
                    # hard_neg_weight = 1.0 + 0.0001 * binary_output

                    # downweight the hard neg part
                    hard_neg_weight = 1.0 - prior_prob
                    # hard_neg_weight = 1 - (((binary_output - 0.5) + prior_prob) /2)
                else:
                    hard_neg_weight = torch.full_like(neg_weight, float(self.prior_hard_neg_weight))
                neg_weight = torch.where(
                    hard_neg_mask,
                    hard_neg_weight,
                    neg_weight,
                )
                attract_weight = prior_prob
            else:
                prior_prob = None
                candidate_mask = None
                hard_neg_mask = None
                attract_mask, cancel_mask, threshold_stats = build_filter_masks(
                    FN_mask, binary_output, diag_mask, self.adaptive_filter_thresholds
                )
                neg_weight = neg_weight.masked_fill(attract_mask, 0.0)
                neg_weight = torch.where(cancel_mask, 1 - binary_output, neg_weight)
                attract_weight = binary_output

            update_filter_stats(
                self.filter_stats,
                self.stats_key,
                FN_mask,
                attract_mask,
                cancel_mask,
                binary_output,
                diag_mask,
                threshold_stats,
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

def loss_ntxent(features, device, filter_neg=False, binary_classifier=None, scope_variable=0, filter_stats=None, stats_key=None, use_fn_mask=True, adaptive_filter_thresholds=False, prior_features=None, prior_bmm=None, prior_hard_neg_weight=1.0, prior_cancel_weighting=True, labels=None, agreement_prior_features=None, agreement_prior_bmm=None, prior_require_agreement=False, branch_agreement_decision=None, branch_agreement_weight=None, replace_binary_with_bmm=False):
    fn = InfoNCE(
        temperature=0.2,
        device=device,
        filter=filter_neg,
        binary_classifier=binary_classifier,
        scope_variable=scope_variable,
        filter_stats=filter_stats,
        stats_key=stats_key,
        use_fn_mask=use_fn_mask,
        adaptive_filter_thresholds=adaptive_filter_thresholds,
        prior_features=prior_features,
        prior_bmm=prior_bmm,
        prior_hard_neg_weight=prior_hard_neg_weight,
        prior_cancel_weighting=prior_cancel_weighting,
        labels=labels,
        agreement_prior_features=agreement_prior_features,
        agreement_prior_bmm=agreement_prior_bmm,
        prior_require_agreement=prior_require_agreement,
        branch_agreement_decision=branch_agreement_decision,
        branch_agreement_weight=branch_agreement_weight,
        replace_binary_with_bmm=replace_binary_with_bmm,
    ) # topk: the percentage of each batch to be filtered
    pos_loss, pos_sim = fn(features[0], features[1])
    return pos_loss, pos_sim
