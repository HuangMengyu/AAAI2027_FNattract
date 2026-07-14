import torch
import torch.nn as nn
import numpy as np
from .attention import Seq_Transformer, Attention

from .filter_stats import update_filter_stats, update_fn_analysis_stats, update_label_agreement_stats
from .prior_bmm import (
    prior_pair_value_matrix,
    prior_positive_probability_matrix,
    similarity_bmm_probability_matrix,
)


def weighted_logsumexp(logits, weights, dim=1, eps=1e-12):
    weights = torch.clamp(weights, min=0.0)
    neg_inf = torch.full_like(weights, -torch.inf)
    log_weights = torch.where(weights > 0, torch.log(weights.clamp_min(eps)), neg_inf)
    return torch.logsumexp(logits + log_weights, dim=dim)


def prior_sample_distance_matrix(prior_features):
    return prior_pair_value_matrix(prior_features, {"metric": "euclidean"})


def prior_segment_distance_matrix(prior_features, segment_idx):
    return prior_pair_value_matrix(prior_features, {"metric": "euclidean"}, segment_idx=segment_idx)


class TC(nn.Module):
    def __init__(self, 
    device,
    final_out_channels=128, 
    timesteps=50, 
    hidden_dim=64,
    depth=4,
    ):
        super(TC, self).__init__()
        self.num_channels = final_out_channels
        self.timestep = timesteps
        self.Wk = nn.ModuleList([nn.Linear(hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.device = device
        
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, final_out_channels // 2),
            nn.BatchNorm1d(final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(final_out_channels // 2, final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=hidden_dim, depth=depth, heads=4, mlp_dim=64)

    def make_binary_pairs(self, anchors, targets):
        return torch.cat((anchors, targets), dim=1)

    def make_binary_pair_grid(self, anchors, targets):
        batch = anchors.shape[0]
        anchors = anchors.unsqueeze(1).expand(-1, batch, -1).reshape(-1, anchors.shape[1])
        targets = targets.unsqueeze(0).expand(batch, -1, -1).reshape(-1, targets.shape[1])
        return self.make_binary_pairs(anchors, targets)

    def project_full(self, features):
        sequence = features.transpose(1, 2)
        z_full = self.seq_transformer(sequence)
        c_full = self.projection_head(z_full)
        return z_full, c_full

    def form_temporal_binary_loss_data(self, pred, encode_samples):
        pairs = []
        labels = []

        for i in np.arange(0, self.timestep):
            anchors = pred[i].detach()
            targets = encode_samples[i].detach()
            batch_size = anchors.shape[0]

            similarity = torch.mm(anchors, torch.transpose(targets, 0, 1))
            pos_neg_similarity = torch.mm(targets, torch.transpose(targets, 0, 1))
            joint_similarity = torch.minimum(similarity, pos_neg_similarity)
            diag_mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
            least_similar_idx = torch.argmin(joint_similarity.masked_fill(diag_mask, float("inf")), dim=1)
            random_neg_idx = torch.randint(0, batch_size - 1, (batch_size,), device=self.device)
            anchor_idx = torch.arange(batch_size, device=self.device)
            random_neg_idx = random_neg_idx + (random_neg_idx >= anchor_idx).long()
            positive_scores = similarity[anchor_idx, anchor_idx]
            least_neg_scores = joint_similarity[anchor_idx, least_similar_idx]
            random_neg_scores = joint_similarity[anchor_idx, random_neg_idx]
            random_neg_labels = (
                (random_neg_scores - least_neg_scores)
                / (positive_scores - least_neg_scores).clamp_min(1e-8)
            ).clamp(0.0, 1.0)

            pos_pairs = self.make_binary_pairs(anchors, targets)
            least_similar_neg_pairs = self.make_binary_pairs(anchors, targets[least_similar_idx])
            random_neg_pairs = self.make_binary_pairs(anchors, targets[random_neg_idx])

            pairs.append(torch.cat((pos_pairs, least_similar_neg_pairs, random_neg_pairs), dim=0))
            labels.append(torch.cat((
                torch.ones(batch_size, device=self.device),
                torch.zeros(batch_size, device=self.device),
                random_neg_labels,
            ), dim=0))

        pairs = torch.cat(pairs, dim=0)
        labels = torch.cat(labels, dim=0)
        shuffle_idx = torch.randperm(pairs.shape[0], device=self.device)
        return pairs[shuffle_idx], labels[shuffle_idx]

    def forward(self, features_aug1, features_aug2, filter_neg=False, binary_classifier=None, scope_variable=0, filter_stats=None, stats_key=None, return_binary_data=False, prior_features=None, prior_bmm=None, prior_hard_neg_weight=1.0, prior_cancel_weighting=True, prior_is_segment=False, labels=None, external_filter_decision=None, external_filter_weight=None, replace_binary_with_bmm=False, fn_filter_use_binary=True, fn_filter_use_prior=True): # aug1 is used on z level, aug 2 is used on h level
        h_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        seq_len = h_aug1.shape[2]
        h_aug1 = h_aug1.transpose(1, 2)

        h_aug2 = features_aug2
        h_aug2 = h_aug2.transpose(1, 2)

        batch = h_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = h_aug2[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = h_aug1[:, :t_samples + 1, :]

        z_t = self.seq_transformer(forward_seq)

        # compute the nce loss
        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(z_t)

        temporal_binary_outputs = []
        temporal_prior_votes = []

        for i in np.arange(0, self.timestep):  # calculate the temporal loss per each timestep and then take average
            total = torch.mm(pred[i], torch.transpose(encode_samples[i], 0, 1))
            
            if filter_neg:
                # S>P: sim(pos_i, neg_j) > sim(anchor_i, neg_j)
                diag_mask = torch.eye(batch, dtype=torch.bool, device=self.device)
                neg_mask = ~diag_mask
                FN_mask = neg_mask

                use_binary_signal = fn_filter_use_binary and (
                    replace_binary_with_bmm or binary_classifier is not None or external_filter_decision is not None
                )
                use_prior_signal = (
                    fn_filter_use_prior
                    and prior_features is not None
                    and prior_bmm is not None
                )

                # use selected signals to further filter out false negatives
                if use_binary_signal or use_prior_signal:
                    binary_output = None
                    if use_binary_signal:
                        with torch.no_grad():
                            if replace_binary_with_bmm:
                                binary_output = similarity_bmm_probability_matrix(total)
                            elif binary_classifier is not None:
                                binary_input = self.make_binary_pair_grid(pred[i], encode_samples[i])
                                binary_output = torch.sigmoid(binary_classifier(binary_input))
                                binary_output = binary_output.reshape(batch, batch)
                            else:
                                binary_output = external_filter_decision.to(device=self.device, dtype=total.dtype)
                                if external_filter_weight is not None:
                                    binary_output = external_filter_weight.to(device=self.device, dtype=total.dtype)
                        temporal_binary_outputs.append(binary_output.detach())
                    neg_weight = neg_mask.float()

                    if use_prior_signal:
                        if prior_is_segment:
                            prior_seq_num = prior_features.shape[1]
                            target_pos = int(t_samples.item()) + i + 1
                            if seq_len > 1:
                                prior_segment_idx = int(round(target_pos * (prior_seq_num - 1) / (seq_len - 1)))
                            else:
                                prior_segment_idx = 0
                            prior_segment_idx = max(0, min(prior_segment_idx, prior_seq_num - 1))
                            prior_prob = prior_positive_probability_matrix(
                                prior_features,
                                prior_bmm,
                                segment_idx=prior_segment_idx,
                            )
                        else:
                            prior_prob = prior_positive_probability_matrix(prior_features, prior_bmm)
                        prior_decision = prior_prob > 0.5
                        if use_binary_signal and external_filter_decision is not None:
                            filter_decision = external_filter_decision.to(device=self.device, dtype=torch.bool)
                            prior_decision = prior_decision & filter_decision
                            if external_filter_weight is not None:
                                filter_weight = external_filter_weight.to(device=self.device, dtype=prior_prob.dtype)
                                prior_prob = torch.minimum(prior_prob, filter_weight)
                        temporal_prior_votes.append(prior_decision.detach())
                        attract_weight = prior_prob
                    else:
                        prior_prob = None
                        attract_weight = binary_output

                    if use_binary_signal and use_prior_signal:
                        candidate_mask = FN_mask & (binary_output > 0.5)
                        attract_mask = candidate_mask & prior_decision
                        cancel_mask = FN_mask & ((binary_output < 0.5) & prior_decision)
                        hard_neg_mask = FN_mask & ((binary_output > 0.5) & ~prior_decision)

                        cancel_weight = torch.ones_like(neg_weight) - prior_prob if prior_cancel_weighting else torch.ones_like(neg_weight)
                        neg_weight = torch.where(cancel_mask, cancel_weight, neg_weight)
                        if isinstance(prior_hard_neg_weight, str) and prior_hard_neg_weight.lower() == "auto":
                            hard_neg_weight = 1 - prior_prob
                        else:
                            hard_neg_weight = torch.full_like(neg_weight, float(prior_hard_neg_weight))
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
                        filter_stats,
                        stats_key,
                        FN_mask,
                        attract_mask,
                        cancel_mask,
                        binary_output,
                        diag_mask,
                        prior_prob=prior_prob,
                        prior_candidate_mask=candidate_mask,
                        prior_hard_neg_mask=hard_neg_mask,
                    )
                    update_fn_analysis_stats(
                        filter_stats,
                        stats_key,
                        labels,
                        diag_mask,
                        binary_output=binary_output,
                        prior_prob=prior_prob,
                    )

                    pos_weight = diag_mask.float() + attract_mask.float() * attract_weight
                    denom_weight = pos_weight + neg_weight
                    log_pos = weighted_logsumexp(total, pos_weight, dim=1)
                    log_den = weighted_logsumexp(total, denom_weight, dim=1)
                    nce = nce + torch.sum(log_pos - log_den)
                else:
                    nce = nce + torch.sum(torch.diag(self.lsoftmax(total)))
            else:
                nce = nce + torch.sum(torch.diag(self.lsoftmax(total)))

        if filter_neg and labels is not None and (temporal_binary_outputs or temporal_prior_votes):
            diag_mask = torch.eye(batch, dtype=torch.bool, device=self.device)
            binary_mean = torch.stack(temporal_binary_outputs, dim=0).mean(dim=0) if temporal_binary_outputs else None
            prior_decision = None
            if temporal_prior_votes:
                prior_vote_fraction = torch.stack([vote.float() for vote in temporal_prior_votes], dim=0).mean(dim=0)
                prior_decision = prior_vote_fraction > 0.5
            update_label_agreement_stats(
                filter_stats,
                stats_key,
                labels,
                diag_mask,
                binary_output=binary_mean,
                prior_decision=prior_decision,
            )
        nce /= -1. * batch * self.timestep

        z_full = self.seq_transformer(h_aug1)
        c_full = self.projection_head(z_full)

        if return_binary_data:
            binary_pairs, binary_labels = self.form_temporal_binary_loss_data(pred, encode_samples)
            return nce, z_full, c_full, binary_pairs, binary_labels

        return nce, z_full, c_full
