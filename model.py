import numpy as np
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TC import TC
from models.loss import loss_ntxent
from models.prior_bmm import (
    prepare_prior_for_torch,
    prior_positive_probability_matrix,
    similarity_bmm_probability_matrix,
)

import torch.optim as optim

from tqdm import trange
from scipy.stats import norm



class base_Model(nn.Module): # encoder
    def __init__(self, input_channels=3, final_out_channels = 128, kernel_size = 25, stride = 3, features_len = 127):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes, bn = True):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.hidden_channels = hidden_channels
        self.fc1 = nn.Linear(self.in_channels, self.hidden_channels)
        self.fc2 = nn.Linear(self.hidden_channels, self.n_classes)
        self.ac = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_channels)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.ac(hidden)
        hidden = self.bn(hidden)
        out = self.fc2(hidden)

        return out

class TFCC(nn.Module):
    def __init__(self, device,
        EEG_encoder,
        EOG_encoder,
        final_out_channels = 128, 
        timesteps=50, 
        hidden_dim=64,
        batch_size=32,
        num_epochs=100,
        warm_epochs=50,
        adaptive_warmup_threshold=0.08,
        lr = 1e-4,
        use_iteration=False,
        num_iter=600,
        depth=4,
        filter_temporal=True,
        filter_intra=True,
        filter_inter=True,
        use_fn_mask=True,
        adaptive_filter_thresholds=False,
        use_prior=False,
        prior_info=None,
        prior_hard_neg_weight=1.0,
        prior_cancel_weighting=True,
        prior_require_modality_agreement=False,
        prior_require_within_modality_agreement=False,
        replace_binary_with_bmm=False,
        temporal_binary_mode=None,
        intra_binary_mode=None,
        inter_binary_mode=None,
        use_intra_sample_for_temporal_filter=False,
        pretrain_labels=None,
        ):
        super(TFCC, self).__init__()
        self.EEG_encoder = EEG_encoder
        self.EOG_encoder = EOG_encoder  
        self.EEG_contrasting = TC(device, final_out_channels, timesteps, hidden_dim, depth)
        self.EOG_contrasting = TC(device, final_out_channels, timesteps, hidden_dim, depth)
        self.batch_size = batch_size
        self.device = device
        self.num_epochs = num_epochs
        self.adaptive_warmup_threshold = adaptive_warmup_threshold
        self.adaptive_warmup = isinstance(warm_epochs, str) and warm_epochs.lower() == "adaptive"
        if self.adaptive_warmup:
            self.warm_epochs = 0
            self.filter_start_epoch = None
        else:
            self.warm_epochs = int(warm_epochs)
            self.filter_start_epoch = self.warm_epochs
        self.device = device

        self.shared_projection_head_1 = nn.Sequential(
            nn.Linear(hidden_dim, final_out_channels // 2),
            nn.BatchNorm1d(final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(final_out_channels // 2, final_out_channels // 4),
        )

        self.shared_projection_head_2 = nn.Sequential(
            nn.Linear(hidden_dim, final_out_channels // 2),
            nn.BatchNorm1d(final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(final_out_channels // 2, final_out_channels // 4),
        )

        self.temporal_binary = nn.Sequential(
            nn.Linear(final_out_channels * 2, final_out_channels),
            nn.BatchNorm1d(final_out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(final_out_channels, 1)
        )
        binary_projection_dim = (final_out_channels // 4) * 2
        self.intra_binary = nn.Sequential(
            nn.Linear(binary_projection_dim, final_out_channels // 4),
            nn.BatchNorm1d(final_out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(final_out_channels // 4, 1)
        )
        self.inter_binary = nn.Sequential(
            nn.Linear(binary_projection_dim, final_out_channels // 4),
            nn.BatchNorm1d(final_out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(final_out_channels // 4, 1)
        )

        self.optimizer = optim.AdamW([{'params':self.EEG_encoder.parameters()},
                                        {'params':self.EOG_encoder.parameters()},
                                        {'params':self.EEG_contrasting.parameters()},
                                        {'params':self.EOG_contrasting.parameters()},
                                        {'params':self.shared_projection_head_1.parameters()},
                                        {'params':self.shared_projection_head_2.parameters()}], lr, betas=(0.5, 0.99), weight_decay=3e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=20)

        self.binary_optimizer = optim.AdamW(
            [{'params':self.temporal_binary.parameters()},
            {'params':self.intra_binary.parameters()},
            {'params':self.inter_binary.parameters()},], 
            lr, betas=(0.5, 0.99), weight_decay=3e-4
        )

        self.use_iteration = use_iteration
        self.num_iter = num_iter
        self.filter_temporal = filter_temporal
        self.filter_intra = filter_intra
        self.filter_inter = filter_inter
        self.use_fn_mask = use_fn_mask
        self.adaptive_filter_thresholds = adaptive_filter_thresholds
        self.use_prior = use_prior and prior_info is not None
        self.prior = prepare_prior_for_torch(prior_info) if self.use_prior else None
        self.prior_mode = self.prior["metadata"].get("prior_mode", "combined") if self.prior is not None else "combined"
        self.prior_hard_neg_weight = prior_hard_neg_weight
        self.prior_cancel_weighting = prior_cancel_weighting
        self.prior_require_modality_agreement = prior_require_modality_agreement
        self.prior_require_within_modality_agreement = prior_require_within_modality_agreement
        self.replace_binary_with_bmm = replace_binary_with_bmm
        self.temporal_binary_mode = self.normalize_binary_mode(temporal_binary_mode, replace_binary_with_bmm)
        self.intra_binary_mode = self.normalize_binary_mode(intra_binary_mode, replace_binary_with_bmm)
        self.inter_binary_mode = self.normalize_binary_mode(inter_binary_mode, replace_binary_with_bmm)
        self.use_intra_sample_for_temporal_filter = use_intra_sample_for_temporal_filter
        self.pretrain_labels = torch.tensor(pretrain_labels, dtype=torch.long) if pretrain_labels is not None else None

        self.scope_variable = 0 if self.adaptive_warmup else self.warm_epochs / self.num_epochs

    def normalize_binary_mode(self, mode, replace_binary_with_bmm):
        if mode is None:
            return "bmm" if replace_binary_with_bmm else "binary"
        mode = mode.lower()
        if mode not in ("binary", "bmm"):
            raise ValueError(f"Unsupported binary mode: {mode}")
        return mode

    def branch_uses_bmm(self, branch):
        if branch == "temporal":
            return self.temporal_binary_mode == "bmm"
        if branch == "intra":
            return self.intra_binary_mode == "bmm"
        if branch == "inter":
            return self.inter_binary_mode == "bmm"
        raise ValueError(f"Unsupported branch: {branch}")

    def should_filter_epoch(self, epoch):
        if self.adaptive_warmup:
            return self.filter_start_epoch is not None and epoch >= self.filter_start_epoch
        return epoch >= self.warm_epochs

    def update_scope_variable(self, epoch):
        filter_start_epoch = self.filter_start_epoch if self.filter_start_epoch is not None else self.warm_epochs
        denominator = max(self.num_epochs - filter_start_epoch, 1)
        self.scope_variable = 0.1 * (epoch - filter_start_epoch) / denominator

    def should_end_adaptive_warmup(self, previous_loss, current_loss, previous_binary_loss, current_binary_loss):
        loss_drop = (previous_loss - current_loss) / (previous_loss + 1e-6)
        binary_loss_drop = (previous_binary_loss - current_binary_loss) / (previous_binary_loss + 1e-6)
        return loss_drop <= self.adaptive_warmup_threshold and binary_loss_drop <= self.adaptive_warmup_threshold, loss_drop, binary_loss_drop

    def uses_any_binary_classifier(self):
        return (
            (self.temporal_binary_mode == "binary" and not self.use_intra_sample_for_temporal_filter)
            or self.intra_binary_mode == "binary"
            or self.inter_binary_mode == "binary"
        )

    def get_prior_features(self, modality, index):
        if not self.use_prior or self.prior is None or index is None:
            return None
        index = index.detach().cpu().long()
        return self.prior["features"][modality][index].to(self.device)

    def get_combined_prior_features(self, index):
        return self.get_prior_features("combined", index)

    def get_prior_bmm(self, key):
        if not self.use_prior or self.prior is None:
            return None
        return self.prior["bmm"].get(key)

    def get_batch_labels(self, index):
        if self.pretrain_labels is None or index is None:
            return None
        index = index.detach().cpu().long()
        return self.pretrain_labels[index].to(self.device)

    def set_modules_eval(self, modules):
        states = []
        for module in modules:
            states.append(module.training)
            module.eval()
        return states

    def restore_module_states(self, modules, states):
        for module, state in zip(modules, states):
            module.train(state)

    def make_sample_binary_matrix(self, left, right, binary_classifier):
        batch_size = left.shape[0]
        binary_input = torch.cat((
            left.unsqueeze(1).expand(-1, batch_size, -1).reshape(-1, left.shape[1]),
            right.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, right.shape[1]),
        ), dim=1)
        return torch.sigmoid(binary_classifier(binary_input)).reshape(batch_size, batch_size)

    def make_sample_bmm_matrix(self, left, right):
        batch_size = left.shape[0]
        left = F.normalize(left.reshape(batch_size, -1), dim=1)
        right = F.normalize(right.reshape(batch_size, -1), dim=1)
        similarity = torch.mm(left, right.t())
        return similarity_bmm_probability_matrix(similarity)

    def sample_branch_decision(
        self,
        left,
        right,
        binary_classifier,
        prior_features=None,
        prior_bmm=None,
        agreement_prior_features=None,
        agreement_prior_bmm=None,
        use_prior_agreement=False,
        use_bmm=False,
    ):
        if use_bmm:
            binary_output = self.make_sample_bmm_matrix(left, right)
        else:
            binary_output = self.make_sample_binary_matrix(left, right, binary_classifier)
        branch_prob = binary_output
        branch_decision = binary_output > 0.5
        if prior_features is not None and prior_bmm is not None:
            prior_prob = prior_positive_probability_matrix(prior_features, prior_bmm)
            prior_decision = prior_prob > 0.5
            if use_prior_agreement and agreement_prior_features is not None and agreement_prior_bmm is not None:
                agreement_prob = prior_positive_probability_matrix(agreement_prior_features, agreement_prior_bmm)
                prior_decision = prior_decision & (agreement_prob > 0.5)
                prior_prob = torch.minimum(prior_prob, agreement_prob)
            branch_decision = branch_decision & prior_decision
            branch_prob = torch.minimum(binary_output, prior_prob)
        return branch_decision, branch_prob

    def majority_branch_agreement(self, decisions, weights):
        stacked_decisions = torch.stack([decision.float() for decision in decisions], dim=0)
        majority_decision = stacked_decisions.sum(dim=0) >= 2.0
        stacked_weights = torch.stack(weights, dim=0)
        majority_weight = stacked_weights.min(dim=0).values
        return majority_decision, majority_weight

    def build_within_modality_agreement(
        self,
        EEG_feat,
        EEG_aug_feat,
        EOG_feat,
        EOG_aug_feat,
        mod1_prior,
        mod2_prior,
        mod1_sample_prior_bmm,
        mod2_sample_prior_bmm,
        mod1_segment_prior_bmm,
        mod2_segment_prior_bmm,
        use_prior_agreement,
    ):
        modules = [
            self.EEG_contrasting,
            self.EOG_contrasting,
            self.temporal_binary,
            self.intra_binary,
            self.inter_binary,
            self.shared_projection_head_1,
            self.shared_projection_head_2,
        ]
        states = self.set_modules_eval(modules)
        try:
            with torch.no_grad():
                seq_len_mod1 = EEG_feat.shape[2]
                seq_len_mod2 = EOG_feat.shape[2]
                t_mod1 = torch.randint(seq_len_mod1 - self.EEG_contrasting.timestep, size=(1,), device=self.device).long()
                t_mod1_aug = torch.randint(seq_len_mod1 - self.EEG_contrasting.timestep, size=(1,), device=self.device).long()
                t_mod2 = torch.randint(seq_len_mod2 - self.EOG_contrasting.timestep, size=(1,), device=self.device).long()
                t_mod2_aug = torch.randint(seq_len_mod2 - self.EOG_contrasting.timestep, size=(1,), device=self.device).long()

                z_T, c_T = self.EEG_contrasting.project_full(EEG_feat)
                z_T_aug, c_T_aug = self.EEG_contrasting.project_full(EEG_aug_feat)
                z_F, c_F = self.EOG_contrasting.project_full(EOG_feat)
                z_F_aug, c_F_aug = self.EOG_contrasting.project_full(EOG_aug_feat)
                c_T_shared = self.shared_projection_head_1(z_T)
                c_F_shared = self.shared_projection_head_2(z_F)

                temporal_mod1_1, temporal_mod1_weight_1 = self.EEG_contrasting.temporal_decision_summary(
                    EEG_feat,
                    EEG_aug_feat,
                    t_mod1,
                    self.temporal_binary,
                    replace_binary_with_bmm=self.branch_uses_bmm("temporal"),
                    prior_features=mod1_prior,
                    prior_bmm=mod1_segment_prior_bmm,
                    prior_is_segment=True,
                    agreement_prior_features=mod2_prior,
                    agreement_prior_bmm=mod2_segment_prior_bmm,
                    prior_require_agreement=use_prior_agreement,
                )
                temporal_mod1_2, temporal_mod1_weight_2 = self.EEG_contrasting.temporal_decision_summary(
                    EEG_aug_feat,
                    EEG_feat,
                    t_mod1_aug,
                    self.temporal_binary,
                    replace_binary_with_bmm=self.branch_uses_bmm("temporal"),
                    prior_features=mod1_prior,
                    prior_bmm=mod1_segment_prior_bmm,
                    prior_is_segment=True,
                    agreement_prior_features=mod2_prior,
                    agreement_prior_bmm=mod2_segment_prior_bmm,
                    prior_require_agreement=use_prior_agreement,
                )
                temporal_mod2_1, temporal_mod2_weight_1 = self.EOG_contrasting.temporal_decision_summary(
                    EOG_feat,
                    EOG_aug_feat,
                    t_mod2,
                    self.temporal_binary,
                    replace_binary_with_bmm=self.branch_uses_bmm("temporal"),
                    prior_features=mod2_prior,
                    prior_bmm=mod2_segment_prior_bmm,
                    prior_is_segment=True,
                    agreement_prior_features=mod1_prior,
                    agreement_prior_bmm=mod1_segment_prior_bmm,
                    prior_require_agreement=use_prior_agreement,
                )
                temporal_mod2_2, temporal_mod2_weight_2 = self.EOG_contrasting.temporal_decision_summary(
                    EOG_aug_feat,
                    EOG_feat,
                    t_mod2_aug,
                    self.temporal_binary,
                    replace_binary_with_bmm=self.branch_uses_bmm("temporal"),
                    prior_features=mod2_prior,
                    prior_bmm=mod2_segment_prior_bmm,
                    prior_is_segment=True,
                    agreement_prior_features=mod1_prior,
                    agreement_prior_bmm=mod1_segment_prior_bmm,
                    prior_require_agreement=use_prior_agreement,
                )
                temporal_mod1 = temporal_mod1_1 | temporal_mod1_2
                temporal_mod1_weight = torch.maximum(temporal_mod1_weight_1, temporal_mod1_weight_2)
                temporal_mod2 = temporal_mod2_1 | temporal_mod2_2
                temporal_mod2_weight = torch.maximum(temporal_mod2_weight_1, temporal_mod2_weight_2)

                intra_mod1, intra_mod1_weight = self.sample_branch_decision(
                    c_T,
                    c_T_aug,
                    self.intra_binary,
                    prior_features=mod1_prior,
                    prior_bmm=mod1_sample_prior_bmm,
                    agreement_prior_features=mod2_prior,
                    agreement_prior_bmm=mod2_sample_prior_bmm,
                    use_prior_agreement=use_prior_agreement,
                    use_bmm=self.branch_uses_bmm("intra"),
                )
                intra_mod2, intra_mod2_weight = self.sample_branch_decision(
                    c_F,
                    c_F_aug,
                    self.intra_binary,
                    prior_features=mod2_prior,
                    prior_bmm=mod2_sample_prior_bmm,
                    agreement_prior_features=mod1_prior,
                    agreement_prior_bmm=mod1_sample_prior_bmm,
                    use_prior_agreement=use_prior_agreement,
                    use_bmm=self.branch_uses_bmm("intra"),
                )
                inter_mod1, inter_mod1_weight = self.sample_branch_decision(
                    c_F_shared,
                    c_T_shared,
                    self.inter_binary,
                    prior_features=mod1_prior,
                    prior_bmm=mod1_sample_prior_bmm,
                    agreement_prior_features=mod2_prior,
                    agreement_prior_bmm=mod2_sample_prior_bmm,
                    use_prior_agreement=use_prior_agreement,
                    use_bmm=self.branch_uses_bmm("inter"),
                )
                inter_mod2, inter_mod2_weight = self.sample_branch_decision(
                    c_T_shared,
                    c_F_shared,
                    self.inter_binary,
                    prior_features=mod2_prior,
                    prior_bmm=mod2_sample_prior_bmm,
                    agreement_prior_features=mod1_prior,
                    agreement_prior_bmm=mod1_sample_prior_bmm,
                    use_prior_agreement=use_prior_agreement,
                    use_bmm=self.branch_uses_bmm("inter"),
                )
                mod1_decision, mod1_weight = self.majority_branch_agreement(
                    [temporal_mod1, intra_mod1, inter_mod1],
                    [temporal_mod1_weight, intra_mod1_weight, inter_mod1_weight],
                )
                mod2_decision, mod2_weight = self.majority_branch_agreement(
                    [temporal_mod2, intra_mod2, inter_mod2],
                    [temporal_mod2_weight, intra_mod2_weight, inter_mod2_weight],
                )
        finally:
            self.restore_module_states(modules, states)

        return {
            "mod1_decision": mod1_decision.detach(),
            "mod1_weight": mod1_weight.detach(),
            "mod2_decision": mod2_decision.detach(),
            "mod2_weight": mod2_weight.detach(),
            "t_mod1": t_mod1.detach(),
            "t_mod1_aug": t_mod1_aug.detach(),
            "t_mod2": t_mod2.detach(),
            "t_mod2_aug": t_mod2_aug.detach(),
        }

    def build_intra_sample_temporal_filter(
        self,
        EEG_feat,
        EEG_aug_feat,
        EOG_feat,
        EOG_aug_feat,
        mod1_prior,
        mod2_prior,
        mod1_sample_prior_bmm,
        mod2_sample_prior_bmm,
        use_prior_agreement,
    ):
        modules = [
            self.EEG_contrasting,
            self.EOG_contrasting,
            self.intra_binary,
        ]
        states = self.set_modules_eval(modules)
        try:
            with torch.no_grad():
                _, c_T = self.EEG_contrasting.project_full(EEG_feat)
                _, c_T_aug = self.EEG_contrasting.project_full(EEG_aug_feat)
                _, c_F = self.EOG_contrasting.project_full(EOG_feat)
                _, c_F_aug = self.EOG_contrasting.project_full(EOG_aug_feat)

                mod1_decision, mod1_weight = self.sample_branch_decision(
                    c_T,
                    c_T_aug,
                    self.intra_binary,
                    prior_features=mod1_prior,
                    prior_bmm=mod1_sample_prior_bmm,
                    agreement_prior_features=mod2_prior,
                    agreement_prior_bmm=mod2_sample_prior_bmm,
                    use_prior_agreement=use_prior_agreement,
                    use_bmm=self.branch_uses_bmm("intra"),
                )
                mod1_aug_decision, mod1_aug_weight = self.sample_branch_decision(
                    c_T_aug,
                    c_T,
                    self.intra_binary,
                    prior_features=mod1_prior,
                    prior_bmm=mod1_sample_prior_bmm,
                    agreement_prior_features=mod2_prior,
                    agreement_prior_bmm=mod2_sample_prior_bmm,
                    use_prior_agreement=use_prior_agreement,
                    use_bmm=self.branch_uses_bmm("intra"),
                )
                mod2_decision, mod2_weight = self.sample_branch_decision(
                    c_F,
                    c_F_aug,
                    self.intra_binary,
                    prior_features=mod2_prior,
                    prior_bmm=mod2_sample_prior_bmm,
                    agreement_prior_features=mod1_prior,
                    agreement_prior_bmm=mod1_sample_prior_bmm,
                    use_prior_agreement=use_prior_agreement,
                    use_bmm=self.branch_uses_bmm("intra"),
                )
                mod2_aug_decision, mod2_aug_weight = self.sample_branch_decision(
                    c_F_aug,
                    c_F,
                    self.intra_binary,
                    prior_features=mod2_prior,
                    prior_bmm=mod2_sample_prior_bmm,
                    agreement_prior_features=mod1_prior,
                    agreement_prior_bmm=mod1_sample_prior_bmm,
                    use_prior_agreement=use_prior_agreement,
                    use_bmm=self.branch_uses_bmm("intra"),
                )
        finally:
            self.restore_module_states(modules, states)

        return {
            "mod1_decision": mod1_decision.detach(),
            "mod1_weight": mod1_weight.detach(),
            "mod1_aug_decision": mod1_aug_decision.detach(),
            "mod1_aug_weight": mod1_aug_weight.detach(),
            "mod2_decision": mod2_decision.detach(),
            "mod2_weight": mod2_weight.detach(),
            "mod2_aug_decision": mod2_aug_decision.detach(),
            "mod2_aug_weight": mod2_aug_weight.detach(),
        }

    def print_filter_stats(self, epoch, filter_stats):
        if not filter_stats:
            print(f"Filter stats epoch {epoch + 1}: no filtering applied")
            return

        for branch in ["temporal", "intra", "inter"]:
            stats = filter_stats.get(branch)
            if stats is None or stats["pairs"] == 0:
                continue

            pairs = stats["pairs"]
            prob_mean = stats["prob_sum"] / pairs
            prob_var = max(stats["prob_sq_sum"] / pairs - prob_mean ** 2, 0.0)
            prob_std = prob_var ** 0.5
            calls = stats["calls"]
            adaptive_attract_threshold = stats.get("adaptive_attract_threshold_sum", 0.0) / calls
            adaptive_cancel_low = stats.get("adaptive_cancel_low_sum", 0.0) / calls
            adaptive_cancel_high = stats.get("adaptive_cancel_high_sum", 0.0) / calls
            prior_prob_count = stats.get("prior_prob_count", 0)
            prior_extra = ""
            if prior_prob_count > 0:
                prior_prob_mean = stats["prior_prob_sum"] / prior_prob_count
                prior_prob_var = max(stats["prior_prob_sq_sum"] / prior_prob_count - prior_prob_mean ** 2, 0.0)
                prior_prob_std = prior_prob_var ** 0.5
                prior_candidates = stats.get("prior_candidates", 0)
                prior_extra = (
                    f" | binary>0.5={stats.get('binary_gt_05', 0) / pairs:.4f} | "
                    f"prior_candidates={prior_candidates / pairs:.4f} | "
                    f"prior_attract={stats.get('prior_attract', 0) / pairs:.4f} | "
                    f"prior_hard_neg={stats.get('prior_hard_neg', 0) / pairs:.4f} | "
                    f"prior_mean={prior_prob_mean:.4f} | "
                    f"prior_std={prior_prob_std:.4f} | "
                    f"prior>0.5={stats.get('prior_gt_05', 0) / max(prior_candidates, 1):.4f}"
                )
            label_pairs = stats.get("label_pairs", 0)
            label_extra = ""
            if label_pairs > 0:
                label_same_rate = stats.get("label_same", 0) / label_pairs
                binary_label_pairs = stats.get("binary_label_pairs", 0)
                prior_label_pairs = stats.get("prior_label_pairs", 0)
                binary_pred_pos = stats.get("binary_pred_pos", 0)
                prior_pred_pos = stats.get("prior_pred_pos", 0)
                binary_true_pos = stats.get("binary_true_pos", 0)
                prior_true_pos = stats.get("prior_true_pos", 0)
                label_parts = [f"label_same={label_same_rate:.4f}"]
                if binary_label_pairs > 0:
                    label_parts.extend([
                        f"binary_label_acc={stats.get('binary_label_correct', 0) / binary_label_pairs:.4f}",
                        f"binary_pos={binary_pred_pos / binary_label_pairs:.4f}",
                        f"binary_prec={binary_true_pos / max(binary_pred_pos, 1):.4f}",
                        f"binary_rec={binary_true_pos / max(stats.get('label_same', 0), 1):.4f}",
                    ])
                if prior_label_pairs > 0:
                    label_parts.extend([
                        f"prior_label_acc={stats.get('prior_label_correct', 0) / prior_label_pairs:.4f}",
                        f"prior_pos={prior_pred_pos / prior_label_pairs:.4f}",
                        f"prior_prec={prior_true_pos / max(prior_pred_pos, 1):.4f}",
                        f"prior_rec={prior_true_pos / max(stats.get('label_same', 0), 1):.4f}",
                    ])
                if stats.get("binary_prior_pairs", 0) > 0:
                    label_parts.append(
                        f"binary_prior_agree={stats.get('binary_prior_agree', 0) / stats.get('binary_prior_pairs', 1):.4f}"
                    )
                label_extra = " | " + " | ".join(label_parts)
            print(
                f"Filter stats epoch {epoch + 1} [{branch}]: "
                f"calls={stats['calls']} | "
                f"FN={stats['fn'] / pairs:.4f} | "
                f"attract={stats['attract'] / pairs:.4f} | "
                f"cancel={stats['cancel'] / pairs:.4f} | "
                f"cancel_count={stats['cancel']} | "
                f"hard_neg_count={stats.get('prior_hard_neg', 0)} | "
                f"binary_mean={prob_mean:.4f} | "
                f"binary_std={prob_std:.4f} | "
                f"p>0.9={stats['gt_09'] / pairs:.4f} | "
                f"0.45<p<0.55={stats['uncertain_045_055'] / pairs:.4f} | "
                f"adapt_attr_thr={adaptive_attract_threshold:.4f} | "
                f"adapt_attr_disabled={stats.get('adaptive_attract_disabled', 0) / calls:.4f} | "
                f"p>adapt_attr={stats.get('gt_adaptive_attract', 0) / pairs:.4f} | "
                f"adapt_cancel_low={adaptive_cancel_low:.4f} | "
                f"adapt_cancel_high={adaptive_cancel_high:.4f} | "
                f"adapt_cancel_window={stats.get('adaptive_cancel_window', 0) / pairs:.4f}"
                f"{prior_extra}"
                f"{label_extra}"
            )


    def forward(self, train_loader):
        lambda_1 = 0.1
        lambda_2 = 0.6
        lambda_3 = 0.4
        previous_avg_loss = None
        previous_avg_binary_loss = None
        
        pbar = trange(self.num_epochs)
        for e in pbar:
            filter_stats = {}
            epoch_losses = []
            epoch_binary_losses = []
            filter_active = self.should_filter_epoch(e)
            if not self.use_iteration:
                for batch in train_loader:
                    EEG, EOG, EEG_aug, EOG_aug, index = batch
                    if index.shape[0] < 2: 
                        continue
                    EEG = EEG.to(self.device)
                    EOG = EOG.to(self.device)
                    EEG_aug = EEG_aug.to(self.device)
                    EOG_aug = EOG_aug.to(self.device)
                    x = (EEG, EOG, EEG_aug, EOG_aug)

                    if filter_active:
                        self.update_scope_variable(e)
                        sum_1, L_C_T, L_C_F, L_C_TF, L_binary = self.loss_compute(x, index=index, filter_neg=True, filter_stats=filter_stats)
                    else:
                        sum_1, L_C_T, L_C_F, L_C_TF, L_binary = self.loss_compute(x, index=index)
                    

                    loss_C_T = torch.mean(L_C_T)
                    loss_C_F = torch.mean(L_C_F)
                    loss_C_TF = torch.mean(L_C_TF)  

                    loss = lambda_1*sum_1 + lambda_2*(loss_C_T + loss_C_F) + lambda_3*loss_C_TF

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_losses.append(float(loss))

                    epoch_binary_losses.append(float(L_binary))
                    if self.uses_any_binary_classifier():
                        self.binary_optimizer.zero_grad()
                        L_binary.backward()
                        self.binary_optimizer.step()

            self.scheduler.step(e)
            if not epoch_losses:
                continue
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_binary_loss = sum(epoch_binary_losses) / len(epoch_binary_losses)
            pbar.set_description(f"avg_loss: {str(avg_loss)}, avg_binary_loss: {str(avg_binary_loss)}")
            if filter_active:
                self.print_filter_stats(e, filter_stats)
            elif self.adaptive_warmup and previous_avg_loss is not None:
                should_start_filtering, loss_drop, binary_loss_drop = self.should_end_adaptive_warmup(
                    previous_avg_loss,
                    avg_loss,
                    previous_avg_binary_loss,
                    avg_binary_loss,
                )
                if should_start_filtering:
                    self.filter_start_epoch = e + 1
                    print(
                        f"Adaptive warmup finished at epoch {e + 1}; "
                        f"filtering starts at epoch {self.filter_start_epoch + 1} "
                        f"(loss_drop={loss_drop:.4f}, binary_loss_drop={binary_loss_drop:.4f})"
                    )
            previous_avg_loss = avg_loss
            previous_avg_binary_loss = avg_binary_loss

    def form_binary_loss_data(self, x, x_aug):
        batch_size = x.shape[0]
        x = x.detach().to(self.device)
        x_aug = x_aug.detach().to(self.device)

        x_norm = F.normalize(x.reshape(batch_size, -1), dim=1)
        x_aug_norm = F.normalize(x_aug.reshape(batch_size, -1), dim=1)
        similarity = torch.mm(x_norm, x_aug_norm.t())
        pos_neg_similarity = torch.mm(x_aug_norm, x_aug_norm.t())
        joint_similarity = torch.minimum(similarity, pos_neg_similarity)
        least_similar_idx = torch.argmin(joint_similarity, dim=1)
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

        pos_pairs = torch.cat((x, x_aug), dim=1)
        least_similar_neg_pairs = torch.cat((x, x_aug[least_similar_idx]), dim=1)
        random_neg_pairs = torch.cat((x, x_aug[random_neg_idx]), dim=1)

        pairs = torch.cat((pos_pairs, least_similar_neg_pairs, random_neg_pairs), dim=0)
        labels = torch.cat((
            torch.ones(batch_size, device=self.device),
            torch.zeros(batch_size, device=self.device),
            random_neg_labels,
        ), dim=0)

        shuffle_idx = torch.randperm(pairs.shape[0], device=self.device)
        return pairs[shuffle_idx], labels[shuffle_idx]

    def loss_compute(self, x, index=None, filter_neg=False, filter_stats=None):
        EEG, EOG, EEG_aug, EOG_aug = x[0], x[1], x[2], x[3]
        EEG_feat = self.EEG_encoder(EEG)
        EEG_aug_feat = self.EEG_encoder(EEG_aug)
        EOG_feat = self.EOG_encoder(EOG)
        EOG_aug_feat = self.EOG_encoder(EOG_aug)

        EEG_feat = F.normalize(EEG_feat, dim=1)
        EEG_aug_feat = F.normalize(EEG_aug_feat, dim=1)
        EOG_feat = F.normalize(EOG_feat, dim=1)
        EOG_aug_feat = F.normalize(EOG_aug_feat, dim=1)

        temporal_filter = filter_neg and self.filter_temporal
        intra_filter = filter_neg and self.filter_intra
        inter_filter = filter_neg and self.filter_inter
        batch_labels = self.get_batch_labels(index)
        if self.prior_mode == "separate":
            mod1_prior = self.get_prior_features("mod1", index)
            mod2_prior = self.get_prior_features("mod2", index)
            mod1_sample_prior_bmm = self.get_prior_bmm("mod1_sample")
            mod2_sample_prior_bmm = self.get_prior_bmm("mod2_sample")
            mod1_segment_prior_bmm = self.get_prior_bmm("mod1_segment")
            mod2_segment_prior_bmm = self.get_prior_bmm("mod2_segment")
            use_prior_agreement = self.prior_require_modality_agreement
        else:
            combined_prior = self.get_combined_prior_features(index)
            combined_sample_prior_bmm = self.get_prior_bmm("combined_sample")
            combined_segment_prior_bmm = self.get_prior_bmm("combined_segment")
            mod1_prior = combined_prior
            mod2_prior = combined_prior
            mod1_sample_prior_bmm = combined_sample_prior_bmm
            mod2_sample_prior_bmm = combined_sample_prior_bmm
            mod1_segment_prior_bmm = combined_segment_prior_bmm
            mod2_segment_prior_bmm = combined_segment_prior_bmm
            use_prior_agreement = False

        branch_agreement = None
        if (
            self.prior_require_within_modality_agreement
            and not self.use_intra_sample_for_temporal_filter
            and filter_neg
            and mod1_prior is not None
            and mod2_prior is not None
        ):
            branch_agreement = self.build_within_modality_agreement(
                EEG_feat,
                EEG_aug_feat,
                EOG_feat,
                EOG_aug_feat,
                mod1_prior,
                mod2_prior,
                mod1_sample_prior_bmm,
                mod2_sample_prior_bmm,
                mod1_segment_prior_bmm,
                mod2_segment_prior_bmm,
                use_prior_agreement,
            )

        mod1_branch_decision = branch_agreement["mod1_decision"] if branch_agreement is not None else None
        mod1_branch_weight = branch_agreement["mod1_weight"] if branch_agreement is not None else None
        mod2_branch_decision = branch_agreement["mod2_decision"] if branch_agreement is not None else None
        mod2_branch_weight = branch_agreement["mod2_weight"] if branch_agreement is not None else None
        t_mod1 = branch_agreement["t_mod1"] if branch_agreement is not None else None
        t_mod1_aug = branch_agreement["t_mod1_aug"] if branch_agreement is not None else None
        t_mod2 = branch_agreement["t_mod2"] if branch_agreement is not None else None
        t_mod2_aug = branch_agreement["t_mod2_aug"] if branch_agreement is not None else None

        intra_sample_temporal_filter = None
        if self.use_intra_sample_for_temporal_filter and temporal_filter:
            intra_sample_temporal_filter = self.build_intra_sample_temporal_filter(
                EEG_feat,
                EEG_aug_feat,
                EOG_feat,
                EOG_aug_feat,
                mod1_prior,
                mod2_prior,
                mod1_sample_prior_bmm,
                mod2_sample_prior_bmm,
                use_prior_agreement,
            )

        temporal_binary_classifier = None if self.use_intra_sample_for_temporal_filter else self.temporal_binary
        temporal_replace_binary_with_bmm = False if self.use_intra_sample_for_temporal_filter else self.branch_uses_bmm("temporal")
        mod1_temporal_prior_bmm = mod1_sample_prior_bmm if self.use_intra_sample_for_temporal_filter else mod1_segment_prior_bmm
        mod2_temporal_prior_bmm = mod2_sample_prior_bmm if self.use_intra_sample_for_temporal_filter else mod2_segment_prior_bmm
        temporal_prior_is_segment = not self.use_intra_sample_for_temporal_filter
        mod1_temporal_branch_decision = (
            intra_sample_temporal_filter["mod1_decision"]
            if intra_sample_temporal_filter is not None
            else mod1_branch_decision
        )
        mod1_temporal_branch_weight = (
            intra_sample_temporal_filter["mod1_weight"]
            if intra_sample_temporal_filter is not None
            else mod1_branch_weight
        )
        mod1_aug_temporal_branch_decision = (
            intra_sample_temporal_filter["mod1_aug_decision"]
            if intra_sample_temporal_filter is not None
            else mod1_branch_decision
        )
        mod1_aug_temporal_branch_weight = (
            intra_sample_temporal_filter["mod1_aug_weight"]
            if intra_sample_temporal_filter is not None
            else mod1_branch_weight
        )
        mod2_temporal_branch_decision = (
            intra_sample_temporal_filter["mod2_decision"]
            if intra_sample_temporal_filter is not None
            else mod2_branch_decision
        )
        mod2_temporal_branch_weight = (
            intra_sample_temporal_filter["mod2_weight"]
            if intra_sample_temporal_filter is not None
            else mod2_branch_weight
        )
        mod2_aug_temporal_branch_decision = (
            intra_sample_temporal_filter["mod2_aug_decision"]
            if intra_sample_temporal_filter is not None
            else mod2_branch_decision
        )
        mod2_aug_temporal_branch_weight = (
            intra_sample_temporal_filter["mod2_aug_weight"]
            if intra_sample_temporal_filter is not None
            else mod2_branch_weight
        )

        # z_T paired with EEG_aug_feat, z_F paired with EOG_aug_feat
        L_T, z_T, c_T, temporal_pairs_1, temporal_labels_1 = self.EEG_contrasting(EEG_feat, EEG_aug_feat, filter_neg = temporal_filter, binary_classifier = temporal_binary_classifier, scope_variable = self.scope_variable, filter_stats=filter_stats, stats_key="temporal", use_fn_mask=self.use_fn_mask, return_binary_data=True, adaptive_filter_thresholds=self.adaptive_filter_thresholds, prior_features=mod1_prior, prior_bmm=mod1_temporal_prior_bmm, prior_hard_neg_weight=self.prior_hard_neg_weight, prior_cancel_weighting=self.prior_cancel_weighting, prior_is_segment=temporal_prior_is_segment, labels=batch_labels, agreement_prior_features=mod2_prior, agreement_prior_bmm=mod2_temporal_prior_bmm, prior_require_agreement=use_prior_agreement, branch_agreement_decision=mod1_temporal_branch_decision, branch_agreement_weight=mod1_temporal_branch_weight, t_samples_override=t_mod1, replace_binary_with_bmm=temporal_replace_binary_with_bmm)
        L_T_aug, _, c_T_aug, temporal_pairs_1_aug, temporal_labels_1_aug = self.EEG_contrasting(EEG_aug_feat, EEG_feat, filter_neg = temporal_filter, binary_classifier = temporal_binary_classifier, scope_variable = self.scope_variable, filter_stats=filter_stats, stats_key="temporal", use_fn_mask=self.use_fn_mask, return_binary_data=True, adaptive_filter_thresholds=self.adaptive_filter_thresholds, prior_features=mod1_prior, prior_bmm=mod1_temporal_prior_bmm, prior_hard_neg_weight=self.prior_hard_neg_weight, prior_cancel_weighting=self.prior_cancel_weighting, prior_is_segment=temporal_prior_is_segment, labels=batch_labels, agreement_prior_features=mod2_prior, agreement_prior_bmm=mod2_temporal_prior_bmm, prior_require_agreement=use_prior_agreement, branch_agreement_decision=mod1_aug_temporal_branch_decision, branch_agreement_weight=mod1_aug_temporal_branch_weight, t_samples_override=t_mod1_aug, replace_binary_with_bmm=temporal_replace_binary_with_bmm)

        L_F, z_F, c_F, temporal_pairs_2, temporal_labels_2 = self.EOG_contrasting(EOG_feat, EOG_aug_feat, filter_neg = temporal_filter, binary_classifier = temporal_binary_classifier, scope_variable = self.scope_variable, filter_stats=filter_stats, stats_key="temporal", use_fn_mask=self.use_fn_mask, return_binary_data=True, adaptive_filter_thresholds=self.adaptive_filter_thresholds, prior_features=mod2_prior, prior_bmm=mod2_temporal_prior_bmm, prior_hard_neg_weight=self.prior_hard_neg_weight, prior_cancel_weighting=self.prior_cancel_weighting, prior_is_segment=temporal_prior_is_segment, labels=batch_labels, agreement_prior_features=mod1_prior, agreement_prior_bmm=mod1_temporal_prior_bmm, prior_require_agreement=use_prior_agreement, branch_agreement_decision=mod2_temporal_branch_decision, branch_agreement_weight=mod2_temporal_branch_weight, t_samples_override=t_mod2, replace_binary_with_bmm=temporal_replace_binary_with_bmm)
        L_F_aug, _, c_F_aug, temporal_pairs_2_aug, temporal_labels_2_aug = self.EOG_contrasting(EOG_aug_feat, EOG_feat, filter_neg = temporal_filter, binary_classifier = temporal_binary_classifier, scope_variable = self.scope_variable, filter_stats=filter_stats, stats_key="temporal", use_fn_mask=self.use_fn_mask, return_binary_data=True, adaptive_filter_thresholds=self.adaptive_filter_thresholds, prior_features=mod2_prior, prior_bmm=mod2_temporal_prior_bmm, prior_hard_neg_weight=self.prior_hard_neg_weight, prior_cancel_weighting=self.prior_cancel_weighting, prior_is_segment=temporal_prior_is_segment, labels=batch_labels, agreement_prior_features=mod1_prior, agreement_prior_bmm=mod1_temporal_prior_bmm, prior_require_agreement=use_prior_agreement, branch_agreement_decision=mod2_aug_temporal_branch_decision, branch_agreement_weight=mod2_aug_temporal_branch_weight, t_samples_override=t_mod2_aug, replace_binary_with_bmm=temporal_replace_binary_with_bmm)

        ############################ Intra-modal Contrastive ##############################

        L_C_T, pos_sim_T = loss_ntxent([c_T, c_T_aug], self.device, filter_neg = intra_filter, binary_classifier = self.intra_binary, scope_variable = self.scope_variable, filter_stats=filter_stats, stats_key="intra", use_fn_mask=self.use_fn_mask, adaptive_filter_thresholds=self.adaptive_filter_thresholds, prior_features=mod1_prior, prior_bmm=mod1_sample_prior_bmm, prior_hard_neg_weight=self.prior_hard_neg_weight, prior_cancel_weighting=self.prior_cancel_weighting, labels=batch_labels, agreement_prior_features=mod2_prior, agreement_prior_bmm=mod2_sample_prior_bmm, prior_require_agreement=use_prior_agreement, branch_agreement_decision=mod1_branch_decision, branch_agreement_weight=mod1_branch_weight, replace_binary_with_bmm=self.branch_uses_bmm("intra"))
        L_C_F, pos_sim_F = loss_ntxent([c_F, c_F_aug], self.device, filter_neg = intra_filter, binary_classifier = self.intra_binary, scope_variable = self.scope_variable, filter_stats=filter_stats, stats_key="intra", use_fn_mask=self.use_fn_mask, adaptive_filter_thresholds=self.adaptive_filter_thresholds, prior_features=mod2_prior, prior_bmm=mod2_sample_prior_bmm, prior_hard_neg_weight=self.prior_hard_neg_weight, prior_cancel_weighting=self.prior_cancel_weighting, labels=batch_labels, agreement_prior_features=mod1_prior, agreement_prior_bmm=mod1_sample_prior_bmm, prior_require_agreement=use_prior_agreement, branch_agreement_decision=mod2_branch_decision, branch_agreement_weight=mod2_branch_weight, replace_binary_with_bmm=self.branch_uses_bmm("intra"))

        
        ############################ Inter-modal Contrastive ##############################
        pos_sim_TF = None

        c_T_shared = self.shared_projection_head_1(z_T)
        c_F_shared = self.shared_projection_head_2(z_F)

        
        L_C_TF_1, pos_sim_TF_1 = loss_ntxent([c_T_shared, c_F_shared], self.device, filter_neg = inter_filter, binary_classifier = self.inter_binary, scope_variable = self.scope_variable, filter_stats=filter_stats, stats_key="inter", use_fn_mask=self.use_fn_mask, adaptive_filter_thresholds=self.adaptive_filter_thresholds, prior_features=mod2_prior, prior_bmm=mod2_sample_prior_bmm, prior_hard_neg_weight=self.prior_hard_neg_weight, prior_cancel_weighting=self.prior_cancel_weighting, labels=batch_labels, agreement_prior_features=mod1_prior, agreement_prior_bmm=mod1_sample_prior_bmm, prior_require_agreement=use_prior_agreement, branch_agreement_decision=mod2_branch_decision, branch_agreement_weight=mod2_branch_weight, replace_binary_with_bmm=self.branch_uses_bmm("inter"))
        L_C_TF_2, pos_sim_TF_2 = loss_ntxent([c_F_shared, c_T_shared], self.device, filter_neg = inter_filter, binary_classifier = self.inter_binary, scope_variable = self.scope_variable, filter_stats=filter_stats, stats_key="inter", use_fn_mask=self.use_fn_mask, adaptive_filter_thresholds=self.adaptive_filter_thresholds, prior_features=mod1_prior, prior_bmm=mod1_sample_prior_bmm, prior_hard_neg_weight=self.prior_hard_neg_weight, prior_cancel_weighting=self.prior_cancel_weighting, labels=batch_labels, agreement_prior_features=mod2_prior, agreement_prior_bmm=mod2_sample_prior_bmm, prior_require_agreement=use_prior_agreement, branch_agreement_decision=mod1_branch_decision, branch_agreement_weight=mod1_branch_weight, replace_binary_with_bmm=self.branch_uses_bmm("inter"))
        
        L_C_TF = (L_C_TF_1 + L_C_TF_2) / 2
        pos_sim_TF = torch.cat((pos_sim_TF_1, pos_sim_TF_2), dim=0)
            
        ############################ Compute Contrastive Loss End ##############################

        ############################# Binary Classification Losses for Interpretability ##############################
        if not self.uses_any_binary_classifier():
            L_binary = EEG_feat.new_zeros(())
        else:
            bce_loss = nn.BCEWithLogitsLoss()
            binary_loss_terms = []
            if self.temporal_binary_mode == "binary" and not self.use_intra_sample_for_temporal_filter:
                temporal_pairs = torch.cat((temporal_pairs_1, temporal_pairs_1_aug, temporal_pairs_2, temporal_pairs_2_aug), dim=0)
                temporal_labels = torch.cat((temporal_labels_1, temporal_labels_1_aug, temporal_labels_2, temporal_labels_2_aug), dim=0)
                temporal_preds = self.temporal_binary(temporal_pairs)
                binary_loss_terms.append(bce_loss(temporal_preds.squeeze(), temporal_labels))
            if self.intra_binary_mode == "binary":
                intra_pairs_1, intra_labels_1 = self.form_binary_loss_data(c_T, c_T_aug)
                intra_pairs_2, intra_labels_2 = self.form_binary_loss_data(c_F, c_F_aug)
                intra_pairs = torch.cat((intra_pairs_1, intra_pairs_2), dim=0)
                intra_labels = torch.cat((intra_labels_1, intra_labels_2), dim=0)
                intra_preds = self.intra_binary(intra_pairs)
                binary_loss_terms.append(bce_loss(intra_preds.squeeze(), intra_labels))
            if self.inter_binary_mode == "binary":
                inter_pairs_1, inter_labels_1 = self.form_binary_loss_data(c_T_shared, c_F_shared)
                inter_pairs_2, inter_labels_2 = self.form_binary_loss_data(c_F_shared, c_T_shared)
                inter_pairs = torch.cat((inter_pairs_1, inter_pairs_2), dim=0)
                inter_labels = torch.cat((inter_labels_1, inter_labels_2), dim=0)
                inter_preds = self.inter_binary(inter_pairs)
                binary_loss_terms.append(bce_loss(inter_preds.squeeze(), inter_labels))
            L_binary = sum(binary_loss_terms) if binary_loss_terms else EEG_feat.new_zeros(())


        ############################ Return losses ##############################
        sum_1 = L_T+L_T_aug+L_F+L_F_aug

        return sum_1, L_C_T, L_C_F, L_C_TF, L_binary

        



class Model(nn.Module):
    def __init__(self, 
        mod1_dims=2,
        mod2_dims=1,
        output_dims=127,
        device='cuda',
        num_class=5,
        final_out_channels = 128, 
        kernel_size = 25, 
        stride = 3, 
        timesteps=50, 
        hidden_dim=64, 
        batch_size=32,
        num_epochs=100,
        warm_epochs=50,
        adaptive_warmup_threshold=0.08,
        lr = 1e-4,
        use_iteration=False,
        num_iter=600,
        depth=4,
        filter_temporal=True,
        filter_intra=True,
        filter_inter=True,
        use_fn_mask=True,
        adaptive_filter_thresholds=False,
        use_prior=False,
        prior_info=None,
        prior_hard_neg_weight=1.0,
        prior_cancel_weighting=True,
        prior_require_modality_agreement=False,
        prior_require_within_modality_agreement=False,
        replace_binary_with_bmm=False,
        temporal_binary_mode=None,
        intra_binary_mode=None,
        inter_binary_mode=None,
        use_intra_sample_for_temporal_filter=False,
        pretrain_labels=None,
        ):
        super(Model, self).__init__()
  
        self.EEG_encoder = base_Model(mod1_dims, final_out_channels, kernel_size, stride, output_dims)
        self.EOG_encoder = base_Model(mod2_dims, final_out_channels, kernel_size, stride, output_dims)
        
        self.TFCC = TFCC(device,
            EEG_encoder = self.EEG_encoder,
            EOG_encoder = self.EOG_encoder, 
            final_out_channels = final_out_channels, 
            timesteps=timesteps, 
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            num_epochs=num_epochs,
            warm_epochs=warm_epochs,
            adaptive_warmup_threshold=adaptive_warmup_threshold,
            lr = lr,
            use_iteration=use_iteration,
            num_iter=num_iter,
            depth=depth,
            filter_temporal=filter_temporal,
            filter_intra=filter_intra,
            filter_inter=filter_inter,
            use_fn_mask=use_fn_mask,
            adaptive_filter_thresholds=adaptive_filter_thresholds,
            use_prior=use_prior,
            prior_info=prior_info,
            prior_hard_neg_weight=prior_hard_neg_weight,
            prior_cancel_weighting=prior_cancel_weighting,
            prior_require_modality_agreement=prior_require_modality_agreement,
            prior_require_within_modality_agreement=prior_require_within_modality_agreement,
            replace_binary_with_bmm=replace_binary_with_bmm,
            temporal_binary_mode=temporal_binary_mode,
            intra_binary_mode=intra_binary_mode,
            inter_binary_mode=inter_binary_mode,
            use_intra_sample_for_temporal_filter=use_intra_sample_for_temporal_filter,
            pretrain_labels=pretrain_labels,
        )
        
        self.classifier = MLP(hidden_dim * 2, hidden_dim, num_class)

    def forward(self, x, ssl = False):
        if ssl == False: # input will be a single sample
            EEG, EOG = x[0], x[1]
            EEG_features = self.EEG_encoder(EEG)
            EOG_features = self.EOG_encoder(EOG)


            EEG_transformer_features = self.TFCC.EEG_contrasting.seq_transformer(EEG_features.transpose(1,2)).reshape(EEG.shape[0], -1)
            EOG_transformer_features = self.TFCC.EOG_contrasting.seq_transformer(EOG_features.transpose(1,2)).reshape(EOG.shape[0], -1)
            features_flat = torch.cat((EEG_transformer_features, EOG_transformer_features), dim=1)
        
            return self.classifier(features_flat)
        return self.TFCC(x) #input will be a dataloader
