import numpy as np
import functools
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TC import TC
from models.loss import loss_ntxent, loss_ntxent_anchor_vs_modalities
from models.filter_stats import (
    build_fn_analysis_rows,
    format_branch_label,
    fn_analysis_enabled,
    init_fn_analysis_stats,
    save_fn_analysis_outputs,
)
from models.prior_bmm import (
    prepare_prior_for_torch,
    prior_positive_probability_matrix,
    # similarity_bmm_probability_matrix,  # Disabled: binary-to-BMM replacement.
)

import torch.optim as optim

from tqdm import trange
from scipy.stats import norm


class ComponentTimer:
    """Accumulate component execution time without changing training semantics."""

    def __init__(self, device, enabled=True):
        self.device = torch.device(device)
        self.enabled = bool(enabled)
        self.use_cuda_events = self.enabled and self.device.type == "cuda" and torch.cuda.is_available()
        self.records = defaultdict(list)
        self.cuda_totals = defaultdict(float)
        self.cpu_totals = defaultdict(float)
        self.pending_cuda_ranges = 0
        self.max_pending_cuda_ranges = 4096

    def start(self):
        if not self.enabled:
            return None
        if self.use_cuda_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def stop(self, name, start):
        if start is None:
            return
        if self.use_cuda_events:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.records[name].append((start, end))
            self.pending_cuda_ranges += 1
            if self.pending_cuda_ranges >= self.max_pending_cuda_ranges:
                self._flush_cuda_records()
        else:
            self.cpu_totals[name] += time.perf_counter() - start

    def _flush_cuda_records(self):
        if not self.records:
            return
        torch.cuda.synchronize(self.device)
        for name, event_pairs in self.records.items():
            self.cuda_totals[name] += sum(
                start.elapsed_time(end) for start, end in event_pairs
            ) / 1000.0
        self.records.clear()
        self.pending_cuda_ranges = 0

    def totals(self, reset=True):
        if not self.enabled:
            return {}
        totals = dict(self.cpu_totals)
        if self.use_cuda_events:
            self._flush_cuda_records()
            for name, value in self.cuda_totals.items():
                totals[name] = totals.get(name, 0.0) + value
        if reset:
            self.records.clear()
            self.cuda_totals.clear()
            self.cpu_totals.clear()
            self.pending_cuda_ranges = 0
        return totals



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
        mod3_encoder=None,
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
        fn_filter_use_binary=True,
        fn_filter_use_prior=True,
        use_prior=False,
        prior_info=None,
        prior_hard_neg_weight=1.0,
        prior_cancel_weighting=True,
        # replace_binary_with_bmm=False,  # Disabled: final method uses binary classifiers.
        temporal_binary_mode=None,
        intra_binary_mode=None,
        inter_binary_mode=None,
        # use_intra_sample_for_temporal_filter=False,  # Disabled: temporal filtering uses temporal classifiers/priors.
        pretrain_labels=None,
        contrast_mode="pairwise",
        fn_analysis=False,
        fn_analysis_dir=None,
        fn_analysis_every=1,
        fn_analysis_threshold=0.5,
        fn_analysis_context=None,
        log_component_timing=True,
        ):
        super(TFCC, self).__init__()
        self.EEG_encoder = EEG_encoder
        self.EOG_encoder = EOG_encoder
        self.mod3_encoder = mod3_encoder
        self.encoders = [self.EEG_encoder, self.EOG_encoder]
        if self.mod3_encoder is not None:
            self.encoders.append(self.mod3_encoder)
        self.EEG_contrasting = TC(device, final_out_channels, timesteps, hidden_dim, depth)
        self.EOG_contrasting = TC(device, final_out_channels, timesteps, hidden_dim, depth)
        self.contrast_modules = [self.EEG_contrasting, self.EOG_contrasting]
        if self.mod3_encoder is not None:
            self.MOD3_contrasting = TC(device, final_out_channels, timesteps, hidden_dim, depth)
            self.contrast_modules.append(self.MOD3_contrasting)
        self.batch_size = batch_size
        self.device = device
        self.contrast_mode = contrast_mode
        if self.contrast_mode not in ["pairwise", "1vsall"]:
            raise ValueError(f"Unsupported contrast_mode: {self.contrast_mode}")
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

        def make_shared_projection_head():
            return nn.Sequential(
                nn.Linear(hidden_dim, final_out_channels // 2),
                nn.BatchNorm1d(final_out_channels // 2),
                nn.ReLU(inplace=True),
                nn.Linear(final_out_channels // 2, final_out_channels // 4),
            )

        self.shared_projection_head_1 = make_shared_projection_head()
        self.shared_projection_head_2 = make_shared_projection_head()
        self.shared_projection_heads = [self.shared_projection_head_1, self.shared_projection_head_2]
        if self.mod3_encoder is not None:
            self.shared_projection_head_3 = make_shared_projection_head()
            self.shared_projection_heads.append(self.shared_projection_head_3)

        def make_temporal_binary():
            return nn.Sequential(
                nn.Linear(final_out_channels * 2, final_out_channels),
                nn.BatchNorm1d(final_out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(final_out_channels, 1)
            )

        binary_projection_dim = (final_out_channels // 4) * 2
        def make_sample_binary():
            return nn.Sequential(
                nn.Linear(binary_projection_dim, final_out_channels // 4),
                nn.BatchNorm1d(final_out_channels // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(final_out_channels // 4, 1)
            )

        self.temporal_binary = make_temporal_binary()
        self.intra_binary = make_sample_binary()
        self.inter_binary = make_sample_binary()
        self.temporal_binaries = {"mod1": self.temporal_binary}
        self.intra_binaries = {"mod1": self.intra_binary}
        for mod_idx in range(2, len(self.encoders) + 1):
            temporal_key = f"mod{mod_idx}"
            temporal_module = make_temporal_binary()
            setattr(self, f"temporal_binary_{temporal_key}", temporal_module)
            self.temporal_binaries[temporal_key] = temporal_module

            intra_module = make_sample_binary()
            setattr(self, f"intra_binary_{temporal_key}", intra_module)
            self.intra_binaries[temporal_key] = intra_module

        self.inter_binaries = {}
        inter_specs = self._get_inter_modal_specs()
        if inter_specs:
            if len(self.encoders) <= 2:
                for spec in inter_specs:
                    self.inter_binaries[self._inter_branch_key(spec)] = self.inter_binary
            elif self.contrast_mode == "1vsall":
                first_branch = True
                for spec in inter_specs:
                    anchor_idx = spec[1]
                    for other_idx in spec[2]:
                        branch_key = self._anchor_branch_key(anchor_idx, other_idx)
                        if first_branch:
                            self.inter_binaries[branch_key] = self.inter_binary
                            first_branch = False
                        else:
                            inter_module = make_sample_binary()
                            setattr(self, f"inter_binary_{branch_key}", inter_module)
                            self.inter_binaries[branch_key] = inter_module
            else:
                first_key = self._inter_branch_key(inter_specs[0])
                self.inter_binaries[first_key] = self.inter_binary
                for spec in inter_specs[1:]:
                    branch_key = self._inter_branch_key(spec)
                    inter_module = make_sample_binary()
                    setattr(self, f"inter_binary_{branch_key}", inter_module)
                    self.inter_binaries[branch_key] = inter_module

        encoder_params = [param for encoder in self.encoders for param in encoder.parameters()]
        contrast_params = [param for module in self.contrast_modules for param in module.parameters()]
        projection_params = [param for module in self.shared_projection_heads for param in module.parameters()]
        self.optimizer = optim.AdamW([{'params':encoder_params},
                                        {'params':contrast_params},
                                        {'params':projection_params}], lr, betas=(0.5, 0.99), weight_decay=3e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=20)

        binary_modules = list(self.temporal_binaries.values()) + list(self.intra_binaries.values()) + list(self.inter_binaries.values())
        seen_modules = set()
        binary_params = []
        for module in binary_modules:
            if id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            binary_params.extend(module.parameters())
        self.binary_optimizer = optim.AdamW([{'params': binary_params}], lr, betas=(0.5, 0.99), weight_decay=3e-4)

        self.use_iteration = use_iteration
        self.num_iter = num_iter
        self.filter_temporal = filter_temporal
        self.filter_intra = filter_intra
        self.filter_inter = filter_inter
        self.fn_filter_use_binary = fn_filter_use_binary
        self.fn_filter_use_prior = fn_filter_use_prior
        self.use_prior = use_prior and prior_info is not None
        self.prior = prepare_prior_for_torch(prior_info) if self.use_prior else None
        self.prior_mode = self.prior["metadata"].get("prior_mode", "combined") if self.prior is not None else "combined"
        self.prior_hard_neg_weight = prior_hard_neg_weight
        self.prior_cancel_weighting = prior_cancel_weighting
        # self.replace_binary_with_bmm = replace_binary_with_bmm
        self.temporal_binary_mode = self.normalize_binary_mode(temporal_binary_mode)
        self.intra_binary_mode = self.normalize_binary_mode(intra_binary_mode)
        self.inter_binary_mode = self.normalize_binary_mode(inter_binary_mode)
        # self.use_intra_sample_for_temporal_filter = use_intra_sample_for_temporal_filter
        self.pretrain_labels = torch.tensor(pretrain_labels, dtype=torch.long) if pretrain_labels is not None else None
        self.fn_analysis = fn_analysis
        self.fn_analysis_dir = fn_analysis_dir
        self.fn_analysis_every = max(1, int(fn_analysis_every))
        self.fn_analysis_threshold = float(fn_analysis_threshold)
        self.fn_analysis_context = fn_analysis_context or {}
        self.fn_analysis_attraction_rows = []
        self.fn_analysis_overlap_rows = []
        self.log_component_timing = bool(log_component_timing)
        self.component_timer = ComponentTimer(self.device, enabled=self.log_component_timing)

        self.scope_variable = 0 if self.adaptive_warmup else self.warm_epochs / self.num_epochs

    def _get_inter_modal_specs(self):
        if len(self.encoders) <= 2:
            return [('pair', 0, 1), ('pair', 1, 0)]
        if self.contrast_mode == "1vsall":
            return [
                ('anchor', anchor_idx, tuple(idx for idx in range(len(self.encoders)) if idx != anchor_idx))
                for anchor_idx in range(len(self.encoders))
            ]
        return [
            ('pair', 0, 1), ('pair', 1, 0),
            ('pair', 0, 2), ('pair', 2, 0),
            ('pair', 1, 2), ('pair', 2, 1),
        ]

    def _inter_branch_key(self, spec):
        if spec[0] == "pair":
            return f"pair_mod{spec[1] + 1}_mod{spec[2] + 1}"
        other_suffix = "_".join(f"mod{idx + 1}" for idx in spec[2])
        return f"anchor_mod{spec[1] + 1}_{other_suffix}"

    def _anchor_branch_key(self, anchor_idx, other_idx):
        return f"anchor_mod{anchor_idx + 1}_mod{other_idx + 1}"

    def _stats_key(self, branch_key):
        return f"inter:{branch_key}"

    def normalize_binary_mode(self, mode):
        if mode is None:
            # return "bmm" if replace_binary_with_bmm else "binary"
            return "binary"
        mode = mode.lower()
        # if mode not in ("binary", "bmm"):
        if mode != "binary":
            raise ValueError(f"Unsupported binary mode: {mode}")
        return mode

    # Disabled: branches can no longer replace their binary classifier with a BMM.
    # def branch_uses_bmm(self, branch):
    #     if branch == "temporal":
    #         return self.temporal_binary_mode == "bmm"
    #     if branch == "intra":
    #         return self.intra_binary_mode == "bmm"
    #     if branch == "inter":
    #         return self.inter_binary_mode == "bmm"
    #     raise ValueError(f"Unsupported branch: {branch}")

    def should_filter_epoch(self, epoch):
        if self.adaptive_warmup:
            return self.filter_start_epoch is not None and epoch >= self.filter_start_epoch
        return epoch >= self.warm_epochs

    def should_analyze_epoch(self, epoch):
        if not self.fn_analysis:
            return False
        filter_start_epoch = self.filter_start_epoch if self.filter_start_epoch is not None else self.warm_epochs
        if epoch < filter_start_epoch:
            return False
        return (epoch - filter_start_epoch) % self.fn_analysis_every == 0

    def make_filter_stats(self, epoch, filter_active):
        filter_stats = {}
        if filter_active and self.should_analyze_epoch(epoch):
            init_fn_analysis_stats(filter_stats, epoch, threshold=self.fn_analysis_threshold)
        return filter_stats

    def append_and_save_fn_analysis(self, epoch, filter_stats):
        if not fn_analysis_enabled(filter_stats):
            return
        attraction_rows, overlap_rows = build_fn_analysis_rows(
            filter_stats,
            context=self.fn_analysis_context,
        )
        self.fn_analysis_attraction_rows.extend(attraction_rows)
        self.fn_analysis_overlap_rows.extend(overlap_rows)
        if self.fn_analysis_dir is not None:
            save_fn_analysis_outputs(
                self.fn_analysis_dir,
                self.fn_analysis_attraction_rows,
                self.fn_analysis_overlap_rows,
                metadata=self.fn_analysis_metadata(),
            )

    def fn_analysis_metadata(self):
        return {
            **self.fn_analysis_context,
            "fn_analysis_threshold": self.fn_analysis_threshold,
            "fn_analysis_every": self.fn_analysis_every,
            "filter_temporal": self.filter_temporal,
            "filter_intra": self.filter_intra,
            "filter_inter": self.filter_inter,
            "fn_filter_use_binary": self.fn_filter_use_binary,
            "fn_filter_use_prior": self.fn_filter_use_prior,
            "use_prior": self.use_prior,
            "prior_mode": self.prior_mode,
            "temporal_binary_mode": self.temporal_binary_mode,
            "intra_binary_mode": self.intra_binary_mode,
            "inter_binary_mode": self.inter_binary_mode,
            "contrast_mode": self.contrast_mode,
        }

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
            self.temporal_binary_mode == "binary"
            or self.intra_binary_mode == "binary"
            or self.inter_binary_mode == "binary"
        )

    def get_prior_features(self, modality, index):
        if not self.use_prior or self.prior is None or index is None:
            return None
        if modality not in self.prior["features"]:
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

    # Disabled: sample decisions always use the trained binary classifier.
    # def make_sample_bmm_matrix(self, left, right):
    #     batch_size = left.shape[0]
    #     left = F.normalize(left.reshape(batch_size, -1), dim=1)
    #     right = F.normalize(right.reshape(batch_size, -1), dim=1)
    #     similarity = torch.mm(left, right.t())
    #     return similarity_bmm_probability_matrix(similarity)

    def sample_branch_decision(
        self,
        left,
        right,
        binary_classifier,
        prior_features=None,
        prior_bmm=None,
        # use_bmm=False,  # Disabled: sample decisions use the binary classifier.
    ):
        # if use_bmm:
        #     binary_output = self.make_sample_bmm_matrix(left, right)
        # else:
        binary_output = self.make_sample_binary_matrix(left, right, binary_classifier)
        branch_prob = binary_output
        branch_decision = binary_output > 0.5
        if prior_features is not None and prior_bmm is not None:
            prior_prob = prior_positive_probability_matrix(prior_features, prior_bmm)
            prior_decision = prior_prob > 0.5
            branch_decision = branch_decision & prior_decision
            branch_prob = torch.minimum(binary_output, prior_prob)
        return branch_decision, branch_prob

#    def build_intra_sample_temporal_filter(
#        self,
#        EEG_feat,
#        EEG_aug_feat,
#        EOG_feat,
#        EOG_aug_feat,
#        mod1_prior,
#        mod2_prior,
#        mod1_sample_prior_bmm,
#        mod2_sample_prior_bmm,
#    ):
#        modules = [
#            self.EEG_contrasting,
#            self.EOG_contrasting,
#            self.intra_binary,
#        ]
#        states = self.set_modules_eval(modules)
#        try:
#            with torch.no_grad():
#                _, c_T = self.EEG_contrasting.project_full(EEG_feat)
#                _, c_T_aug = self.EEG_contrasting.project_full(EEG_aug_feat)
#                _, c_F = self.EOG_contrasting.project_full(EOG_feat)
#                _, c_F_aug = self.EOG_contrasting.project_full(EOG_aug_feat)
#
#                mod1_decision, mod1_weight = self.sample_branch_decision(
#                    c_T,
#                    c_T_aug,
#                    self.intra_binary,
#                    prior_features=mod1_prior,
#                    prior_bmm=mod1_sample_prior_bmm,
#                    # use_bmm=self.branch_uses_bmm("intra"),
#                )
#                mod1_aug_decision, mod1_aug_weight = self.sample_branch_decision(
#                    c_T_aug,
#                    c_T,
#                    self.intra_binary,
#                    prior_features=mod1_prior,
#                    prior_bmm=mod1_sample_prior_bmm,
#                    # use_bmm=self.branch_uses_bmm("intra"),
#                )
#                mod2_decision, mod2_weight = self.sample_branch_decision(
#                    c_F,
#                    c_F_aug,
#                    self.intra_binary,
#                    prior_features=mod2_prior,
#                    prior_bmm=mod2_sample_prior_bmm,
#                    # use_bmm=self.branch_uses_bmm("intra"),
#                )
#                mod2_aug_decision, mod2_aug_weight = self.sample_branch_decision(
#                    c_F_aug,
#                    c_F,
#                    self.intra_binary,
#                    prior_features=mod2_prior,
#                    prior_bmm=mod2_sample_prior_bmm,
#                    # use_bmm=self.branch_uses_bmm("intra"),
#                )
#        finally:
#            self.restore_module_states(modules, states)
#
#        return {
#            "mod1_decision": mod1_decision.detach(),
#            "mod1_weight": mod1_weight.detach(),
#            "mod1_aug_decision": mod1_aug_decision.detach(),
#            "mod1_aug_weight": mod1_aug_weight.detach(),
#            "mod2_decision": mod2_decision.detach(),
#            "mod2_weight": mod2_weight.detach(),
#            "mod2_aug_decision": mod2_aug_decision.detach(),
#            "mod2_aug_weight": mod2_aug_weight.detach(),
#        }
#
    def print_filter_stats(self, epoch, filter_stats):
        if not filter_stats:
            print(f"Filter stats epoch {epoch + 1}: no filtering applied")
            return

        branch_order = ["temporal", "intra", "inter"]
        branch_order.extend(
            sorted(
                key for key in filter_stats.keys()
                if key.startswith("temporal:") or key.startswith("intra:") or key.startswith("inter:")
            )
        )
        for branch in branch_order:
            stats = filter_stats.get(branch)
            if stats is None or stats["pairs"] == 0:
                continue

            pairs = stats["pairs"]
            prob_mean = stats["prob_sum"] / pairs
            prob_var = max(stats["prob_sq_sum"] / pairs - prob_mean ** 2, 0.0)
            prob_std = prob_var ** 0.5
            calls = stats["calls"]
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
                    # f"prior_hard_neg={stats.get('prior_hard_neg', 0) / pairs:.4f} | "
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
                f"Filter stats epoch {epoch + 1} [{format_branch_label(branch)}]: "
                f"calls={stats['calls']} | "
                f"FN={stats['fn'] / pairs:.4f} | "
                f"attract={stats['attract'] / pairs:.4f} | "
                # f"cancel={stats['cancel'] / pairs:.4f} | "
                # f"cancel_count={stats['cancel']} | "
                # f"hard_neg_count={stats.get('prior_hard_neg', 0)} | "
                f"binary_mean={prob_mean:.4f} | "
                f"binary_std={prob_std:.4f} | "
                f"p>0.9={stats['gt_09'] / pairs:.4f} | "
                f"0.45<p<0.55={stats['uncertain_045_055'] / pairs:.4f}"
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
            if self.log_component_timing and self.component_timer.use_cuda_events:
                torch.cuda.synchronize(self.device)
            epoch_wall_start = time.perf_counter()
            epoch_losses = []
            epoch_binary_losses = []
            filter_active = self.should_filter_epoch(e)
            filter_stats = self.make_filter_stats(e, filter_active)
            if not self.use_iteration:
                for batch in train_loader:
                    *batch_tensors, index = batch
                    if index.shape[0] < 2: 
                        continue
                    batch_tensors = [tensor.to(self.device) for tensor in batch_tensors]
                    x = tuple(batch_tensors)

                    if filter_active:
                        self.update_scope_variable(e)
                        sum_1, L_C_intra, L_C_TF, L_binary = self.loss_compute(x, index=index, filter_neg=True, filter_stats=filter_stats)
                    else:
                        sum_1, L_C_intra, L_C_TF, L_binary = self.loss_compute(x, index=index)
                    

                    loss_C_intra = torch.mean(L_C_intra)
                    loss_C_TF = torch.mean(L_C_TF)  

                    loss = lambda_1*sum_1 + lambda_2*loss_C_intra + lambda_3*loss_C_TF

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_losses.append(float(loss))

                    epoch_binary_losses.append(float(L_binary))
                    if self.uses_any_binary_classifier():
                        binary_backward_timing_start = self.component_timer.start()
                        self.binary_optimizer.zero_grad()
                        L_binary.backward()
                        self.binary_optimizer.step()
                        self.component_timer.stop("binary_backward_step", binary_backward_timing_start)

            epoch_component_times = self.component_timer.totals(reset=True)
            epoch_wall_time = time.perf_counter() - epoch_wall_start
            if self.log_component_timing:
                binary_pair_time = epoch_component_times.get("binary_pair_build", 0.0)
                binary_forward_time = epoch_component_times.get("binary_loss_forward", 0.0)
                binary_backward_time = epoch_component_times.get("binary_backward_step", 0.0)
                binary_training_time = binary_pair_time + binary_forward_time + binary_backward_time
                filtering_time = epoch_component_times.get("fn_filtering", 0.0)
                timing_source = "CUDA-event GPU time" if self.component_timer.use_cuda_events else "wall time"
                print(
                    f"Component timing epoch {e + 1} [{timing_source}, filter_active={filter_active}]: "
                    f"binary_pair_build={binary_pair_time:.3f}s | "
                    f"binary_loss_forward={binary_forward_time:.3f}s | "
                    f"binary_backward_step={binary_backward_time:.3f}s | "
                    f"binary_training_total={binary_training_time:.3f}s | "
                    f"fn_filtering={filtering_time:.3f}s | "
                    f"epoch_wall={epoch_wall_time:.3f}s"
                )

            self.scheduler.step(e)
            if not epoch_losses:
                continue
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_binary_loss = sum(epoch_binary_losses) / len(epoch_binary_losses)
            pbar.set_description(f"avg_loss: {str(avg_loss)}, avg_binary_loss: {str(avg_binary_loss)}")
            if filter_active:
                self.print_filter_stats(e, filter_stats)
                self.append_and_save_fn_analysis(e, filter_stats)
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
        if self.fn_analysis and self.fn_analysis_dir is not None and self.fn_analysis_attraction_rows:
            print(f"Saved false-negative attraction analysis to {self.fn_analysis_dir}")

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
        num_modalities = len(self.encoders)
        modalities = x[:num_modalities]
        augmented_modalities = x[num_modalities: num_modalities * 2]
        if len(modalities) != len(augmented_modalities):
            raise ValueError(
                f"Expected {num_modalities} original and augmented modalities, got {len(x)} tensors"
            )

        features = [
            F.normalize(encoder(modality), dim=1)
            for encoder, modality in zip(self.encoders, modalities)
        ]
        augmented_features = [
            F.normalize(encoder(modality), dim=1)
            for encoder, modality in zip(self.encoders, augmented_modalities)
        ]

        temporal_filter = filter_neg and self.filter_temporal
        intra_filter = filter_neg and self.filter_intra
        inter_filter = filter_neg and self.filter_inter
        batch_labels = self.get_batch_labels(index)

        if self.prior_mode == "separate":
            priors = [self.get_prior_features(f"mod{i + 1}", index) for i in range(num_modalities)]
            sample_prior_bmms = [self.get_prior_bmm(f"mod{i + 1}_sample") for i in range(num_modalities)]
            segment_prior_bmms = [self.get_prior_bmm(f"mod{i + 1}_segment") for i in range(num_modalities)]
        else:
            combined_prior = self.get_combined_prior_features(index)
            combined_sample_prior_bmm = self.get_prior_bmm("combined_sample")
            combined_segment_prior_bmm = self.get_prior_bmm("combined_segment")
            priors = [combined_prior for _ in range(num_modalities)]
            sample_prior_bmms = [combined_sample_prior_bmm for _ in range(num_modalities)]
            segment_prior_bmms = [combined_segment_prior_bmm for _ in range(num_modalities)]

        # temporal_replace_binary_with_bmm = self.branch_uses_bmm("temporal")
        temporal_prior_is_segment = True

        temporal_losses = []
        temporal_binary_items = []
        projected = []
        projected_aug = []
        for mod_idx, (contrast_module, feature, augmented_feature) in enumerate(zip(self.contrast_modules, features, augmented_features)):
            mod_key = f"mod{mod_idx + 1}"
            temporal_binary_classifier = self.temporal_binaries[mod_key]
            temporal_prior_bmm = segment_prior_bmms[mod_idx]
            loss_forward, z, c, pairs, labels = contrast_module(
                feature,
                augmented_feature,
                filter_neg=temporal_filter,
                binary_classifier=temporal_binary_classifier,
                scope_variable=self.scope_variable,
                filter_stats=filter_stats,
                stats_key=f"temporal:{mod_key}",
                return_binary_data=True,
                fn_filter_use_binary=self.fn_filter_use_binary,
                fn_filter_use_prior=self.fn_filter_use_prior,
                prior_features=priors[mod_idx],
                prior_bmm=temporal_prior_bmm,
                prior_hard_neg_weight=self.prior_hard_neg_weight,
                prior_cancel_weighting=self.prior_cancel_weighting,
                prior_is_segment=temporal_prior_is_segment,
                labels=batch_labels,
                # replace_binary_with_bmm=temporal_replace_binary_with_bmm,
                timing_recorder=self.component_timer,
            )
            loss_backward, _, c_aug, pairs_aug, labels_aug = contrast_module(
                augmented_feature,
                feature,
                filter_neg=temporal_filter,
                binary_classifier=temporal_binary_classifier,
                scope_variable=self.scope_variable,
                filter_stats=filter_stats,
                stats_key=f"temporal:{mod_key}",
                return_binary_data=True,
                fn_filter_use_binary=self.fn_filter_use_binary,
                fn_filter_use_prior=self.fn_filter_use_prior,
                prior_features=priors[mod_idx],
                prior_bmm=temporal_prior_bmm,
                prior_hard_neg_weight=self.prior_hard_neg_weight,
                prior_cancel_weighting=self.prior_cancel_weighting,
                prior_is_segment=temporal_prior_is_segment,
                labels=batch_labels,
                # replace_binary_with_bmm=temporal_replace_binary_with_bmm,
                timing_recorder=self.component_timer,
            )
            temporal_losses.extend([loss_forward, loss_backward])
            temporal_binary_items.extend([
                (temporal_binary_classifier, pairs, labels),
                (temporal_binary_classifier, pairs_aug, labels_aug),
            ])
            projected.append((z, c))
            projected_aug.append(c_aug)

        intra_losses = []
        intra_binary_items = []
        for mod_idx, ((_, c), c_aug) in enumerate(zip(projected, projected_aug)):
            mod_key = f"mod{mod_idx + 1}"
            intra_binary_classifier = self.intra_binaries[mod_key]
            loss_intra, _ = loss_ntxent(
                [c, c_aug],
                self.device,
                filter_neg=intra_filter,
                binary_classifier=intra_binary_classifier,
                scope_variable=self.scope_variable,
                filter_stats=filter_stats,
                stats_key=f"intra:{mod_key}",
                fn_filter_use_binary=self.fn_filter_use_binary,
                fn_filter_use_prior=self.fn_filter_use_prior,
                prior_features=priors[mod_idx],
                prior_bmm=sample_prior_bmms[mod_idx],
                prior_hard_neg_weight=self.prior_hard_neg_weight,
                prior_cancel_weighting=self.prior_cancel_weighting,
                labels=batch_labels,
                # replace_binary_with_bmm=self.branch_uses_bmm("intra"),
                timing_recorder=self.component_timer,
            )
            intra_losses.append(loss_intra)
            if self.intra_binary_mode == "binary":
                binary_pair_timing_start = self.component_timer.start()
                pairs, labels = self.form_binary_loss_data(c, c_aug)
                self.component_timer.stop("binary_pair_build", binary_pair_timing_start)
                intra_binary_items.append((intra_binary_classifier, pairs, labels))

        shared_projected = [
            projection_head(z)
            for projection_head, (z, _) in zip(self.shared_projection_heads, projected)
        ]

        def inter_prior(branch_key, fallback_idx):
            if self.prior_mode == "separate":
                branch_features = self.get_prior_features(branch_key, index)
                branch_bmm = self.get_prior_bmm(f"{branch_key}_sample")
                if branch_features is not None and branch_bmm is not None:
                    return branch_features, branch_bmm
            return priors[fallback_idx], sample_prior_bmms[fallback_idx]

        inter_losses = []
        inter_binary_items = []
        for spec in self._get_inter_modal_specs():
            if spec[0] == "pair":
                left_idx, right_idx = spec[1], spec[2]
                branch_key = self._inter_branch_key(spec)
                inter_binary_classifier = self.inter_binaries[branch_key]
                branch_prior_features, branch_prior_bmm = inter_prior(branch_key, right_idx)
                loss_inter, _ = loss_ntxent(
                    [shared_projected[left_idx], shared_projected[right_idx]],
                    self.device,
                    filter_neg=inter_filter,
                    binary_classifier=inter_binary_classifier,
                    scope_variable=self.scope_variable,
                    filter_stats=filter_stats,
                    stats_key=self._stats_key(branch_key),
                    fn_filter_use_binary=self.fn_filter_use_binary,
                    fn_filter_use_prior=self.fn_filter_use_prior,
                    prior_features=branch_prior_features,
                    prior_bmm=branch_prior_bmm,
                    prior_hard_neg_weight=self.prior_hard_neg_weight,
                    prior_cancel_weighting=self.prior_cancel_weighting,
                    labels=batch_labels,
                    # replace_binary_with_bmm=self.branch_uses_bmm("inter"),
                    timing_recorder=self.component_timer,
                )
                inter_losses.append(loss_inter)
                if self.inter_binary_mode == "binary":
                    binary_pair_timing_start = self.component_timer.start()
                    pairs, labels = self.form_binary_loss_data(shared_projected[left_idx], shared_projected[right_idx])
                    self.component_timer.stop("binary_pair_build", binary_pair_timing_start)
                    inter_binary_items.append((inter_binary_classifier, pairs, labels))
            else:
                anchor_idx = spec[1]
                other_idxs = list(spec[2])
                branch_configs = []
                for other_idx in other_idxs:
                    branch_key = self._anchor_branch_key(anchor_idx, other_idx)
                    branch_prior_features, branch_prior_bmm = inter_prior(branch_key, other_idx)
                    branch_configs.append({
                        "binary_classifier": self.inter_binaries[branch_key],
                        "prior_features": branch_prior_features,
                        "prior_bmm": branch_prior_bmm,
                        "stats_key": self._stats_key(branch_key),
                    })
                    if self.inter_binary_mode == "binary":
                        binary_pair_timing_start = self.component_timer.start()
                        pairs, labels = self.form_binary_loss_data(shared_projected[anchor_idx], shared_projected[other_idx])
                        self.component_timer.stop("binary_pair_build", binary_pair_timing_start)
                        inter_binary_items.append((self.inter_binaries[branch_key], pairs, labels))
                loss_inter, _ = loss_ntxent_anchor_vs_modalities(
                    shared_projected[anchor_idx],
                    [shared_projected[other_idx] for other_idx in other_idxs],
                    self.device,
                    filter_neg=inter_filter,
                    branch_configs=branch_configs,
                    scope_variable=self.scope_variable,
                    filter_stats=filter_stats,
                    fn_filter_use_binary=self.fn_filter_use_binary,
                    fn_filter_use_prior=self.fn_filter_use_prior,
                    prior_hard_neg_weight=self.prior_hard_neg_weight,
                    prior_cancel_weighting=self.prior_cancel_weighting,
                    labels=batch_labels,
                    # replace_binary_with_bmm=self.branch_uses_bmm("inter"),
                    timing_recorder=self.component_timer,
                )
                inter_losses.append(loss_inter)

        binary_forward_timing_start = self.component_timer.start()
        if not self.uses_any_binary_classifier():
            L_binary = features[0].new_zeros(())
        else:
            bce_loss = nn.BCEWithLogitsLoss()
            binary_loss_terms = []
            # Disabled: temporal filtering no longer reuses intra-sample classifiers.
            # if self.temporal_binary_mode == "binary" and not self.use_intra_sample_for_temporal_filter:
            if self.temporal_binary_mode == "binary":
                for binary_classifier, pairs, labels in temporal_binary_items:
                    temporal_preds = binary_classifier(pairs)
                    binary_loss_terms.append(bce_loss(temporal_preds.squeeze(), labels))
            if self.intra_binary_mode == "binary":
                for binary_classifier, pairs, labels in intra_binary_items:
                    intra_preds = binary_classifier(pairs)
                    binary_loss_terms.append(bce_loss(intra_preds.squeeze(), labels))
            if self.inter_binary_mode == "binary":
                for binary_classifier, pairs, labels in inter_binary_items:
                    inter_preds = binary_classifier(pairs)
                    binary_loss_terms.append(bce_loss(inter_preds.squeeze(), labels))
            L_binary = sum(binary_loss_terms) if binary_loss_terms else features[0].new_zeros(())
        self.component_timer.stop("binary_loss_forward", binary_forward_timing_start)

        sum_1 = sum(temporal_losses)
        L_C_intra = sum(intra_losses) / len(intra_losses)
        L_C_TF = sum(inter_losses) / len(inter_losses)

        return sum_1, L_C_intra, L_C_TF, L_binary

        



class Model(nn.Module):
    def __init__(self, 
        mod1_dims=2,
        mod2_dims=1,
        mod3_dims=None,
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
        fn_filter_use_binary=True,
        fn_filter_use_prior=True,
        use_prior=False,
        prior_info=None,
        prior_hard_neg_weight=1.0,
        prior_cancel_weighting=True,
        # replace_binary_with_bmm=False,  # Disabled: final method uses binary classifiers.
        temporal_binary_mode=None,
        intra_binary_mode=None,
        inter_binary_mode=None,
        # use_intra_sample_for_temporal_filter=False,  # Disabled: temporal filtering uses temporal classifiers/priors.
        pretrain_labels=None,
        contrast_mode="pairwise",
        fn_analysis=False,
        fn_analysis_dir=None,
        fn_analysis_every=1,
        fn_analysis_threshold=0.5,
        fn_analysis_context=None,
        log_component_timing=True,
        ):
        super(Model, self).__init__()
  
        self.EEG_encoder = base_Model(mod1_dims, final_out_channels, kernel_size, stride, output_dims)
        self.EOG_encoder = base_Model(mod2_dims, final_out_channels, kernel_size, stride, output_dims)
        self.MOD3_encoder = base_Model(mod3_dims, final_out_channels, kernel_size, stride, output_dims) if mod3_dims is not None else None
        self.num_modalities = 3 if mod3_dims is not None else 2
        
        self.TFCC = TFCC(device,
            EEG_encoder = self.EEG_encoder,
            EOG_encoder = self.EOG_encoder, 
            mod3_encoder = self.MOD3_encoder,
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
            fn_filter_use_binary=fn_filter_use_binary,
            fn_filter_use_prior=fn_filter_use_prior,
            use_prior=use_prior,
            prior_info=prior_info,
            prior_hard_neg_weight=prior_hard_neg_weight,
            prior_cancel_weighting=prior_cancel_weighting,
            # replace_binary_with_bmm=replace_binary_with_bmm,
            temporal_binary_mode=temporal_binary_mode,
            intra_binary_mode=intra_binary_mode,
            inter_binary_mode=inter_binary_mode,
            # use_intra_sample_for_temporal_filter=use_intra_sample_for_temporal_filter,
            pretrain_labels=pretrain_labels,
            contrast_mode=contrast_mode,
            fn_analysis=fn_analysis,
            fn_analysis_dir=fn_analysis_dir,
            fn_analysis_every=fn_analysis_every,
            fn_analysis_threshold=fn_analysis_threshold,
            fn_analysis_context=fn_analysis_context,
            log_component_timing=log_component_timing,
        )
        
        self.classifier = MLP(hidden_dim * self.num_modalities, hidden_dim, num_class)

    def forward(self, x, ssl = False):
        if ssl == False: # input will be a single sample
            encoder_list = [self.EEG_encoder, self.EOG_encoder]
            if self.MOD3_encoder is not None:
                encoder_list.append(self.MOD3_encoder)
            modality_features = [encoder(modality) for encoder, modality in zip(encoder_list, x)]
            transformer_features = [
                contrast_module.seq_transformer(modality_feature.transpose(1,2)).reshape(modality_feature.shape[0], -1)
                for contrast_module, modality_feature in zip(self.TFCC.contrast_modules, modality_features)
            ]
            features_flat = torch.cat(transformer_features, dim=1)
        
            return self.classifier(features_flat)
        return self.TFCC(x) #input will be a dataloader
