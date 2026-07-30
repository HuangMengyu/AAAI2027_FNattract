#!/usr/bin/env python3
import argparse
import csv
import os
import random
import sys
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

np = None
torch = None
F = None
DataLoader = None
tqdm = None
SSLDataSet = None
get_idx = None
get_path_loader_new = None
tfcc_model = None
load_or_build_magnitude_prior = None
prior_positive_probability_matrix = None


def import_runtime_dependencies():
    global np, torch, F, DataLoader, tqdm
    global SSLDataSet, get_idx, get_path_loader_new, tfcc_model
    global load_or_build_magnitude_prior, prior_positive_probability_matrix

    import numpy as numpy
    import torch as torch_module
    import torch.nn.functional as torch_functional
    from torch.utils.data import DataLoader as TorchDataLoader
    from tqdm import tqdm as tqdm_function

    from base_models import SSLDataSet as RepoSSLDataSet
    from common_utils.dataset_loader import get_idx as repo_get_idx
    from common_utils.dataset_loader import get_path_loader_new as repo_get_path_loader_new
    from model import Model as repo_tfcc_model
    from models.prior_bmm import load_or_build_magnitude_prior as repo_load_or_build_magnitude_prior
    from models.prior_bmm import prior_positive_probability_matrix as repo_prior_positive_probability_matrix

    np = numpy
    torch = torch_module
    F = torch_functional
    DataLoader = TorchDataLoader
    tqdm = tqdm_function
    SSLDataSet = RepoSSLDataSet
    get_idx = repo_get_idx
    get_path_loader_new = repo_get_path_loader_new
    tfcc_model = repo_tfcc_model
    load_or_build_magnitude_prior = repo_load_or_build_magnitude_prior
    prior_positive_probability_matrix = repo_prior_positive_probability_matrix


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    if value in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Unsupported boolean value.")


def parse_warm_epochs(value):
    value = str(value).strip()
    if value.lower() == "adaptive":
        return "adaptive"
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--warm_epochs must be an integer or 'adaptive'") from exc


def dataset_config(dataset_name):
    loader_dataset_name = dataset_name.replace("_3", "")
    mod3_dims = None
    if loader_dataset_name == "SleepEDFx":
        if dataset_name == "SleepEDFx_3":
            input_path = "/mimer/NOBACKUP/groups/naiss2025-22-1224/SleepEDFx/SleepTelemetry_preprocessed"
            mod1_dims, mod2_dims, mod3_dims = 2, 1, 1
        else:
            input_path = "/mimer/NOBACKUP/groups/naiss2025-22-1224/datasets_subject-wise/SleepEDFx/SleepCassette"
            mod1_dims, mod2_dims = 2, 1
        n_classes = 5
        time_steps = 50
    elif loader_dataset_name == "PAMAP2":
        if dataset_name == "PAMAP2_3":
            input_path = "/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_processed_3modality"
            mod1_dims, mod2_dims, mod3_dims = 9, 9, 9
        else:
            input_path = "/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_256_overlap128_normalized_9classes"
            mod1_dims, mod2_dims = 9, 9
        n_classes = 9
        time_steps = 5
    elif loader_dataset_name in ("UCI-HAR", "UCI-HAR_total"):
        input_path = (
            "/mimer/NOBACKUP/groups/naiss2025-22-1224/UCI-HAR_total"
            if dataset_name == "UCI-HAR_total"
            else "/mimer/NOBACKUP/groups/naiss2025-22-1224/UCI-HAR"
        )
        mod1_dims, mod2_dims = 3, 3
        n_classes = 6
        time_steps = 3
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    return {
        "input_path": input_path,
        "mod1_dims": mod1_dims,
        "mod2_dims": mod2_dims,
        "mod3_dims": mod3_dims,
        "n_classes": n_classes,
        "time_steps": time_steps,
    }


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_pretrain_data(input_path, dataset_name, fold, valid_ratio):
    parser = {"filepath": input_path, "Fold": int(fold)}
    path, path_name = get_path_loader_new(input_path, dataset_name=dataset_name)
    train_idx, val_idx, test_idx = get_idx(parser, path, dataset_name=dataset_name, valid_ratio=valid_ratio)
    print("Training subjects:", train_idx)
    print("Validation subjects:", val_idx)
    print("Testing subjects:", test_idx)

    train_data = []
    train_label = []
    for subject_idx in tqdm(train_idx, desc="Loading pretraining subjects"):
        if subject_idx not in path_name:
            raise KeyError(f"{subject_idx} not in path_name")
        for data_path, label_path in path_name[subject_idx]:
            train_data.append(np.load(data_path, mmap_mode="r"))
            train_label.append(np.load(label_path, mmap_mode="r"))

    train_data = np.concatenate(train_data, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    if train_data.shape[0] != train_label.shape[0]:
        raise ValueError("Pretraining data and label sizes do not match")
    print("Number of pretrain samples:", train_data.shape)
    return train_data, train_label


def resolve_checkpoint_file(checkpoint_path, seed, fold, checkpoint_file_name):
    if not checkpoint_path:
        raise ValueError("--checkpoint_path is required")
    if os.path.isfile(checkpoint_path):
        return checkpoint_path

    candidates = [
        os.path.join(checkpoint_path, str(seed), checkpoint_file_name.format(fold=fold)),
        os.path.join(checkpoint_path, checkpoint_file_name.format(fold=fold)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find checkpoint file. Tried:\n" + "\n".join(f"  {candidate}" for candidate in candidates)
    )


def build_model(config, opt, device, prior_info, pretrain_label):
    return tfcc_model(
        mod1_dims=config["mod1_dims"],
        mod2_dims=config["mod2_dims"],
        mod3_dims=config["mod3_dims"],
        output_dims=opt.output_dims,
        device=device,
        num_class=config["n_classes"],
        final_out_channels=opt.final_out_channels,
        kernel_size=25,
        stride=3,
        timesteps=config["time_steps"],
        hidden_dim=opt.hidden_dim,
        batch_size=opt.batch_size,
        num_epochs=opt.epochs,
        warm_epochs=opt.warm_epochs,
        adaptive_warmup_threshold=opt.adaptive_warmup_threshold,
        lr=opt.lr,
        use_iteration=opt.use_iteration,
        num_iter=opt.num_iter,
        depth=opt.depth,
        filter_temporal=True,
        filter_intra=True,
        filter_inter=True,
        fn_filter_use_binary=True,
        fn_filter_use_prior=True,
        use_prior=True,
        prior_info=prior_info,
        prior_hard_neg_weight=opt.prior_hard_neg_weight,
        prior_cancel_weighting=opt.prior_cancel_weighting,
        temporal_binary_mode="binary",
        intra_binary_mode="binary",
        inter_binary_mode="binary",
        pretrain_labels=pretrain_label,
        contrast_mode=opt.three_mod_contrast,
        log_component_timing=False,
    ).to(device)


def load_model(config, opt, device, prior_info, pretrain_label):
    checkpoint_file = resolve_checkpoint_file(
        opt.checkpoint_path,
        opt.seed,
        opt.fold,
        opt.checkpoint_file_name,
    )
    print("Loading checkpoint:", checkpoint_file)
    model = build_model(config, opt, device, prior_info, pretrain_label)
    state = torch.load(checkpoint_file, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=opt.strict_load)
    if missing:
        print(f"Missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys while loading checkpoint: {len(unexpected)}")
    model.eval()
    return model


def sanitize_name(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def as_numpy_1d(tensor):
    return tensor.detach().cpu().reshape(-1).numpy()


def make_offdiag_indices(batch_size, device):
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
    left, right = torch.where(mask)
    return left, right


def append_branch_scores(rows, branch, batch_id, batch_indices, batch_labels, binary_prob, gmm_prob):
    batch_size = binary_prob.shape[0]
    left, right = make_offdiag_indices(batch_size, binary_prob.device)
    left_idx = batch_indices[left].detach().cpu().long().numpy()
    right_idx = batch_indices[right].detach().cpu().long().numpy()
    left_label = batch_labels[left].detach().cpu().long().numpy()
    right_label = batch_labels[right].detach().cpu().long().numpy()
    binary_values = as_numpy_1d(binary_prob[left, right])
    gmm_values = as_numpy_1d(gmm_prob[left, right])

    for row_idx in range(binary_values.shape[0]):
        rows.append({
            "branch": branch,
            "batch_id": batch_id,
            "left_index": int(left_idx[row_idx]),
            "right_index": int(right_idx[row_idx]),
            "left_label": int(left_label[row_idx]),
            "right_label": int(right_label[row_idx]),
            "true_negative": bool(left_label[row_idx] != right_label[row_idx]),
            "binary_prob": float(binary_values[row_idx]),
            "gmm_prob": float(gmm_values[row_idx]),
        })


def sample_binary_matrix(left, right, binary_classifier):
    batch_size = left.shape[0]
    binary_input = torch.cat((
        left.unsqueeze(1).expand(-1, batch_size, -1).reshape(-1, left.shape[1]),
        right.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, right.shape[1]),
    ), dim=1)
    return torch.sigmoid(binary_classifier(binary_input)).reshape(batch_size, batch_size)


def temporal_branch_scores(contrast_module, feature, augmented_feature, binary_classifier, prior_features, prior_bmm):
    h_aug1 = feature.transpose(1, 2)
    h_aug2 = augmented_feature.transpose(1, 2)
    batch_size = h_aug1.shape[0]
    seq_len = h_aug1.shape[1]
    if seq_len <= contrast_module.timestep:
        raise ValueError(
            f"Sequence length {seq_len} must be larger than timestep {contrast_module.timestep} "
            "to mirror TC.forward"
        )

    t_samples = torch.randint(seq_len - contrast_module.timestep, size=(1,), device=feature.device).long()
    encode_samples = torch.empty(
        (contrast_module.timestep, batch_size, contrast_module.num_channels),
        dtype=feature.dtype,
        device=feature.device,
    )
    for step in np.arange(1, contrast_module.timestep + 1):
        encode_samples[step - 1] = h_aug2[:, t_samples + step, :].view(batch_size, contrast_module.num_channels)

    forward_seq = h_aug1[:, :t_samples + 1, :]
    z_t = contrast_module.seq_transformer(forward_seq)
    pred = torch.empty_like(encode_samples)
    for step in np.arange(0, contrast_module.timestep):
        pred[step] = contrast_module.Wk[step](z_t)

    binary_outputs = []
    gmm_outputs = []
    prior_seq_num = prior_features.shape[1]
    for step in np.arange(0, contrast_module.timestep):
        binary_input = contrast_module.make_binary_pair_grid(pred[step], encode_samples[step])
        binary_outputs.append(torch.sigmoid(binary_classifier(binary_input)).reshape(batch_size, batch_size))

        target_pos = int(t_samples.item()) + int(step) + 1
        if seq_len > 1:
            prior_segment_idx = int(round(target_pos * (prior_seq_num - 1) / (seq_len - 1)))
        else:
            prior_segment_idx = 0
        prior_segment_idx = max(0, min(prior_segment_idx, prior_seq_num - 1))
        gmm_outputs.append(
            prior_positive_probability_matrix(prior_features, prior_bmm, segment_idx=prior_segment_idx)
        )

    return torch.stack(binary_outputs, dim=0).mean(dim=0), torch.stack(gmm_outputs, dim=0).mean(dim=0)


def get_prior_bundle(tfcc, branch_type, mod_idx=None, branch_key=None, fallback_idx=None, index=None):
    if tfcc.prior_mode == "separate":
        if branch_type == "temporal":
            return (
                tfcc.get_prior_features(f"mod{mod_idx + 1}", index),
                tfcc.get_prior_bmm(f"mod{mod_idx + 1}_segment"),
            )
        if branch_type == "intra":
            return (
                tfcc.get_prior_features(f"mod{mod_idx + 1}", index),
                tfcc.get_prior_bmm(f"mod{mod_idx + 1}_sample"),
            )
        if branch_type == "inter":
            branch_features = tfcc.get_prior_features(branch_key, index)
            branch_bmm = tfcc.get_prior_bmm(f"{branch_key}_sample")
            if branch_features is not None and branch_bmm is not None:
                return branch_features, branch_bmm
            if fallback_idx is not None:
                return (
                    tfcc.get_prior_features(f"mod{fallback_idx + 1}", index),
                    tfcc.get_prior_bmm(f"mod{fallback_idx + 1}_sample"),
                )

    combined_features = tfcc.get_combined_prior_features(index)
    if branch_type == "temporal":
        return combined_features, tfcc.get_prior_bmm("combined_segment")
    return combined_features, tfcc.get_prior_bmm("combined_sample")


def collect_pair_scores(model, dataset_name, pretrain_data, pretrain_label, opt, device):
    dataset = SSLDataSet(pretrain_data, dataset_name=dataset_name)
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    tfcc = model.TFCC
    rows = []

    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(loader, desc="Scoring SSL batches")):
            if opt.max_batches is not None and batch_id >= opt.max_batches:
                break

            *batch_tensors, index = batch
            if index.shape[0] < 2:
                continue

            index = index.to(device)
            batch_labels = torch.as_tensor(pretrain_label[index.detach().cpu().long().numpy()], device=device)
            batch_tensors = [tensor.to(device) for tensor in batch_tensors]
            num_modalities = len(tfcc.encoders)
            modalities = batch_tensors[:num_modalities]
            augmented_modalities = batch_tensors[num_modalities:num_modalities * 2]

            features = [
                F.normalize(encoder(modality), dim=1)
                for encoder, modality in zip(tfcc.encoders, modalities)
            ]
            augmented_features = [
                F.normalize(encoder(modality), dim=1)
                for encoder, modality in zip(tfcc.encoders, augmented_modalities)
            ]

            projected = []
            projected_aug = []
            for mod_idx, (contrast_module, feature, augmented_feature) in enumerate(
                zip(tfcc.contrast_modules, features, augmented_features)
            ):
                mod_key = f"mod{mod_idx + 1}"
                temporal_prior_features, temporal_prior_bmm = get_prior_bundle(
                    tfcc,
                    "temporal",
                    mod_idx=mod_idx,
                    index=index,
                )
                if temporal_prior_features is not None and temporal_prior_bmm is not None:
                    binary_prob, gmm_prob = temporal_branch_scores(
                        contrast_module,
                        feature,
                        augmented_feature,
                        tfcc.temporal_binaries[mod_key],
                        temporal_prior_features,
                        temporal_prior_bmm,
                    )
                    append_branch_scores(
                        rows,
                        f"temporal:{mod_key}",
                        batch_id,
                        index,
                        batch_labels,
                        binary_prob,
                        gmm_prob,
                    )

                z, c = contrast_module.project_full(feature)
                _, c_aug = contrast_module.project_full(augmented_feature)
                projected.append((z, c))
                projected_aug.append(c_aug)

            for mod_idx, ((_, c), c_aug) in enumerate(zip(projected, projected_aug)):
                mod_key = f"mod{mod_idx + 1}"
                prior_features, prior_bmm = get_prior_bundle(tfcc, "intra", mod_idx=mod_idx, index=index)
                if prior_features is None or prior_bmm is None:
                    continue
                append_branch_scores(
                    rows,
                    f"intra:{mod_key}",
                    batch_id,
                    index,
                    batch_labels,
                    sample_binary_matrix(c, c_aug, tfcc.intra_binaries[mod_key]),
                    prior_positive_probability_matrix(prior_features, prior_bmm),
                )

            shared_projected = [
                projection_head(z)
                for projection_head, (z, _) in zip(tfcc.shared_projection_heads, projected)
            ]
            for spec in tfcc._get_inter_modal_specs():
                if spec[0] == "pair":
                    left_idx, right_idx = spec[1], spec[2]
                    branch_key = tfcc._inter_branch_key(spec)
                    prior_features, prior_bmm = get_prior_bundle(
                        tfcc,
                        "inter",
                        branch_key=branch_key,
                        fallback_idx=right_idx,
                        index=index,
                    )
                    if prior_features is None or prior_bmm is None:
                        continue
                    append_branch_scores(
                        rows,
                        tfcc._stats_key(branch_key),
                        batch_id,
                        index,
                        batch_labels,
                        sample_binary_matrix(
                            shared_projected[left_idx],
                            shared_projected[right_idx],
                            tfcc.inter_binaries[branch_key],
                        ),
                        prior_positive_probability_matrix(prior_features, prior_bmm),
                    )
                else:
                    anchor_idx = spec[1]
                    for other_idx in spec[2]:
                        branch_key = tfcc._anchor_branch_key(anchor_idx, other_idx)
                        prior_features, prior_bmm = get_prior_bundle(
                            tfcc,
                            "inter",
                            branch_key=branch_key,
                            fallback_idx=other_idx,
                            index=index,
                        )
                        if prior_features is None or prior_bmm is None:
                            continue
                        append_branch_scores(
                            rows,
                            tfcc._stats_key(branch_key),
                            batch_id,
                            index,
                            batch_labels,
                            sample_binary_matrix(
                                shared_projected[anchor_idx],
                                shared_projected[other_idx],
                                tfcc.inter_binaries[branch_key],
                            ),
                            prior_positive_probability_matrix(prior_features, prior_bmm),
                        )

    return rows


def save_scores_csv(rows, out_dir, dataset_name, fold, seed):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{sanitize_name(dataset_name)}_fold{fold}_seed{seed}_binary_gmm_pairs.csv")
    fieldnames = [
        "branch",
        "batch_id",
        "left_index",
        "right_index",
        "left_label",
        "right_label",
        "true_negative",
        "binary_prob",
        "gmm_prob",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Saved pair scores:", csv_path)
    return csv_path


def validate_rows(rows):
    for row in rows:
        if row["left_index"] == row["right_index"]:
            raise ValueError("Diagonal/self pair found in output rows")
        if not (0.0 <= row["binary_prob"] <= 1.0):
            raise ValueError(f"binary_prob out of range: {row['binary_prob']}")
        if not (0.0 <= row["gmm_prob"] <= 1.0):
            raise ValueError(f"gmm_prob out of range: {row['gmm_prob']}")
        if row["true_negative"] != (row["left_label"] != row["right_label"]):
            raise ValueError("true_negative does not match left_label != right_label")


def plot_branch(rows, out_dir, dataset_name, fold, seed, max_plot_pairs, plot_seed, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    rows_by_branch = defaultdict(list)
    for row in rows:
        rows_by_branch[row["branch"]].append(row)

    rng = np.random.RandomState(plot_seed)
    plot_paths = []
    for branch, branch_rows in sorted(rows_by_branch.items()):
        plot_rows = branch_rows
        if max_plot_pairs is not None and max_plot_pairs > 0 and len(branch_rows) > max_plot_pairs:
            selected = rng.choice(len(branch_rows), size=max_plot_pairs, replace=False)
            plot_rows = [branch_rows[idx] for idx in selected]

        true_negative = np.array([row["true_negative"] for row in plot_rows], dtype=bool)
        binary_prob = np.array([row["binary_prob"] for row in plot_rows], dtype=np.float32)
        gmm_prob = np.array([row["gmm_prob"] for row in plot_rows], dtype=np.float32)

        fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=dpi)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.scatter(
            binary_prob[true_negative],
            gmm_prob[true_negative],
            s=10,
            c="#6b7280",
            alpha=0.45,
            edgecolors="none",
            label="true negative",
            zorder=2,
        )
        ax.scatter(
            binary_prob[~true_negative],
            gmm_prob[~true_negative],
            s=16,
            c="#dc2626",
            alpha=0.85,
            edgecolors="white",
            linewidths=0.25,
            label="same label",
            zorder=3,
        )
        ax.axvline(0.5, color="#111827", linewidth=1.0, linestyle="--")
        ax.axhline(0.5, color="#111827", linewidth=1.0, linestyle="--")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Binary classifier probability")
        ax.set_ylabel("GMM prior probability")
        ax.set_title(f"{dataset_name} fold {fold} seed {seed} {branch}")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()

        image_path = os.path.join(
            out_dir,
            f"{sanitize_name(dataset_name)}_fold{fold}_seed{seed}_{sanitize_name(branch)}_binary_gmm_scatter.png",
        )
        fig.savefig(image_path)
        plt.close(fig)
        plot_paths.append(image_path)
        print("Saved scatter plot:", image_path)
    return plot_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot binary-classifier probability vs GMM prior probability for every off-diagonal "
            "SSL batch pair in the pretraining split."
        )
    )
    parser.add_argument("--dataset_name", type=str, default="SleepEDFx", choices=["SleepEDFx", "SleepEDFx_3", "PAMAP2", "PAMAP2_3", "UCI-HAR", "UCI-HAR_total"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--current_num_fold", "--fold", dest="fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--checkpoint_file_name", type=str, default="Pretrained_Model_{fold}.pkl")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="plots_binary_gmm_pair_scatter")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--max_plot_pairs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--strict_load", type=str2bool, default=False)
    parser.add_argument("--plot_seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument("--final_out_channels", type=int, default=128)
    parser.add_argument("--output_dims", type=int, default=127)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warm_epochs", "--warm_epoch", type=parse_warm_epochs, default=50)
    parser.add_argument("--adaptive_warmup_threshold", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_iteration", type=str2bool, default=False)
    parser.add_argument("--num_iter", type=int, default=600)

    parser.add_argument("--prior_save_dir", type=str, default=None)
    parser.add_argument("--prior_mode", type=str, default="separate", choices=["combined", "separate"])
    parser.add_argument("--prior_model", type=str, default="gmm", choices=["bmm", "gmm"])
    parser.add_argument("--prior_gmm_metric", type=str, default="cosine", choices=["euclidean", "cosine", "cosine_euclidean"])
    parser.add_argument("--prior_delta_mode", type=str, default="concat", choices=["none", "delta", "concat", "both"])
    parser.add_argument("--prior_fit_max_iter", type=int, default=200)
    parser.add_argument("--prior_num_random_pairs", type=int, default=3500)
    parser.add_argument("--prior_num_self_pairs", type=int, default=500)
    parser.add_argument("--prior_segment_len", type=int, default=4)
    parser.add_argument("--prior_hard_neg_weight", default=1.0)
    parser.add_argument("--prior_cancel_weighting", type=str2bool, default=False)
    parser.add_argument("--three_mod_contrast", type=str, default="pairwise", choices=["pairwise", "1vsall"])
    parser.add_argument("--prior_plot", type=str2bool, default=False)
    return parser.parse_args()


def main():
    opt = parse_args()
    import_runtime_dependencies()
    set_seed(opt.seed)
    config = dataset_config(opt.dataset_name)
    if opt.data_path:
        config["input_path"] = opt.data_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("dataset_name:", opt.dataset_name)
    print("input_path:", config["input_path"])
    print("fold:", opt.fold)
    print("seed:", opt.seed)
    print("checkpoint_path:", opt.checkpoint_path)
    print("device:", device)

    pretrain_data, pretrain_label = load_pretrain_data(
        config["input_path"],
        opt.dataset_name,
        opt.fold,
        opt.valid_ratio,
    )

    prior_save_dir = opt.prior_save_dir or os.path.join(opt.out_dir, "prior_cache")
    prior_info = load_or_build_magnitude_prior(
        pretrain_data,
        opt.dataset_name,
        fold=opt.fold,
        seed=opt.seed,
        save_dir=prior_save_dir,
        segment_len=opt.prior_segment_len,
        num_random_pairs=opt.prior_num_random_pairs,
        num_self_pairs=opt.prior_num_self_pairs,
        prior_mode=opt.prior_mode,
        prior_model=opt.prior_model,
        prior_gmm_metric=opt.prior_gmm_metric,
        prior_delta_mode=opt.prior_delta_mode,
        prior_fit_max_iter=opt.prior_fit_max_iter,
        three_mod_contrast=opt.three_mod_contrast,
        plot=opt.prior_plot,
    )

    model = load_model(config, opt, device, prior_info, pretrain_label)
    rows = collect_pair_scores(model, opt.dataset_name, pretrain_data, pretrain_label, opt, device)
    if not rows:
        raise RuntimeError("No pair scores were collected. Check batch size, max_batches, and prior settings.")
    validate_rows(rows)
    save_scores_csv(rows, opt.out_dir, opt.dataset_name, opt.fold, opt.seed)
    plot_paths = plot_branch(
        rows,
        opt.out_dir,
        opt.dataset_name,
        opt.fold,
        opt.seed,
        opt.max_plot_pairs,
        opt.plot_seed,
        opt.dpi,
    )
    print(f"Done. Saved {len(rows)} scored pairs across {len(plot_paths)} branch plots.")


if __name__ == "__main__":
    main()
