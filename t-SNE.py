import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from base_models import FTDataSet
from common_utils.dataset_loader import get_idx, get_path_loader_new
from model import Model as tfcc_model


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
    elif loader_dataset_name in ["UCI-HAR", "UCI-HAR_total"]:
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


def load_fold_data(input_path, dataset_name, fold):
    parser = {"Fold": int(fold)}
    path, path_name = get_path_loader_new(input_path, dataset_name=dataset_name)
    _, _, test_idx = get_idx(parser, path, dataset_name=dataset_name)

    test_data = []
    test_label = []
    for subject_idx in test_idx:
        for data_path, label_path in path_name[subject_idx]:
            test_data.append(np.load(data_path, mmap_mode="r"))
            test_label.append(np.load(label_path, mmap_mode="r"))

    test_data = np.concatenate(test_data, axis=0)
    test_label = np.concatenate(test_label, axis=0)
    if test_data.shape[0] != test_label.shape[0]:
        raise ValueError("Test data and label sizes do not match")
    print("Number of test samples:", test_data.shape)
    return test_data, test_label


def maybe_subsample(data, labels, max_samples, seed):
    if max_samples is None or max_samples <= 0 or data.shape[0] <= max_samples:
        return data, labels
    rng = np.random.RandomState(seed)
    selected = rng.choice(data.shape[0], size=max_samples, replace=False)
    selected.sort()
    print(f"Subsampled {max_samples} / {data.shape[0]} samples for visualization")
    return data[selected], labels[selected]


def sanitize_features(features):
    if np.any(np.isnan(features)):
        print("Features contain NaNs; replacing with finite values for visualization")
    if np.any(np.isinf(features)):
        print("Features contain Infs; replacing with finite values for visualization")
    return np.nan_to_num(features)


def pca_preprocess(features, seed=42, apply_pca=True, reducer_name="visualization"):
    features = sanitize_features(np.asarray(features, dtype=np.float32))
    n_samples, n_features = features.shape

    if apply_pca:
        pca_components = min(50, n_features, n_samples - 1)
        if pca_components >= 2 and n_features > pca_components:
            features = PCA(n_components=pca_components, random_state=seed).fit_transform(features)
            print(f"Finished PCA preprocessing: {pca_components} components")
        else:
            print("PCA skipped because the feature dimensionality is already small enough")
    else:
        print(f"PCA disabled for {reducer_name} preprocessing")

    return features


def tsne_fit(features, seed=42, perplexity=30.0, apply_pca=True):
    features = pca_preprocess(
        features,
        seed=seed,
        apply_pca=apply_pca,
        reducer_name="t-SNE",
    )
    n_samples = features.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least two samples for t-SNE")

    perplexity = min(float(perplexity), max(1.0, (n_samples - 1) / 3.0))
    embedding = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    ).fit_transform(features)
    print("Finished t-SNE fitting.")
    return embedding


def umap_fit(
    features,
    seed=42,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    apply_pca=True,
):
    features = pca_preprocess(
        features,
        seed=seed,
        apply_pca=apply_pca,
        reducer_name="UMAP",
    )
    n_samples = features.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least three samples for UMAP")

    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP requested but umap-learn is not installed. Install it with `pip install umap-learn`."
        ) from exc

    n_neighbors = min(max(2, int(n_neighbors)), n_samples - 1)
    embedding = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=float(min_dist),
        metric=metric,
    ).fit_transform(features)
    print("Finished UMAP fitting.")
    return embedding


def fit_embedding(
    features,
    reducer,
    seed=42,
    perplexity=30.0,
    apply_pca=True,
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    umap_metric="euclidean",
):
    if reducer == "tsne":
        return tsne_fit(
            features,
            seed=seed,
            perplexity=perplexity,
            apply_pca=apply_pca,
        )
    if reducer == "umap":
        return umap_fit(
            features,
            seed=seed,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            apply_pca=apply_pca,
        )
    raise ValueError(f"Unsupported reducer: {reducer}")


def plot_embedding(embedding, labels, title, image_path, reducer_label):
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f9f9f9")

    unique_labels = np.unique(labels)
    cmap = plt.cm.tab20
    colors = [cmap(i % 20) for i in range(len(unique_labels))]

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            color=colors[idx],
            s=22,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.35,
            label=f"Class {label}",
        )

    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel(f"{reducer_label} 1", fontsize=10)
    ax.set_ylabel(f"{reducer_label} 2", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#d0d0d0",
        fontsize=8,
    )
    for handle in legend.legendHandles:
        handle.set_alpha(0.95)

    fig.tight_layout()
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)
    print(f"{reducer_label} plot saved to:", image_path)


def resolve_checkpoint_file(checkpoint_path, seed, fold, checkpoint_file_name):
    if not checkpoint_path:
        return None
    if os.path.isfile(checkpoint_path):
        return checkpoint_path

    candidates = []
    if seed is not None:
        candidates.append(os.path.join(checkpoint_path, str(seed), checkpoint_file_name.format(fold=fold)))
    candidates.append(os.path.join(checkpoint_path, checkpoint_file_name.format(fold=fold)))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find checkpoint file. Tried:\n" + "\n".join(f"  {candidate}" for candidate in candidates)
    )


def build_model(config, device, output_dims=127, final_out_channels=128, hidden_dim=64):
    return tfcc_model(
        mod1_dims=config["mod1_dims"],
        mod2_dims=config["mod2_dims"],
        mod3_dims=config["mod3_dims"],
        output_dims=output_dims,
        device=device,
        num_class=config["n_classes"],
        final_out_channels=final_out_channels,
        kernel_size=25,
        stride=3,
        timesteps=config["time_steps"],
        hidden_dim=hidden_dim,
    ).to(device)


def load_model(config, checkpoint_file, device, strict=False):
    model = build_model(config, device)
    state = torch.load(checkpoint_file, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"Missing keys while loading {checkpoint_file}: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys while loading {checkpoint_file}: {len(unexpected)}")
    model.eval()
    return model


def raw_features(data):
    features = data.reshape(data.shape[0], -1)
    print("Raw feature shape:", features.shape)
    return features


def model_features(model, data, labels, dataset_name, device, batch_size, feature_level):
    dataset = FTDataSet(data, labels, dataset_name=dataset_name, multi_label=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    features = []
    output_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting {feature_level} features"):
            *modalities, y = tuple(t.to(device) for t in batch)
            encoded = [
                encoder(modality)
                for encoder, modality in zip(model.TFCC.encoders, modalities)
            ]

            if feature_level == "encoder":
                batch_features = torch.cat(
                    [feature.reshape(feature.shape[0], -1) for feature in encoded],
                    dim=1,
                )
            elif feature_level == "projector":
                transformer_features = [
                    contrast_module.seq_transformer(feature.transpose(1, 2))
                    for contrast_module, feature in zip(model.TFCC.contrast_modules, encoded)
                ]
                projected = [
                    contrast_module.projection_head(feature)
                    for contrast_module, feature in zip(model.TFCC.contrast_modules, transformer_features)
                ]
                batch_features = torch.cat(
                    [feature.reshape(feature.shape[0], -1) for feature in projected],
                    dim=1,
                )
            elif feature_level == "transformer":
                transformer_features = [
                    contrast_module.seq_transformer(feature.transpose(1, 2)).reshape(feature.shape[0], -1)
                    for contrast_module, feature in zip(model.TFCC.contrast_modules, encoded)
                ]
                batch_features = torch.cat(transformer_features, dim=1)
            else:
                raise ValueError(f"Unsupported feature_level: {feature_level}")

            features.append(batch_features.cpu().numpy())
            output_labels.append(y.cpu().numpy())

    features = np.concatenate(features, axis=0)
    output_labels = np.concatenate(output_labels, axis=0)
    print(f"{feature_level} feature shape:", features.shape)
    return features, output_labels


def add_model_run(runs, name, checkpoint_path):
    if checkpoint_path:
        runs.append((name, checkpoint_path))


def str2bool(value):
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Unsupported boolean value.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="SleepEDFx")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline_checkpoint_path", type=str, default="")
    parser.add_argument("--method_checkpoint_path", type=str, default="")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Backward-compatible alias for --method_checkpoint_path",
    )
    parser.add_argument("--baseline_name", type=str, default="baseline")
    parser.add_argument("--method_name", type=str, default="method")
    parser.add_argument("--checkpoint_file_name", type=str, default="Finetuned_Model_{fold}.pkl")
    parser.add_argument("--img_save_dir", type=str, default="tsne_outputs")
    parser.add_argument("--experiment_log", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument(
        "--reducer",
        "--embedding_method",
        dest="reducer",
        type=str,
        default="tsne",
        choices=["tsne", "umap"],
        help="2D embedding method to use",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_metric", type=str, default="euclidean")
    parser.add_argument("--tsne_seed", type=int, default=42)
    parser.add_argument(
        "--feature_level",
        type=str,
        default="transformer",
        choices=["encoder", "transformer", "projector"],
    )
    parser.add_argument(
        "--apply_pca",
        type=str2bool,
        default=True,
        help="Whether to apply PCA before running the embedding method",
    )
    parser.add_argument("--include_raw", type=str2bool, default=True)
    parser.add_argument("--strict_load", type=str2bool, default=False)
    opt = parser.parse_args()

    if opt.checkpoint_path and not opt.method_checkpoint_path:
        opt.method_checkpoint_path = opt.checkpoint_path

    config = dataset_config(opt.dataset_name)
    if opt.data_path:
        config["input_path"] = opt.data_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("dataset_name:", opt.dataset_name)
    print("input_path:", config["input_path"])
    print("fold:", opt.fold)
    print("seed:", opt.seed)
    print("feature_level:", opt.feature_level)
    print("reducer:", opt.reducer)
    print("apply_pca:", opt.apply_pca)
    print("device:", device)

    data, labels = load_fold_data(config["input_path"], opt.dataset_name, opt.fold)
    data, labels = maybe_subsample(data, labels, opt.max_samples, opt.tsne_seed)

    tag_parts = [opt.dataset_name, f"fold{opt.fold}", f"seed{opt.seed}"]
    if opt.experiment_log:
        tag_parts.append(opt.experiment_log)
    if opt.reducer != "tsne":
        tag_parts.append(opt.reducer)
    tag = "_".join(tag_parts)
    reducer_label = "t-SNE" if opt.reducer == "tsne" else "UMAP"

    if opt.include_raw:
        features = raw_features(data)
        embedding = fit_embedding(
            features,
            reducer=opt.reducer,
            seed=opt.tsne_seed,
            perplexity=opt.perplexity,
            apply_pca=opt.apply_pca,
            umap_n_neighbors=opt.umap_n_neighbors,
            umap_min_dist=opt.umap_min_dist,
            umap_metric=opt.umap_metric,
        )
        plot_embedding(
            embedding,
            labels,
            f"{opt.dataset_name} raw data {reducer_label}",
            os.path.join(opt.img_save_dir, f"{tag}_raw.png"),
            reducer_label,
        )

    runs = []
    add_model_run(runs, opt.baseline_name, opt.baseline_checkpoint_path)
    add_model_run(runs, opt.method_name, opt.method_checkpoint_path)

    for run_name, checkpoint_path in runs:
        checkpoint_file = resolve_checkpoint_file(
            checkpoint_path,
            opt.seed,
            opt.fold,
            opt.checkpoint_file_name,
        )
        print(f"Loading {run_name} checkpoint:", checkpoint_file)
        model = load_model(config, checkpoint_file, device, strict=opt.strict_load)
        features, model_labels = model_features(
            model,
            data,
            labels,
            opt.dataset_name,
            device,
            opt.batch_size,
            opt.feature_level,
        )
        embedding = fit_embedding(
            features,
            reducer=opt.reducer,
            seed=opt.tsne_seed,
            perplexity=opt.perplexity,
            apply_pca=opt.apply_pca,
            umap_n_neighbors=opt.umap_n_neighbors,
            umap_min_dist=opt.umap_min_dist,
            umap_metric=opt.umap_metric,
        )
        plot_embedding(
            embedding,
            model_labels,
            f"{opt.dataset_name} {run_name} {opt.feature_level} {reducer_label}",
            os.path.join(opt.img_save_dir, f"{tag}_{run_name}_{opt.feature_level}.png"),
            reducer_label,
        )


if __name__ == "__main__":
    main()
