import argparse
import csv
import gc
import json
import os
import random
import sys
import types

import numpy as np

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, **kwargs):
        return iterable

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = tqdm
    sys.modules["tqdm"] = tqdm_module

from common_utils.dataset_loader import get_idx, get_path_loader_new
from models.statistic_prior import (
    extract_magnitude_features,
    extract_prior_features,
    save_computed_prior,
)


DATASET_PATHS = {
    "SleepEDFx": "/mimer/NOBACKUP/groups/naiss2025-22-1224/datasets_subject-wise/SleepEDFx/SleepCassette",
    "SleepEDFx_3": "/mimer/NOBACKUP/groups/naiss2025-22-1224/SleepEDFx/SleepTelemetry_preprocessed",
    "PAMAP2": "/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_256_overlap128_normalized_9classes",
    "PAMAP2_3": "/mimer/NOBACKUP/groups/naiss2025-22-1224/PAMAP2_processed_3modality",
    "UCI-HAR": "/mimer/NOBACKUP/groups/naiss2025-22-1224/UCI-HAR",
    "UCI-HAR_total": "/mimer/NOBACKUP/groups/naiss2025-22-1224/UCI-HAR_total",
}


PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze PSD prior cosine similarity for same-label and "
            "different-label training pairs at segment and sample granularity."
        )
    )
    parser.add_argument("--dataset_name", type=str, required=True, choices=DATASET_PATHS)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--segment_len", type=int, default=4)
    parser.add_argument("--sampling_rate", type=float, default=1.0)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--max_pairs_per_type", type=int, default=100000)
    parser.add_argument(
        "--feature_type",
        type=str,
        default="psd",
        choices=["psd", "magnitude", "both"],
        help="Feature family to analyze.",
    )
    parser.add_argument(
        "--magnitude_method",
        type=str,
        default="rms",
        choices=["rms", "l2", "mean_abs"],
        help="Segment magnitude statistic used when feature_type includes magnitude.",
    )
    parser.add_argument(
        "--magnitude_transform",
        type=str,
        default="log",
        choices=["log", "none"],
        help="Optional transform applied to magnitude features before normalization.",
    )
    parser.add_argument(
        "--normalize_method",
        type=str,
        default="zscore",
        choices=["zscore", "l2", "none"],
        help=(
            "Normalization after log PSD flattening. Use 'none' for log PSD "
            "features without post-log normalization."
        ),
    )
    parser.add_argument(
        "--normalize_axis",
        type=str,
        default="channel",
        choices=["channel", "segment"],
        help=(
            "Axis used for feature normalization before metric computation. "
            "'channel' normalizes each segment vector; 'segment' normalizes "
            "each feature/channel trajectory across seq_num."
        ),
    )
    parser.add_argument("--save_pair_values", action="store_true")
    parser.add_argument("--save_features", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_training_data(dataset_name, dataset_path, fold, valid_ratio):
    parser = {"filepath": dataset_path, "Fold": fold}
    path, path_name = get_path_loader_new(dataset_path, dataset_name=dataset_name)
    train_idx, val_idx, test_idx = get_idx(
        parser,
        path,
        dataset_name=dataset_name,
        valid_ratio=valid_ratio,
    )

    train_data = []
    train_label = []
    subject_ids = []

    for subject_id in tqdm(train_idx, desc="Loading training subjects"):
        if subject_id not in path_name:
            raise ValueError(f"{subject_id} not found in path_name")

        for data_path, label_path in path_name[subject_id]:
            data = np.load(data_path, mmap_mode="r")
            label = np.load(label_path, mmap_mode="r")
            train_data.append(data)
            train_label.append(label)
            subject_ids.extend([subject_id] * data.shape[0])

            if len(train_data) % 5 == 0:
                gc.collect()

    train_data = np.concatenate(train_data, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    subject_ids = np.asarray(subject_ids)

    if train_data.shape[0] != train_label.shape[0]:
        raise ValueError("Training data and labels have different sample counts")

    split_info = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    return train_data, train_label, subject_ids, split_info


def maybe_subsample(data, labels, subject_ids, max_samples, seed):
    if max_samples is None or max_samples <= 0 or data.shape[0] <= max_samples:
        return data, labels, subject_ids

    rng = np.random.RandomState(seed)
    indices = rng.choice(data.shape[0], size=max_samples, replace=False)
    indices = np.sort(indices)
    return data[indices], labels[indices], subject_ids[indices]


def encode_labels(labels):
    labels = np.asarray(labels)
    if labels.ndim == 1:
        encoded, unique = np.unique(labels, return_inverse=True)
        return unique, encoded

    flat_labels = labels.reshape(labels.shape[0], -1)
    encoded_rows, unique = np.unique(flat_labels, axis=0, return_inverse=True)
    return unique, encoded_rows


def _all_positive_pairs(class_indices):
    left_parts = []
    right_parts = []
    for indices in class_indices:
        if indices.size < 2:
            continue
        rows, cols = np.triu_indices(indices.size, k=1)
        left_parts.append(indices[rows])
        right_parts.append(indices[cols])

    if not left_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    return np.concatenate(left_parts), np.concatenate(right_parts)


def _all_negative_pairs(class_indices):
    left_parts = []
    right_parts = []
    for i in range(len(class_indices)):
        for j in range(i + 1, len(class_indices)):
            left, right = np.meshgrid(class_indices[i], class_indices[j], indexing="ij")
            left_parts.append(left.ravel())
            right_parts.append(right.ravel())

    if not left_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    return np.concatenate(left_parts), np.concatenate(right_parts)


def _sample_positive_pairs(class_indices, max_pairs, rng):
    valid_classes = [indices for indices in class_indices if indices.size >= 2]
    if not valid_classes:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    pair_counts = np.asarray(
        [indices.size * (indices.size - 1) // 2 for indices in valid_classes],
        dtype=np.float64,
    )
    class_probs = pair_counts / pair_counts.sum()
    selected_classes = rng.choice(len(valid_classes), size=max_pairs, p=class_probs)

    left = np.empty(max_pairs, dtype=np.int64)
    right = np.empty(max_pairs, dtype=np.int64)
    for pair_idx, class_idx in enumerate(selected_classes):
        indices = valid_classes[class_idx]
        chosen = rng.choice(indices, size=2, replace=False)
        left[pair_idx] = min(chosen[0], chosen[1])
        right[pair_idx] = max(chosen[0], chosen[1])

    return left, right


def _sample_negative_pairs(class_indices, max_pairs, rng):
    valid_classes = [indices for indices in class_indices if indices.size > 0]
    if len(valid_classes) < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    class_sizes = np.asarray([indices.size for indices in valid_classes], dtype=np.float64)
    class_probs = class_sizes / class_sizes.sum()

    left = np.empty(max_pairs, dtype=np.int64)
    right = np.empty(max_pairs, dtype=np.int64)
    for pair_idx in range(max_pairs):
        first_class = rng.choice(len(valid_classes), p=class_probs)
        second_probs = class_probs.copy()
        second_probs[first_class] = 0
        second_probs /= second_probs.sum()
        second_class = rng.choice(len(valid_classes), p=second_probs)

        first = rng.choice(valid_classes[first_class])
        second = rng.choice(valid_classes[second_class])
        left[pair_idx] = min(first, second)
        right[pair_idx] = max(first, second)

    return left, right


def build_pair_indices(encoded_labels, max_pairs_per_type, seed):
    rng = np.random.RandomState(seed)
    class_indices = [
        np.where(encoded_labels == label)[0]
        for label in np.unique(encoded_labels)
    ]

    if max_pairs_per_type is None or max_pairs_per_type <= 0:
        pos_left, pos_right = _all_positive_pairs(class_indices)
        neg_left, neg_right = _all_negative_pairs(class_indices)
    else:
        pos_left, pos_right = _sample_positive_pairs(
            class_indices,
            max_pairs_per_type,
            rng,
        )
        neg_left, neg_right = _sample_negative_pairs(
            class_indices,
            max_pairs_per_type,
            rng,
        )

    return {
        "positive_same_label": (pos_left, pos_right),
        "negative_different_label": (neg_left, neg_right),
    }


def l2_normalize(features, eps=1e-8):
    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / (norm + eps)


def pairwise_cosine_values(features, pair_indices, eps=1e-8):
    left, right = pair_indices
    if left.size == 0:
        return np.empty(0, dtype=np.float64)

    features = l2_normalize(features, eps=eps)
    return np.sum(features[left] * features[right], axis=-1)


def pairwise_euclidean_values(features, pair_indices):
    left, right = pair_indices
    if left.size == 0:
        return np.empty(0, dtype=np.float64)

    diff = features[left] - features[right]
    return np.linalg.norm(diff, axis=-1)


def pairwise_metric_values(features, pair_indices):
    return {
        "cosine": pairwise_cosine_values(features, pair_indices),
        "euclidean": pairwise_euclidean_values(features, pair_indices),
    }


def summarize_values(values):
    values = np.asarray(values)
    summary = {"count": int(values.size)}
    if values.size == 0:
        for key in ["mean", "std", "min", "max"]:
            summary[key] = np.nan
        for percentile in PERCENTILES:
            summary[f"p{percentile}"] = np.nan
        return summary

    summary.update(
        {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    )
    percentile_values = np.percentile(values, PERCENTILES)
    for percentile, value in zip(PERCENTILES, percentile_values):
        summary[f"p{percentile}"] = float(value)
    return summary


def add_summary_row(
    rows,
    values,
    dataset_name,
    fold,
    feature_type,
    metric,
    granularity,
    modality,
    pair_type,
    segment_idx=None,
):
    row = {
        "dataset_name": dataset_name,
        "fold": fold,
        "feature_type": feature_type,
        "metric": metric,
        "granularity": granularity,
        "modality": modality,
        "segment_idx": "" if segment_idx is None else segment_idx,
        "pair_type": pair_type,
    }
    row.update(summarize_values(values))
    rows.append(row)


def analyze_segment_similarity(
    feature_type,
    feature_sets,
    pair_indices,
    dataset_name,
    fold,
    save_pair_values=False,
):
    rows = []
    raw_values = {}

    for modality, features in feature_sets.items():
        seq_num = features.shape[1]
        for segment_idx in tqdm(
            range(seq_num),
            desc=f"Segment analysis: {modality}",
        ):
            segment_features = features[:, segment_idx, :]
            for pair_type, indices in pair_indices.items():
                metric_values = pairwise_metric_values(segment_features, indices)
                for metric, values in metric_values.items():
                    add_summary_row(
                        rows,
                        values,
                        dataset_name,
                        fold,
                        feature_type,
                        metric,
                        "segment",
                        modality,
                        pair_type,
                        segment_idx=segment_idx,
                    )
                    if save_pair_values:
                        key = (
                            f"{feature_type}_{metric}_{modality}_segment_"
                            f"{segment_idx:04d}_{pair_type}"
                        )
                        raw_values[key] = values

    return rows, raw_values


def analyze_sample_similarity(
    feature_type,
    feature_sets,
    pair_indices,
    dataset_name,
    fold,
    save_pair_values=False,
):
    rows = []
    raw_values = {}

    for modality, features in feature_sets.items():
        sample_features = features.reshape(features.shape[0], -1)
        for pair_type, indices in pair_indices.items():
            metric_values = pairwise_metric_values(sample_features, indices)
            for metric, values in metric_values.items():
                add_summary_row(
                    rows,
                    values,
                    dataset_name,
                    fold,
                    feature_type,
                    metric,
                    "sample",
                    modality,
                    pair_type,
                )
                if save_pair_values:
                    key = f"{feature_type}_{metric}_{modality}_sample_{pair_type}"
                    raw_values[key] = values

    return rows, raw_values


def write_summary_csv(path, rows):
    fieldnames = [
        "dataset_name",
        "fold",
        "feature_type",
        "metric",
        "granularity",
        "modality",
        "segment_idx",
        "pair_type",
        "count",
        "mean",
        "std",
        "min",
        "p1",
        "p5",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_feature_sets(args, train_data):
    feature_groups = {}

    if args.feature_type in ["psd", "both"]:
        modality_features = extract_prior_features(
            train_data,
            args.dataset_name,
            segment_len=args.segment_len,
            sampling_rate=args.sampling_rate,
            normalize_method=args.normalize_method,
            normalize_axis=args.normalize_axis,
        )
        feature_groups["psd"] = {
            **{
                f"mod{mod_idx}": features
                for mod_idx, features in enumerate(modality_features, start=1)
            },
            "combined": np.concatenate(modality_features, axis=-1),
        }

    if args.feature_type in ["magnitude", "both"]:
        modality_features = extract_magnitude_features(
            train_data,
            args.dataset_name,
            segment_len=args.segment_len,
            method=args.magnitude_method,
            log_transform=args.magnitude_transform == "log",
            normalize_method=args.normalize_method,
            normalize_axis=args.normalize_axis,
        )
        feature_groups["magnitude"] = {
            **{
                f"mod{mod_idx}": features
                for mod_idx, features in enumerate(modality_features, start=1)
            },
            "combined": np.concatenate(modality_features, axis=-1),
        }

    return feature_groups


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_path = args.dataset_path or DATASET_PATHS[args.dataset_name]
    os.makedirs(args.output_dir, exist_ok=True)

    train_data, train_label, subject_ids, split_info = load_training_data(
        args.dataset_name,
        dataset_path,
        args.fold,
        args.valid_ratio,
    )
    train_data, train_label, subject_ids = maybe_subsample(
        train_data,
        train_label,
        subject_ids,
        args.max_samples,
        args.seed,
    )
    encoded_labels, unique_labels = encode_labels(train_label)

    print("Training data shape:", train_data.shape)
    print("Training label shape:", train_label.shape)
    print("Analysis sample count:", train_data.shape[0])
    print("Number of encoded labels:", len(unique_labels))

    feature_groups = build_feature_sets(args, train_data)

    pair_indices = build_pair_indices(
        encoded_labels,
        args.max_pairs_per_type,
        args.seed,
    )
    pair_counts = {
        pair_type: int(indices[0].size)
        for pair_type, indices in pair_indices.items()
    }
    print("Pair counts:", pair_counts)

    segment_rows = []
    sample_rows = []
    segment_values = {}
    sample_values = {}
    for feature_type, feature_sets in feature_groups.items():
        current_segment_rows, current_segment_values = analyze_segment_similarity(
            feature_type,
            feature_sets,
            pair_indices,
            args.dataset_name,
            args.fold,
            save_pair_values=args.save_pair_values,
        )
        current_sample_rows, current_sample_values = analyze_sample_similarity(
            feature_type,
            feature_sets,
            pair_indices,
            args.dataset_name,
            args.fold,
            save_pair_values=args.save_pair_values,
        )
        segment_rows.extend(current_segment_rows)
        sample_rows.extend(current_sample_rows)
        segment_values.update(current_segment_values)
        sample_values.update(current_sample_values)

    segment_csv = os.path.join(args.output_dir, "segment_similarity_summary.csv")
    sample_csv = os.path.join(args.output_dir, "sample_similarity_summary.csv")
    write_summary_csv(segment_csv, segment_rows)
    write_summary_csv(sample_csv, sample_rows)

    metadata = {
        "dataset_name": args.dataset_name,
        "dataset_path": dataset_path,
        "fold": args.fold,
        "segment_len": args.segment_len,
        "sampling_rate": args.sampling_rate,
        "valid_ratio": args.valid_ratio,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "max_pairs_per_type": args.max_pairs_per_type,
        "feature_type": args.feature_type,
        "magnitude_method": args.magnitude_method,
        "magnitude_transform": args.magnitude_transform,
        "normalize_method": args.normalize_method,
        "normalize_axis": args.normalize_axis,
        "pair_counts": pair_counts,
        "feature_shapes": {
            feature_type: {
                name: list(features.shape)
                for name, features in feature_sets.items()
            }
            for feature_type, feature_sets in feature_groups.items()
        },
        "split_info": split_info,
    }
    metadata_path = os.path.join(args.output_dir, "analysis_metadata.json")
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)

    prior_arrays = {
        "labels": train_label,
        "encoded_labels": encoded_labels,
        "subject_ids": subject_ids,
    }
    if args.save_features:
        for feature_type, feature_sets in feature_groups.items():
            for name, features in feature_sets.items():
                prior_arrays[f"{feature_type}_{name}"] = features
    if args.save_pair_values:
        prior_arrays.update(segment_values)
        prior_arrays.update(sample_values)

    save_computed_prior(
        os.path.join(args.output_dir, "psd_prior_analysis_arrays.npz"),
        prior_arrays,
    )

    print("Saved segment summary:", segment_csv)
    print("Saved sample summary:", sample_csv)
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
