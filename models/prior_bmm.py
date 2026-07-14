import json
import math
import os

import numpy as np
import torch

from .statistic_prior import extract_magnitude_features


def _as_float32(array):
    return np.asarray(array, dtype=np.float32)


def euclidean_pair_values(features, left_idx, right_idx):
    left = features[left_idx]
    right = features[right_idx]
    return np.linalg.norm(left - right, axis=-1)


def l2_normalize(features, eps=1e-8):
    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / (norm + eps)


def cosine_pair_values(features, left_idx, right_idx):
    normalized = l2_normalize(features)
    return np.sum(normalized[left_idx] * normalized[right_idx], axis=-1)


def sample_offdiag_pairs(num_samples, num_pairs, rng):
    left = rng.randint(0, num_samples, size=num_pairs)
    right = rng.randint(0, num_samples - 1, size=num_pairs)
    right = right + (right >= left)
    return left, right

def sample_all_pairs(num_samples, num_pairs, rng):
    left = rng.randint(0, num_samples, size=num_pairs)
    right = rng.randint(0, num_samples, size=num_pairs)
    # right = right + (right >= left)
    return left, right


def distance_to_beta_input(distance, distance_scale, eps=1e-6):
    distance_scale = np.maximum(np.asarray(distance_scale, dtype=np.float64), eps)
    values = np.asarray(distance, dtype=np.float64) / distance_scale
    return np.clip(values, eps, 1.0 - eps)


def _is_dual_prior_features(features):
    return features.ndim >= 4 and features.shape[-1] == 2


def _flatten_dual_features(features):
    return [
        features[..., variable_idx].reshape(features.shape[0], -1)
        for variable_idx in range(features.shape[-1])
    ]


def _dual_euclidean_pair_values(features, left_idx, right_idx):
    return np.stack(
        [
            euclidean_pair_values(variable_features, left_idx, right_idx)
            for variable_features in _flatten_dual_features(features)
        ],
        axis=1,
    )


def _dual_cosine_pair_values(features, left_idx, right_idx):
    return np.stack(
        [
            cosine_pair_values(variable_features, left_idx, right_idx)
            for variable_features in _flatten_dual_features(features)
        ],
        axis=1,
    )


def collect_sample_prior_values(features, num_random_pairs, num_self_pairs, rng, metric="euclidean", eps=1e-6):
    # left, right = sample_offdiag_pairs(flat_features.shape[0], num_random_pairs, rng)
    left, right = sample_all_pairs(features.shape[0], num_random_pairs, rng)
    if _is_dual_prior_features(features):
        if metric == "cosine_euclidean":
            raise ValueError("prior_delta_mode='both' supports euclidean or cosine metrics, not cosine_euclidean")
        random_euclidean = _dual_euclidean_pair_values(features, left, right)
        distance_scale = np.maximum(np.max(random_euclidean, axis=0), eps).astype(np.float64)
        if metric == "euclidean":
            random_values = random_euclidean
            self_values = np.zeros((num_self_pairs, features.shape[-1]), dtype=np.float64)
        elif metric == "cosine":
            random_values = _dual_cosine_pair_values(features, left, right)
            self_values = np.ones((num_self_pairs, features.shape[-1]), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported prior_gmm_metric: {metric}")
        values = np.concatenate((random_values, self_values), axis=0)
        return values, distance_scale

    flat_features = features.reshape(features.shape[0], -1)
    random_euclidean = euclidean_pair_values(flat_features, left, right)
    distance_scale = float(max(np.max(random_euclidean), eps))
    if metric == "euclidean":
        random_values = random_euclidean
        self_values = np.zeros(num_self_pairs, dtype=np.float64)
    elif metric == "cosine":
        random_values = cosine_pair_values(flat_features, left, right)
        self_values = np.ones(num_self_pairs, dtype=np.float64)
    elif metric == "cosine_euclidean":
        random_cosine = cosine_pair_values(flat_features, left, right)
        random_values = np.stack((random_cosine, random_euclidean), axis=1)
        self_values = np.tile(np.array([[1.0, 0.0]], dtype=np.float64), (num_self_pairs, 1))
    else:
        raise ValueError(f"Unsupported prior_gmm_metric: {metric}")
    values = np.concatenate((random_values, self_values), axis=0)
    return values, distance_scale


def collect_segment_prior_values(features, num_random_pairs, num_self_pairs, rng, metric="euclidean", eps=1e-6):
    num_samples, seq_num = features.shape[:2]
    # left, right = sample_offdiag_pairs(num_samples, num_random_pairs, rng)
    left, right = sample_all_pairs(num_samples, num_random_pairs, rng)
    segment_idx = rng.randint(0, seq_num, size=num_random_pairs)
    if _is_dual_prior_features(features):
        if metric == "cosine_euclidean":
            raise ValueError("prior_delta_mode='both' supports euclidean or cosine metrics, not cosine_euclidean")
        left_features = features[left, segment_idx]
        right_features = features[right, segment_idx]
        random_euclidean = np.stack(
            [
                np.linalg.norm(left_features[..., variable_idx] - right_features[..., variable_idx], axis=-1)
                for variable_idx in range(features.shape[-1])
            ],
            axis=1,
        )
        distance_scale = np.maximum(np.max(random_euclidean, axis=0), eps).astype(np.float64)
        if metric == "euclidean":
            random_values = random_euclidean
            self_values = np.zeros((num_self_pairs, features.shape[-1]), dtype=np.float64)
        elif metric == "cosine":
            random_values = []
            for variable_idx in range(features.shape[-1]):
                random_values.append(
                    np.sum(
                        l2_normalize(left_features[..., variable_idx])
                        * l2_normalize(right_features[..., variable_idx]),
                        axis=-1,
                    )
                )
            random_values = np.stack(random_values, axis=1)
            self_values = np.ones((num_self_pairs, features.shape[-1]), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported prior_gmm_metric: {metric}")
        values = np.concatenate((random_values, self_values), axis=0)
        return values, distance_scale

    left_features = features[left, segment_idx]
    right_features = features[right, segment_idx]
    random_euclidean = np.linalg.norm(left_features - right_features, axis=-1)
    distance_scale = float(max(np.max(random_euclidean), eps))
    if metric == "euclidean":
        random_values = random_euclidean
        self_values = np.zeros(num_self_pairs, dtype=np.float64)
    elif metric == "cosine":
        random_values = np.sum(l2_normalize(left_features) * l2_normalize(right_features), axis=-1)
        self_values = np.ones(num_self_pairs, dtype=np.float64)
    elif metric == "cosine_euclidean":
        random_cosine = np.sum(l2_normalize(left_features) * l2_normalize(right_features), axis=-1)
        random_values = np.stack((random_cosine, random_euclidean), axis=1)
        self_values = np.tile(np.array([[1.0, 0.0]], dtype=np.float64), (num_self_pairs, 1))
    else:
        raise ValueError(f"Unsupported prior_gmm_metric: {metric}")
    values = np.concatenate((random_values, self_values), axis=0)
    return values, distance_scale


def collect_sample_distances(features, num_random_pairs, num_self_pairs, rng, eps=1e-6):
    return collect_sample_prior_values(
        features,
        num_random_pairs,
        num_self_pairs,
        rng,
        metric="euclidean",
        eps=eps,
    )


def collect_segment_distances(features, num_random_pairs, num_self_pairs, rng, eps=1e-6):
    return collect_segment_prior_values(
        features,
        num_random_pairs,
        num_self_pairs,
        rng,
        metric="euclidean",
        eps=eps,
    )


def _log_beta_pdf(values, alpha, beta, eps=1e-12):
    values = np.clip(values, eps, 1.0 - eps)
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    lgamma = np.vectorize(math.lgamma)
    log_norm = lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta)
    return log_norm + (alpha - 1.0) * np.log(values) + (beta - 1.0) * np.log1p(-values)


def _weighted_beta_moments(values, weights, eps=1e-6):
    weight_sum = np.sum(weights) + eps
    if values.ndim == 1:
        weighted_values = weights * values
        mean = np.sum(weighted_values) / weight_sum
        var = np.sum(weights * (values - mean) ** 2) / weight_sum
    else:
        weighted_values = weights[:, None] * values
        mean = np.sum(weighted_values, axis=0) / weight_sum
        var = np.sum(weights[:, None] * (values - mean) ** 2, axis=0) / weight_sum
    mean = np.clip(mean, eps, 1.0 - eps)
    max_var = mean * (1.0 - mean) - eps
    var = np.clip(var, eps, max_var)
    concentration = mean * (1.0 - mean) / var - 1.0
    alpha = np.maximum(mean * concentration, eps)
    beta = np.maximum((1.0 - mean) * concentration, eps)
    return alpha, beta


def fit_beta_mixture(values, max_iter=200, eps=1e-6):
    values = np.clip(np.asarray(values, dtype=np.float64), eps, 1.0 - eps)
    scalar_input = values.ndim == 1
    fit_values = values.reshape(-1, 1) if scalar_input else values.reshape(values.shape[0], -1)
    score = fit_values.mean(axis=1)
    median = np.median(score)
    responsibilities = np.stack((score <= median, score > median), axis=1).astype(np.float64)
    responsibilities += eps
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)

    weights = np.array([0.5, 0.5], dtype=np.float64)
    feature_dim = fit_values.shape[1]
    alpha = np.tile(np.array([[2.0], [5.0]], dtype=np.float64), (1, feature_dim))
    beta = np.tile(np.array([[5.0], [2.0]], dtype=np.float64), (1, feature_dim))
    previous_ll = -np.inf
    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        weights = responsibilities.mean(axis=0)
        weights = np.clip(weights, eps, 1.0)
        weights /= weights.sum()

        for component in range(2):
            alpha[component], beta[component] = _weighted_beta_moments(
                fit_values,
                responsibilities[:, component],
                eps=eps,
            )

        log_prob = np.stack(
            [
                np.log(weights[component] + eps)
                + np.sum(_log_beta_pdf(fit_values, alpha[component], beta[component], eps=eps), axis=1)
                for component in range(2)
            ],
            axis=1,
        )
        row_max = np.max(log_prob, axis=1, keepdims=True)
        prob = np.exp(log_prob - row_max)
        prob_sum = np.sum(prob, axis=1, keepdims=True) + eps
        responsibilities = prob / prob_sum
        log_likelihood = float(np.sum(row_max + np.log(prob_sum)))
        if abs(log_likelihood - previous_ll) < 1e-5:
            converged = True
            break
        previous_ll = log_likelihood

    means = alpha / (alpha + beta)
    variances = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
    if scalar_input:
        alpha_out = alpha.reshape(2)
        beta_out = beta.reshape(2)
        means_out = means.reshape(2)
        variances_out = variances.reshape(2)
        high_component = int(np.argmax(means_out))
    else:
        alpha_out = alpha
        beta_out = beta
        means_out = means
        variances_out = variances
        high_component = int(np.argmax(means.mean(axis=1)))
    return {
        "alpha": alpha_out.astype(np.float32),
        "beta": beta_out.astype(np.float32),
        "weights": weights.astype(np.float32),
        "means": means_out.astype(np.float32),
        "variances": variances_out.astype(np.float32),
        "high_component": high_component,
        "positive_component": high_component,
        "converged": converged,
        "n_iter": n_iter,
        "model_type": "bmm",
    }


def fit_distance_beta_mixture(distances, distance_scale, max_iter=100, eps=1e-6):
    values = distance_to_beta_input(distances, distance_scale, eps=eps)
    params = fit_beta_mixture(values, max_iter=max_iter, eps=eps)
    means = np.asarray(params["means"], dtype=np.float64).reshape(2, -1)
    positive_component = int(np.argmin(means.mean(axis=1)))
    params["positive_component"] = positive_component
    params["distance_scale"] = np.array(distance_scale, dtype=np.float32)
    params["metric"] = "euclidean"
    return params


def _log_gaussian_pdf(values, means, variances, eps=1e-8):
    variances = np.maximum(variances, eps)
    return -0.5 * (np.log(2.0 * np.pi * variances) + ((values - means) ** 2) / variances)


def fit_gaussian_mixture(values, max_iter=200, eps=1e-8):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
        scalar_input = True
    else:
        values = values.reshape(values.shape[0], -1)
        scalar_input = False
    means = np.percentile(values, [25, 75], axis=0).astype(np.float64)
    if np.allclose(means[0], means[1]):
        means = np.stack((values.min(axis=0), values.max(axis=0)), axis=0).astype(np.float64)
    global_var = np.var(values, axis=0) + eps
    variances = np.stack((global_var, global_var), axis=0).astype(np.float64)
    weights = np.array([0.5, 0.5], dtype=np.float64)
    previous_ll = -np.inf
    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        log_prob = np.stack(
            [
                np.log(weights[component] + eps)
                + np.sum(_log_gaussian_pdf(values, means[component], variances[component], eps=eps), axis=1)
                for component in range(2)
            ],
            axis=1,
        )
        row_max = np.max(log_prob, axis=1, keepdims=True)
        prob = np.exp(log_prob - row_max)
        prob_sum = np.sum(prob, axis=1, keepdims=True) + eps
        responsibilities = prob / prob_sum
        log_likelihood = float(np.sum(row_max + np.log(prob_sum)))

        resp_sum = responsibilities.sum(axis=0) + eps
        weights = resp_sum / resp_sum.sum()
        means = (responsibilities[:, :, None] * values[:, None, :]).sum(axis=0) / resp_sum[:, None]
        variances = (
            responsibilities[:, :, None] * ((values[:, None, :] - means) ** 2)
        ).sum(axis=0) / resp_sum[:, None]
        variances = np.maximum(variances, eps)

        if abs(log_likelihood - previous_ll) < 1e-5:
            converged = True
            break
        previous_ll = log_likelihood

    if scalar_input:
        means_out = means.reshape(2)
        variances_out = variances.reshape(2)
    else:
        means_out = means
        variances_out = variances
    positive_component = int(np.argmin(means_out if scalar_input else means[:, -1]))
    return {
        "weights": weights.astype(np.float32),
        "means": means_out.astype(np.float32),
        "variances": variances_out.astype(np.float32),
        "positive_component": positive_component,
        "converged": converged,
        "n_iter": n_iter,
        "model_type": "gmm",
        "metric": "euclidean",
    }


def _positive_component_for_gmm(params, metric):
    means = np.asarray(params["means"], dtype=np.float64)
    means_2d = means.reshape(2, -1)
    if metric == "euclidean":
        return int(np.argmin(means_2d.mean(axis=1)))
    if metric == "cosine":
        return int(np.argmax(means_2d.mean(axis=1)))
    if metric == "cosine_euclidean":
        cosine = means_2d[:, 0]
        euclidean = means_2d[:, 1]
        cosine_range = max(float(np.ptp(cosine)), 1e-8)
        euclidean_range = max(float(np.ptp(euclidean)), 1e-8)
        score = (cosine - cosine.min()) / cosine_range - (euclidean - euclidean.min()) / euclidean_range
        return int(np.argmax(score))
    raise ValueError(f"Unsupported prior_gmm_metric: {metric}")


def fit_distance_mixture(distances, distance_scale, prior_model, max_iter=200, eps=1e-6, metric="euclidean"):
    prior_model = prior_model.lower()
    metric = metric.lower()
    if prior_model == "bmm":
        if metric == "euclidean":
            return fit_distance_beta_mixture(distances, distance_scale, max_iter=max_iter, eps=eps)
        if metric == "cosine":
            return fit_cosine_beta_mixture(distances, max_iter=max_iter, eps=eps)
        raise ValueError("BMM prior currently supports euclidean and cosine metrics")
    if prior_model == "gmm":
        params = fit_gaussian_mixture(distances, max_iter=max_iter, eps=eps)
        params["positive_component"] = _positive_component_for_gmm(params, metric)
        params["metric"] = metric
        return params
    raise ValueError(f"Unsupported prior_model: {prior_model}")


def fit_similarity_beta_mixture(similarities, max_iter=200, eps=1e-6):
    similarities = np.asarray(similarities, dtype=np.float64)
    scalar_input = similarities.ndim == 1
    values_for_fit = similarities.reshape(-1, 1) if scalar_input else similarities.reshape(similarities.shape[0], -1)
    finite_mask = np.all(np.isfinite(values_for_fit), axis=1)
    values_for_fit = values_for_fit[finite_mask]
    if values_for_fit.size == 0:
        raise ValueError("No finite similarity values were provided for BMM fitting")

    similarity_min = np.min(values_for_fit, axis=0)
    similarity_max = np.max(values_for_fit, axis=0)
    similarity_range = similarity_max - similarity_min
    if np.all(similarity_range < eps):
        similarity_scale = np.ones_like(similarity_range)
        scaled_min = similarity_min - 0.5 * similarity_scale
        return {
            "alpha": np.array([2.0, 2.0], dtype=np.float32) if scalar_input else np.full((2, values_for_fit.shape[1]), 2.0, dtype=np.float32),
            "beta": np.array([2.0, 2.0], dtype=np.float32) if scalar_input else np.full((2, values_for_fit.shape[1]), 2.0, dtype=np.float32),
            "weights": np.array([0.5, 0.5], dtype=np.float32),
            "means": np.array([0.5, 0.5], dtype=np.float32) if scalar_input else np.full((2, values_for_fit.shape[1]), 0.5, dtype=np.float32),
            "variances": np.array([0.05, 0.05], dtype=np.float32) if scalar_input else np.full((2, values_for_fit.shape[1]), 0.05, dtype=np.float32),
            "high_component": 1,
            "positive_component": 1,
            "converged": True,
            "n_iter": 0,
            "similarity_min": np.array(scaled_min.reshape(()) if scalar_input else scaled_min, dtype=np.float32),
            "similarity_scale": np.array(similarity_scale.reshape(()) if scalar_input else similarity_scale, dtype=np.float32),
            "similarity_max": np.array(similarity_max.reshape(()) if scalar_input else similarity_max, dtype=np.float32),
            "model_type": "similarity_bmm",
            "metric": "similarity",
        }
    else:
        similarity_scale = np.maximum(similarity_range, eps)
        scaled_min = similarity_min
        values = (values_for_fit - scaled_min) / similarity_scale

    values = np.clip(values, eps, 1.0 - eps)
    fit_values = values.reshape(-1) if scalar_input else values
    params = fit_beta_mixture(fit_values, max_iter=max_iter, eps=eps)
    means = np.asarray(params["means"], dtype=np.float64).reshape(2, -1)
    params["positive_component"] = int(np.argmax(means.mean(axis=1)))
    params["similarity_min"] = np.array(scaled_min.reshape(()) if scalar_input else scaled_min, dtype=np.float32)
    params["similarity_scale"] = np.array(similarity_scale.reshape(()) if scalar_input else similarity_scale, dtype=np.float32)
    params["similarity_max"] = np.array(similarity_max.reshape(()) if scalar_input else similarity_max, dtype=np.float32)
    params["model_type"] = "similarity_bmm"
    params["metric"] = "similarity"
    return params


def fit_cosine_beta_mixture(similarities, max_iter=200, eps=1e-6):
    similarities = np.asarray(similarities, dtype=np.float64)
    scalar_input = similarities.ndim == 1
    values = similarities.reshape(-1, 1) if scalar_input else similarities.reshape(similarities.shape[0], -1)
    finite_mask = np.all(np.isfinite(values), axis=1)
    values = values[finite_mask]
    if values.size == 0:
        raise ValueError("No finite cosine similarity values were provided for BMM fitting")

    values = np.clip(values, eps, 1.0 - eps)
    fit_values = values.reshape(-1) if scalar_input else values
    params = fit_beta_mixture(fit_values, max_iter=max_iter, eps=eps)
    means = np.asarray(params["means"], dtype=np.float64).reshape(2, -1)
    params["positive_component"] = int(np.argmax(means.mean(axis=1)))
    params["model_type"] = "cosine_bmm"
    params["metric"] = "cosine"
    return params


def print_mixture_fit_result(name, params):
    scale_text = ""
    if "distance_scale" in params:
        scale_value = np.asarray(params["distance_scale"])
        if scale_value.ndim == 0:
            scale_text = f" | distance_scale={float(scale_value):.6f}"
        else:
            scale_text = f" | distance_scale={scale_value.tolist()}"
    print(
        f"Prior fit [{name}] model={params.get('model_type')} | "
        f"converged={params.get('converged')} | "
        f"n_iter={params.get('n_iter')} | "
        f"weights={np.asarray(params['weights']).tolist()} | "
        f"means={np.asarray(params['means']).tolist()} | "
        f"variances={np.asarray(params['variances']).tolist()} | "
        f"positive_component={params.get('positive_component')}"
        f"{scale_text}"
    )


def _safe_plot_name(name):
    return "".join(ch if ch.isalnum() or ch in ["-", "_"] else "_" for ch in name)


def _points_to_svg(points):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def save_mixture_fit_svg(name, distances, x, mixture_pdf, component_pdf, mean_lines, x_label, plot_dir, artifact_stem, bins=80):
    from xml.sax.saxutils import escape

    width, height = 900, 560
    left, right, top, bottom = 80, 30, 55, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_x = float(max(np.max(x), 1e-8))
    hist_density, hist_edges = np.histogram(distances, bins=bins, range=(0.0, max_x), density=True)
    y_max = float(max(np.max(hist_density), np.max(mixture_pdf), 1e-8) * 1.08)

    def sx(value):
        return left + (float(value) / max_x) * plot_width

    def sy(value):
        return top + plot_height - (float(value) / y_max) * plot_height

    plot_path = os.path.join(plot_dir, f"{artifact_stem}_{_safe_plot_name(name)}_fit.svg")
    colors = ["#E45756", "#54A24B"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">Prior distance fit: {escape(name)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>',
        f'<text x="{width / 2:.1f}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="13">{escape(x_label)}</text>',
        f'<text x="22" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 22 {top + plot_height / 2:.1f})" font-family="Arial" font-size="13">density</text>',
        f'<text x="{left}" y="{top + plot_height + 20}" text-anchor="middle" font-family="Arial" font-size="11">0</text>',
        f'<text x="{left + plot_width}" y="{top + plot_height + 20}" text-anchor="middle" font-family="Arial" font-size="11">{max_x:.3g}</text>',
        f'<text x="{left - 8}" y="{top + plot_height}" text-anchor="end" font-family="Arial" font-size="11">0</text>',
        f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-family="Arial" font-size="11">{y_max:.3g}</text>',
    ]

    for density, edge_left, edge_right in zip(hist_density, hist_edges[:-1], hist_edges[1:]):
        bar_x = sx(edge_left)
        bar_w = max(sx(edge_right) - sx(edge_left) - 1, 0.5)
        bar_y = sy(density)
        bar_h = top + plot_height - bar_y
        parts.append(f'<rect x="{bar_x:.2f}" y="{bar_y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#4C78A8" opacity="0.35"/>')

    mixture_points = [(sx(value), sy(pdf)) for value, pdf in zip(x, mixture_pdf)]
    parts.append(f'<polyline points="{_points_to_svg(mixture_points)}" fill="none" stroke="#111111" stroke-width="2.4"/>')
    for component, pdf in enumerate(component_pdf):
        points = [(sx(value), sy(value_pdf)) for value, value_pdf in zip(x, pdf)]
        parts.append(f'<polyline points="{_points_to_svg(points)}" fill="none" stroke="{colors[component]}" stroke-width="1.8" stroke-dasharray="6,4"/>')
        mean_x = sx(mean_lines[component])
        parts.append(f'<line x1="{mean_x:.2f}" y1="{top}" x2="{mean_x:.2f}" y2="{top + plot_height}" stroke="{colors[component]}" opacity="0.75"/>')

    legend_x = left + plot_width - 185
    legend_y = top + 15
    parts.extend([
        f'<rect x="{legend_x - 10}" y="{legend_y - 14}" width="185" height="82" fill="white" stroke="#ddd" opacity="0.92"/>',
        f'<rect x="{legend_x}" y="{legend_y}" width="16" height="10" fill="#4C78A8" opacity="0.35"/><text x="{legend_x + 24}" y="{legend_y + 10}" font-family="Arial" font-size="12">distance histogram</text>',
        f'<line x1="{legend_x}" y1="{legend_y + 26}" x2="{legend_x + 16}" y2="{legend_y + 26}" stroke="#111111" stroke-width="2.4"/><text x="{legend_x + 24}" y="{legend_y + 30}" font-family="Arial" font-size="12">mixture</text>',
        f'<line x1="{legend_x}" y1="{legend_y + 46}" x2="{legend_x + 16}" y2="{legend_y + 46}" stroke="{colors[0]}" stroke-width="1.8" stroke-dasharray="6,4"/><text x="{legend_x + 24}" y="{legend_y + 50}" font-family="Arial" font-size="12">component 0</text>',
        f'<line x1="{legend_x}" y1="{legend_y + 64}" x2="{legend_x + 16}" y2="{legend_y + 64}" stroke="{colors[1]}" stroke-width="1.8" stroke-dasharray="6,4"/><text x="{legend_x + 24}" y="{legend_y + 68}" font-family="Arial" font-size="12">component 1</text>',
        "</svg>",
    ])
    with open(plot_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))
    print(f"Saved prior fit plot [{name}]: {plot_path}")
    return plot_path


def save_mixture_fit_plot(name, distances, params, plot_dir, artifact_stem, bins=80, eps=1e-8):
    if plot_dir is None:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        plt = None
        print(f"matplotlib unavailable for prior fit plot [{name}] ({exc}); saving SVG fallback")

    distances = np.asarray(distances, dtype=np.float64)
    os.makedirs(plot_dir, exist_ok=True)
    min_distance = float(np.min(distances))
    max_distance = float(np.max(distances))
    if np.isclose(max_distance, min_distance):
        min_distance -= 0.5
        max_distance = 1.0
    if min_distance > 0:
        min_distance = 0.0
    x = np.linspace(min_distance, max_distance, 512)
    weights = np.asarray(params["weights"], dtype=np.float64)
    means = np.asarray(params["means"], dtype=np.float64)
    variances = np.asarray(params["variances"], dtype=np.float64)
    model_type = params.get("model_type", "bmm")

    component_pdf = []
    if model_type == "bmm":
        scale = max(float(params["distance_scale"]), eps)
        z = np.clip(x / scale, eps, 1.0 - eps)
        for component in range(2):
            alpha = float(params["alpha"][component])
            beta = float(params["beta"][component])
            pdf_z = np.exp(_log_beta_pdf(z, alpha, beta, eps=eps))
            component_pdf.append(weights[component] * pdf_z / scale)
        mean_lines = means * scale
        x_label = "raw Euclidean distance (BMM fitted on distance / scale)"
    elif model_type == "cosine_bmm":
        z = np.clip(x, eps, 1.0 - eps)
        for component in range(2):
            alpha = float(params["alpha"][component])
            beta = float(params["beta"][component])
            pdf_z = np.exp(_log_beta_pdf(z, alpha, beta, eps=eps))
            component_pdf.append(weights[component] * pdf_z)
        mean_lines = means
        x_label = "cosine similarity (BMM fitted directly)"
    elif model_type == "similarity_bmm":
        similarity_min = float(params["similarity_min"])
        similarity_scale = max(float(params["similarity_scale"]), eps)
        z = np.clip((x - similarity_min) / similarity_scale, eps, 1.0 - eps)
        for component in range(2):
            alpha = float(params["alpha"][component])
            beta = float(params["beta"][component])
            pdf_z = np.exp(_log_beta_pdf(z, alpha, beta, eps=eps))
            component_pdf.append(weights[component] * pdf_z / similarity_scale)
        mean_lines = similarity_min + means * similarity_scale
        x_label = "cosine similarity (BMM fitted on scaled similarity)"
    else:
        for component in range(2):
            variance = max(float(variances[component]), eps)
            pdf = np.exp(_log_gaussian_pdf(x, float(means[component]), variance, eps=eps))
            component_pdf.append(weights[component] * pdf)
        mean_lines = means
        metric = params.get("metric", "euclidean")
        x_label = "cosine similarity" if metric == "cosine" else "raw Euclidean distance"

    mixture_pdf = np.sum(np.stack(component_pdf, axis=0), axis=0)
    if plt is None:
        return save_mixture_fit_svg(
            name,
            distances,
            x,
            mixture_pdf,
            component_pdf,
            mean_lines,
            x_label,
            plot_dir,
            artifact_stem,
            bins=bins,
        )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.hist(distances, bins=bins, density=True, alpha=0.35, color="#4C78A8", label="distance histogram")
    ax.plot(x, mixture_pdf, color="#111111", linewidth=2.0, label=f"{model_type.upper()} mixture")
    colors = ["#E45756", "#54A24B"]
    for component, pdf in enumerate(component_pdf):
        ax.plot(x, pdf, color=colors[component], linewidth=1.5, linestyle="--", label=f"component {component}")
        ax.axvline(mean_lines[component], color=colors[component], linewidth=1.0, alpha=0.7)
    ax.set_title(f"Prior distance fit: {name}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(plot_dir, f"{artifact_stem}_{_safe_plot_name(name)}_fit.png")
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved prior fit plot [{name}]: {plot_path}")
    return plot_path


def mixture_positive_distance_responsibility_torch(distance, params, eps=1e-6):
    model_type = params.get("model_type", "bmm")
    if model_type == "gmm":
        values = distance
        means = params["means"].to(device=values.device, dtype=values.dtype)
        variances = params["variances"].to(device=values.device, dtype=values.dtype).clamp_min(eps)
        weights = params["weights"].to(device=values.device, dtype=values.dtype)
        positive_component = int(params["positive_component"])

        if means.ndim == 1:
            flat_values = values.reshape(-1, 1)
            output_shape = values.shape
            means = means.reshape(2, 1)
            variances = variances.reshape(2, 1)
        else:
            flat_values = values.reshape(-1, means.shape[-1])
            output_shape = values.shape[:-1]
        log_prob = (
            torch.log(weights + eps)
            - 0.5 * (
                torch.log(2.0 * torch.tensor(math.pi, device=values.device, dtype=values.dtype) * variances)
                + ((flat_values[:, None, :] - means[None, :, :]) ** 2) / variances[None, :, :]
            ).sum(dim=2)
        )
        responsibilities = torch.softmax(log_prob, dim=1)
        return responsibilities[:, positive_component].reshape(output_shape)

    if model_type == "cosine_bmm":
        return mixture_positive_cosine_bmm_responsibility_torch(distance, params, eps=eps)

    if model_type == "similarity_bmm":
        return mixture_positive_similarity_responsibility_torch(distance, params, eps=eps)

    distance_scale = params["distance_scale"].to(device=distance.device, dtype=distance.dtype)
    values = torch.clamp(distance / torch.clamp(distance_scale, min=eps), eps, 1.0 - eps)
    alpha = params["alpha"].to(device=values.device, dtype=values.dtype)
    beta = params["beta"].to(device=values.device, dtype=values.dtype)
    weights = params["weights"].to(device=values.device, dtype=values.dtype)
    positive_component = int(params["positive_component"])

    if alpha.ndim == 1:
        flat_values = values.reshape(-1, 1)
        output_shape = values.shape
        alpha = alpha.reshape(2, 1)
        beta = beta.reshape(2, 1)
    else:
        flat_values = values.reshape(-1, alpha.shape[-1])
        output_shape = values.shape[:-1]
    log_norm = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    log_prob = (
        torch.log(weights + eps)
        + (
            log_norm[None, :, :]
            + (alpha[None, :, :] - 1.0) * torch.log(flat_values[:, None, :])
            + (beta[None, :, :] - 1.0) * torch.log1p(-flat_values[:, None, :])
        ).sum(dim=2)
    )
    responsibilities = torch.softmax(log_prob, dim=1)
    return responsibilities[:, positive_component].reshape(output_shape)


def prior_pair_value_matrix(prior_features, params, segment_idx=None, eps=1e-8):
    if segment_idx is None:
        if prior_features.ndim >= 4 and prior_features.shape[-1] == 2:
            features = prior_features.reshape(prior_features.shape[0], -1, 2)
        else:
            features = prior_features.reshape(prior_features.shape[0], -1)
    else:
        features = prior_features[:, segment_idx, :]
    metric = params.get("metric", "euclidean")
    if features.ndim == 3 and features.shape[-1] == 2:
        if metric == "cosine_euclidean":
            raise ValueError("Dual magnitude/delta priors support euclidean or cosine metrics, not cosine_euclidean")
        values = []
        for variable_idx in range(features.shape[-1]):
            variable_features = features[..., variable_idx]
            if metric == "euclidean":
                values.append(torch.cdist(variable_features, variable_features, p=2))
            elif metric == "cosine":
                normalized = torch.nn.functional.normalize(variable_features, dim=1, eps=eps)
                values.append(torch.mm(normalized, normalized.t()))
            else:
                raise ValueError(f"Unsupported prior metric: {metric}")
        return torch.stack(values, dim=-1)

    if metric == "euclidean":
        return torch.cdist(features, features, p=2)
    normalized = torch.nn.functional.normalize(features, dim=1, eps=eps)
    cosine = torch.mm(normalized, normalized.t())
    if metric == "cosine":
        return cosine
    if metric == "cosine_euclidean":
        euclidean = torch.cdist(features, features, p=2)
        return torch.stack((cosine, euclidean), dim=-1)
    raise ValueError(f"Unsupported prior metric: {metric}")


def prior_positive_probability_matrix(prior_features, params, segment_idx=None):
    prior_values = prior_pair_value_matrix(prior_features, params, segment_idx=segment_idx)
    return mixture_positive_distance_responsibility_torch(prior_values, params)


def mixture_positive_similarity_responsibility_torch(similarity, params, eps=1e-6):
    similarity_min = params["similarity_min"].to(device=similarity.device, dtype=similarity.dtype)
    similarity_scale = params["similarity_scale"].to(device=similarity.device, dtype=similarity.dtype)
    values = torch.clamp(
        (similarity - similarity_min) / torch.clamp(similarity_scale, min=eps),
        eps,
        1.0 - eps,
    )
    alpha = params["alpha"].to(device=values.device, dtype=values.dtype)
    beta = params["beta"].to(device=values.device, dtype=values.dtype)
    weights = params["weights"].to(device=values.device, dtype=values.dtype)
    positive_component = int(params["positive_component"])

    if alpha.ndim == 1:
        flat_values = values.reshape(-1, 1)
        output_shape = values.shape
        alpha = alpha.reshape(2, 1)
        beta = beta.reshape(2, 1)
    else:
        flat_values = values.reshape(-1, alpha.shape[-1])
        output_shape = values.shape[:-1]
    log_norm = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    log_prob = (
        torch.log(weights + eps)
        + (
            log_norm[None, :, :]
            + (alpha[None, :, :] - 1.0) * torch.log(flat_values[:, None, :])
            + (beta[None, :, :] - 1.0) * torch.log1p(-flat_values[:, None, :])
        ).sum(dim=2)
    )
    responsibilities = torch.softmax(log_prob, dim=1)
    return responsibilities[:, positive_component].reshape(output_shape)


def mixture_positive_cosine_bmm_responsibility_torch(similarity, params, eps=1e-6):
    values = torch.clamp(similarity, eps, 1.0 - eps)
    alpha = params["alpha"].to(device=values.device, dtype=values.dtype)
    beta = params["beta"].to(device=values.device, dtype=values.dtype)
    weights = params["weights"].to(device=values.device, dtype=values.dtype)
    positive_component = int(params["positive_component"])

    if alpha.ndim == 1:
        flat_values = values.reshape(-1, 1)
        output_shape = values.shape
        alpha = alpha.reshape(2, 1)
        beta = beta.reshape(2, 1)
    else:
        flat_values = values.reshape(-1, alpha.shape[-1])
        output_shape = values.shape[:-1]
    log_norm = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    log_prob = (
        torch.log(weights + eps)
        + (
            log_norm[None, :, :]
            + (alpha[None, :, :] - 1.0) * torch.log(flat_values[:, None, :])
            + (beta[None, :, :] - 1.0) * torch.log1p(-flat_values[:, None, :])
        ).sum(dim=2)
    )
    responsibilities = torch.softmax(log_prob, dim=1)
    return responsibilities[:, positive_component].reshape(output_shape)


def similarity_bmm_probability_matrix(similarity, max_iter=200, eps=1e-6):
    params = fit_similarity_beta_mixture(
        similarity.detach().cpu().numpy(),
        max_iter=max_iter,
        eps=eps,
    )
    torch_params = bmm_params_to_torch(params)
    return mixture_positive_similarity_responsibility_torch(similarity, torch_params, eps=eps)


def bmm_params_to_torch(params):
    torch_params = {
        "weights": torch.tensor(params["weights"], dtype=torch.float32),
        "means": torch.tensor(params["means"], dtype=torch.float32),
        "variances": torch.tensor(params["variances"], dtype=torch.float32),
        "positive_component": int(params.get("positive_component", params.get("high_component", np.argmax(params["means"])))),
        "converged": bool(params.get("converged", False)),
        "n_iter": int(params.get("n_iter", 0)),
        "model_type": params.get("model_type", "bmm"),
    }
    if "alpha" in params:
        torch_params["alpha"] = torch.tensor(params["alpha"], dtype=torch.float32)
    if "beta" in params:
        torch_params["beta"] = torch.tensor(params["beta"], dtype=torch.float32)
    if "high_component" in params:
        torch_params["high_component"] = int(params["high_component"])
    if "distance_scale" in params:
        torch_params["distance_scale"] = torch.tensor(params["distance_scale"], dtype=torch.float32)
    if "similarity_min" in params:
        torch_params["similarity_min"] = torch.tensor(params["similarity_min"], dtype=torch.float32)
    if "similarity_scale" in params:
        torch_params["similarity_scale"] = torch.tensor(params["similarity_scale"], dtype=torch.float32)
    if "similarity_max" in params:
        torch_params["similarity_max"] = torch.tensor(params["similarity_max"], dtype=torch.float32)
    if "metric" in params:
        torch_params["metric"] = params["metric"]
    return torch_params


def compute_adjacent_delta_features(features):
    delta = np.zeros_like(features)
    delta[:, 1:, :] = features[:, 1:, :] - features[:, :-1, :]
    return delta


def apply_delta_prior_mode(features, prior_delta_mode):
    prior_delta_mode = prior_delta_mode.lower()
    if prior_delta_mode == "none":
        return features

    delta = compute_adjacent_delta_features(features)
    if prior_delta_mode == "delta":
        return delta
    if prior_delta_mode == "concat":
        return np.concatenate((features, delta), axis=-1)
    if prior_delta_mode == "both":
        return np.stack((features, delta), axis=-1)
    raise ValueError(f"Unsupported prior_delta_mode: {prior_delta_mode}")


def concatenate_prior_feature_sets(feature_sets):
    if feature_sets and _is_dual_prior_features(feature_sets[0]):
        return np.concatenate(feature_sets, axis=-2)
    return np.concatenate(feature_sets, axis=-1)


def build_magnitude_prior(
    data,
    dataset_name,
    segment_len=4,
    num_random_pairs=3500,
    num_self_pairs=500,
    seed=0,
    prior_mode="combined",
    prior_model="bmm",
    prior_gmm_metric="euclidean",
    prior_delta_mode="none",
    prior_fit_max_iter=200,
    three_mod_contrast="pairwise",
    plot=False,
    plot_dir=None,
    artifact_stem=None,
):
    prior_mode = prior_mode.lower()
    prior_model = prior_model.lower()
    prior_gmm_metric = prior_gmm_metric.lower()
    prior_delta_mode = prior_delta_mode.lower()
    three_mod_contrast = three_mod_contrast.lower()
    if prior_model == "bmm" and prior_gmm_metric == "cosine_euclidean":
        raise ValueError("BMM prior supports euclidean or cosine; use --prior_model gmm for cosine_euclidean")
    if prior_delta_mode == "both" and prior_gmm_metric == "cosine_euclidean":
        raise ValueError("prior_delta_mode='both' supports euclidean or cosine metrics, not cosine_euclidean")
    if prior_delta_mode not in ["none", "delta", "concat", "both"]:
        raise ValueError(f"Unsupported prior_delta_mode: {prior_delta_mode}")
    if three_mod_contrast not in ["pairwise", "1vsall"]:
        raise ValueError(f"Unsupported three_mod_contrast: {three_mod_contrast}")
    rng = np.random.RandomState(seed)
    modality_features = extract_magnitude_features(
        data,
        dataset_name,
        segment_len=segment_len,
        method="rms",
        log_transform=True,
        normalize_method="none",
    )
    modality_features = [
        _as_float32(apply_delta_prior_mode(features, prior_delta_mode))
        for features in modality_features
    ]
    combined_features = _as_float32(concatenate_prior_feature_sets(modality_features))
    branch_features = {}
    if len(modality_features) >= 3:
        if three_mod_contrast == "pairwise":
            inter_specs = [
                (0, 1), (1, 0),
                (0, 2), (2, 0),
                (1, 2), (2, 1),
            ]
            for left_idx, right_idx in inter_specs:
                branch_features[f"pair_mod{left_idx + 1}_mod{right_idx + 1}"] = _as_float32(
                    concatenate_prior_feature_sets((modality_features[left_idx], modality_features[right_idx]))
                )
        else:
            for anchor_idx in range(len(modality_features)):
                for other_idx in range(len(modality_features)):
                    if anchor_idx == other_idx:
                        continue
                    branch_features[f"anchor_mod{anchor_idx + 1}_mod{other_idx + 1}"] = _as_float32(
                        concatenate_prior_feature_sets((modality_features[anchor_idx], modality_features[other_idx]))
                    )

    if prior_mode == "combined":
        feature_sets = [
            ("combined_sample", combined_features, "sample"),
            ("combined_segment", combined_features, "segment"),
        ]
    elif prior_mode == "separate":
        feature_sets = []
        for mod_idx, features in enumerate(modality_features, start=1):
            feature_sets.extend([
                (f"mod{mod_idx}_sample", features, "sample"),
                (f"mod{mod_idx}_segment", features, "segment"),
            ])
        for branch_name, features in branch_features.items():
            feature_sets.append((f"{branch_name}_sample", features, "sample"))
    else:
        raise ValueError(f"Unsupported prior_mode: {prior_mode}")

    bmm = {}
    plot_paths = {}
    for name, features, granularity in feature_sets:
        if granularity == "sample":
            prior_values, distance_scale = collect_sample_prior_values(
                features,
                num_random_pairs,
                num_self_pairs,
                rng,
                metric=prior_gmm_metric,
            )
        else:
            prior_values, distance_scale = collect_segment_prior_values(
                features,
                num_random_pairs,
                num_self_pairs,
                rng,
                metric=prior_gmm_metric,
            )
        bmm[name] = fit_distance_mixture(
            prior_values,
            distance_scale,
            prior_model=prior_model,
            max_iter=prior_fit_max_iter,
            metric=prior_gmm_metric,
        )
        print_mixture_fit_result(name, bmm[name])
        if plot and np.asarray(prior_values).ndim == 1:
            plot_paths[name] = save_mixture_fit_plot(
                name,
                prior_values,
                bmm[name],
                plot_dir=plot_dir,
                artifact_stem=artifact_stem or "magnitude_prior",
            )
        elif plot:
            print(f"Skipping prior fit plot [{name}] for multidimensional metric={prior_gmm_metric}")

    return {
        **{
            f"mod{mod_idx}_features": features
            for mod_idx, features in enumerate(modality_features, start=1)
        },
        **{
            f"{branch_name}_features": features
            for branch_name, features in branch_features.items()
        },
        "combined_features": combined_features,
        "bmm": bmm,
        "plot_paths": plot_paths,
    }


def prior_artifact_path(
    save_dir,
    dataset_name,
    fold,
    seed,
    segment_len,
    prior_mode="combined",
    prior_model="bmm",
    prior_gmm_metric="euclidean",
    prior_delta_mode="none",
    prior_fit_max_iter=200,
    three_mod_contrast="pairwise",
):
    prior_model = prior_model.lower()
    prior_gmm_metric = prior_gmm_metric.lower()
    prior_delta_mode = prior_delta_mode.lower()
    three_mod_contrast = three_mod_contrast.lower()
    metric_part = prior_gmm_metric
    contrast_part = f"_contrast-{three_mod_contrast}"
    filename = (
        f"magnitude_prior_{prior_mode}_{prior_model}_sample_segment_{metric_part}_"
        f"delta-{prior_delta_mode}_iter{prior_fit_max_iter}{contrast_part}_"
        f"{dataset_name}_fold{fold}_seed{seed}_seg{segment_len}.npz"
    )
    return os.path.join(save_dir, filename)


def save_prior_artifact(path, prior, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {
        "combined_features": prior["combined_features"],
        "metadata_json": np.array(json.dumps(metadata)),
    }
    for key, value in prior.items():
        if key.endswith("_features"):
            arrays[key] = value
    for name, params in prior["bmm"].items():
        arrays[f"{name}_weights"] = params["weights"]
        arrays[f"{name}_means"] = params["means"]
        arrays[f"{name}_variances"] = params["variances"]
        arrays[f"{name}_positive_component"] = np.array(params["positive_component"], dtype=np.int64)
        arrays[f"{name}_converged"] = np.array(params.get("converged", False), dtype=np.bool_)
        arrays[f"{name}_n_iter"] = np.array(params.get("n_iter", 0), dtype=np.int64)
        arrays[f"{name}_model_type"] = np.array(params.get("model_type", metadata.get("prior_model", "bmm")))
        if "alpha" in params:
            arrays[f"{name}_alpha"] = params["alpha"]
        if "beta" in params:
            arrays[f"{name}_beta"] = params["beta"]
        if "high_component" in params:
            arrays[f"{name}_high_component"] = np.array(params.get("high_component", np.argmax(params["means"])), dtype=np.int64)
        if "distance_scale" in params:
            arrays[f"{name}_distance_scale"] = np.array(params["distance_scale"], dtype=np.float32)
        if "similarity_min" in params:
            arrays[f"{name}_similarity_min"] = np.array(params["similarity_min"], dtype=np.float32)
        if "similarity_scale" in params:
            arrays[f"{name}_similarity_scale"] = np.array(params["similarity_scale"], dtype=np.float32)
        if "similarity_max" in params:
            arrays[f"{name}_similarity_max"] = np.array(params["similarity_max"], dtype=np.float32)
    np.savez_compressed(path, **arrays)
    return path


def load_prior_artifact(path):
    loaded = np.load(path, allow_pickle=False)
    metadata = json.loads(str(loaded["metadata_json"]))
    prior_model = metadata.get("prior_model", "bmm")
    prior_metric = metadata.get("prior_gmm_metric", metadata.get("metric", "euclidean"))
    prior_keys = metadata.get("prior_keys", ["combined_sample"])
    bmm = {}
    for name in prior_keys:
        model_type = prior_model
        model_type_key = f"{name}_model_type"
        if model_type_key in loaded:
            model_type = str(loaded[model_type_key])
        elif prior_model == "bmm" and prior_metric == "cosine":
            model_type = "cosine_bmm"
        params = {
            "weights": loaded[f"{name}_weights"],
            "means": loaded[f"{name}_means"],
            "variances": loaded[f"{name}_variances"],
            "positive_component": int(loaded[f"{name}_positive_component"]),
            "converged": bool(loaded[f"{name}_converged"]),
            "n_iter": int(loaded[f"{name}_n_iter"]),
            "model_type": model_type,
            "metric": prior_metric,
        }
        if prior_model == "bmm":
            params.update({
                "alpha": loaded[f"{name}_alpha"],
                "beta": loaded[f"{name}_beta"],
                "high_component": int(loaded[f"{name}_high_component"]),
            })
            if f"{name}_distance_scale" in loaded:
                params["distance_scale"] = np.array(loaded[f"{name}_distance_scale"], dtype=np.float32)
            if f"{name}_similarity_min" in loaded:
                params["similarity_min"] = np.array(loaded[f"{name}_similarity_min"], dtype=np.float32)
            if f"{name}_similarity_scale" in loaded:
                params["similarity_scale"] = np.array(loaded[f"{name}_similarity_scale"], dtype=np.float32)
            if f"{name}_similarity_max" in loaded:
                params["similarity_max"] = np.array(loaded[f"{name}_similarity_max"], dtype=np.float32)
        bmm[name] = params
    num_modalities = int(metadata.get("num_modalities", 2))
    prior = {
        "combined_features": loaded["combined_features"],
        "bmm": bmm,
        "metadata": metadata,
        "path": path,
        "loaded": True,
    }
    for key in loaded.files:
        if key.endswith("_features"):
            prior[key] = loaded[key]
    return prior


def load_or_build_magnitude_prior(
    data,
    dataset_name,
    fold,
    seed,
    save_dir,
    segment_len=4,
    num_random_pairs=3500,
    num_self_pairs=500,
    prior_mode="combined",
    prior_model="bmm",
    prior_gmm_metric="euclidean",
    prior_delta_mode="none",
    prior_fit_max_iter=200,
    three_mod_contrast="pairwise",
    plot=False,
):
    prior_mode = prior_mode.lower()
    prior_model = prior_model.lower()
    prior_gmm_metric = prior_gmm_metric.lower()
    prior_delta_mode = prior_delta_mode.lower()
    three_mod_contrast = three_mod_contrast.lower()
    if prior_model == "bmm" and prior_gmm_metric == "cosine_euclidean":
        raise ValueError("BMM prior supports euclidean or cosine; use --prior_model gmm for cosine_euclidean")
    if prior_delta_mode == "both" and prior_gmm_metric == "cosine_euclidean":
        raise ValueError("prior_delta_mode='both' supports euclidean or cosine metrics, not cosine_euclidean")
    if prior_delta_mode not in ["none", "delta", "concat", "both"]:
        raise ValueError(f"Unsupported prior_delta_mode: {prior_delta_mode}")
    path = prior_artifact_path(
        save_dir,
        dataset_name,
        fold,
        seed,
        segment_len,
        prior_mode,
        prior_model,
        prior_gmm_metric,
        prior_delta_mode,
        prior_fit_max_iter,
        three_mod_contrast,
    )
    # do not load saved prior
    # if os.path.exists(path):
    #     prior = load_prior_artifact(path)
    #     print(f"Loaded magnitude prior artifact: {path}")
    #     return prior

    artifact_stem = os.path.splitext(os.path.basename(path))[0]
    prior = build_magnitude_prior(
        data,
        dataset_name,
        segment_len=segment_len,
        num_random_pairs=num_random_pairs,
        num_self_pairs=num_self_pairs,
        seed=seed,
        prior_mode=prior_mode,
        prior_model=prior_model,
        prior_gmm_metric=prior_gmm_metric,
        prior_delta_mode=prior_delta_mode,
        prior_fit_max_iter=prior_fit_max_iter,
        three_mod_contrast=three_mod_contrast,
        plot=plot,
        plot_dir=save_dir,
        artifact_stem=artifact_stem,
    )
    if prior_delta_mode == "none":
        prior_feature = "log_rms_magnitude"
    elif prior_delta_mode == "delta":
        prior_feature = "log_rms_delta"
    elif prior_delta_mode == "both":
        prior_feature = "log_rms_magnitude_and_delta_separate"
    else:
        prior_feature = "log_rms_magnitude_delta_concat"
    num_modalities = len([key for key in prior if key.startswith("mod") and key.endswith("_features")])
    feature_shapes = {
        key[:-len("_features")]: list(value.shape)
        for key, value in prior.items()
        if key.endswith("_features")
    }
    metadata = {
        "dataset_name": dataset_name,
        "num_modalities": num_modalities,
        "three_mod_contrast": three_mod_contrast,
        "fold": fold,
        "seed": seed,
        "segment_len": segment_len,
        "num_random_pairs": num_random_pairs,
        "num_self_pairs": num_self_pairs,
        "prior_feature": prior_feature,
        "prior_delta_mode": prior_delta_mode,
        "prior_fit_max_iter": prior_fit_max_iter,
        "normalization": "none",
        "metric": prior_gmm_metric,
        "prior_gmm_metric": prior_gmm_metric,
        "prior_mode": prior_mode,
        "prior_model": prior_model,
        "prior_keys": sorted(prior["bmm"].keys()),
        "prior_level": "sample_segment",
        "plot_paths": prior.get("plot_paths", {}),
        "feature_shapes": feature_shapes,
    }
    # do not save prior artifact
    # save_prior_artifact(path, prior, metadata)
    prior["metadata"] = metadata
    prior["path"] = path
    prior["loaded"] = False
    # print(f"Saved magnitude prior artifact: {path}")
    return prior


def prepare_prior_for_torch(prior):
    if prior is None:
        return None
    return {
        "features": {
            **{
                key[:-len("_features")]: torch.tensor(value, dtype=torch.float32)
                for key, value in prior.items()
                if key.endswith("_features")
            },
        },
        "bmm": {
            name: bmm_params_to_torch(params)
            for name, params in prior["bmm"].items()
        },
        "metadata": prior.get("metadata", {}),
        "path": prior.get("path"),
    }
