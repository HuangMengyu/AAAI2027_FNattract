import os
import numpy as np

# pretrain data construct is usually (size, channel, seq_len)

def channel_separation(time, dataset_name='SleepEDFx'):
    if dataset_name == 'SleepEDFx':
        EEG_channels = time[:, :-1, :]
        EOG_channel = time[:, -1:, :]
        return EEG_channels, EOG_channel
    elif dataset_name == 'SleepEDFx_3':
        mod1 = time[:, :2, :]
        mod2 = time[:, 2:3, :]
        mod3 = time[:, 3:4, :]
        return mod1, mod2, mod3
    elif dataset_name == 'PAMAP2':
        acc_channels = time[:, :-9, :]
        gyro_channels = time[:, -9:, :]
        return acc_channels, gyro_channels
    elif dataset_name == 'PAMAP2_3':
        mod1 = time[:, :9, :]
        mod2 = time[:, 9:18, :]
        mod3 = time[:, 18:27, :]
        return mod1, mod2, mod3
    elif dataset_name in ['UCI-HAR', 'UCI-HAR_total']:
        acc_channels = time[:, :-3, :]
        gyro_channels = time[:, -3:, :]
        return acc_channels, gyro_channels
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

def compute_psd(data, sampling_rate=1.0, axis=-2):
    """
    Compute one-sided PSD for segmented real-valued signals.

    Expected input shape is (..., segment_len, channels), so the default output
    shape is (..., n_freq_bins, channels).
    """
    data = np.asarray(data)
    segment_len = data.shape[axis]
    if segment_len <= 0:
        raise ValueError("segment_len must be positive")

    fft_values = np.fft.rfft(data, axis=axis)
    psd = (np.abs(fft_values) ** 2) / (sampling_rate * segment_len)

    if segment_len > 1:
        scale = np.ones(psd.shape[axis], dtype=psd.dtype)
        if segment_len % 2 == 0:
            scale[1:-1] *= 2
        else:
            scale[1:] *= 2

        scale_shape = [1] * psd.ndim
        scale_shape[axis] = scale.shape[0]
        psd *= scale.reshape(scale_shape)

    return psd

def _segment_modality(data, segment_len):
    sample_size, seq_len, channels = data.shape
    seg_num = seq_len // segment_len
    if seg_num == 0:
        raise ValueError(
            f"segment_len={segment_len} is longer than sequence length={seq_len}"
        )

    valid_len = seg_num * segment_len
    data = data[:, :valid_len, :]
    return np.reshape(data, (sample_size, seg_num, segment_len, channels))

def _group_pamap2_sensors(data):
    sample_size, seg_num, segment_len, channels = data.shape
    if channels % 3 != 0:
        raise ValueError(
            f"PAMAP2 modality channel count must be divisible by 3, got {channels}"
        )

    sensor_num = channels // 3
    data = np.reshape(data, (sample_size, seg_num, segment_len, sensor_num, 3))
    data = np.transpose(data, (0, 1, 3, 2, 4))
    return np.reshape(data, (sample_size, seg_num * sensor_num, segment_len, 3))

def _ungroup_pamap2_psd(psd, sensor_num):
    sample_size, flattened_seg_num, n_freq_bins, axis_num = psd.shape
    if axis_num != 3:
        raise ValueError(f"PAMAP2 PSD axis count should be 3, got {axis_num}")
    if flattened_seg_num % sensor_num != 0:
        raise ValueError(
            "PAMAP2 flattened segment count must be divisible by sensor count"
        )

    seg_num = flattened_seg_num // sensor_num
    psd = np.reshape(psd, (sample_size, seg_num, sensor_num, n_freq_bins, axis_num))
    psd = np.transpose(psd, (0, 1, 3, 2, 4))
    return np.reshape(psd, (sample_size, seg_num, n_freq_bins, sensor_num * axis_num))

def _normalize(data, axis=-1, eps=1e-8, method="zscore"):
    if method == "zscore":
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        return (data - mean) / (std + eps)
    elif method == "l2":
        norm = np.linalg.norm(data, axis=axis, keepdims=True)
        return data / (norm + eps)
    elif method is None or method == "none":
        return data
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

def _feature_normalize_axis(normalize_axis):
    if normalize_axis == "channel":
        return -1
    elif normalize_axis == "segment":
        return 1
    else:
        raise ValueError(f"Unsupported normalize_axis: {normalize_axis}")

def log_normalize_psd(psd, eps=1e-8, normalize_method="zscore", normalize_axis="channel"):
    """
    Convert PSD to log-normalized segment features.

    Input shape:
        (batch_size, seq_num, n_freq_bins, channels)

    Output shape:
        (batch_size, seq_num, n_freq_bins * channels)
    """
    psd = np.asarray(psd)
    if psd.ndim != 4:
        raise ValueError(
            "PSD should have shape (batch_size, seq_num, n_freq_bins, channels)"
        )

    log_psd = np.log(psd + eps)
    batch_size, seq_num, n_freq_bins, channels = log_psd.shape
    features = np.reshape(log_psd, (batch_size, seq_num, n_freq_bins * channels))
    return _normalize(
        features,
        axis=_feature_normalize_axis(normalize_axis),
        eps=eps,
        method=normalize_method,
    )

def compute_segment_magnitude(data, method="rms", axis=-2):
    """
    Compute one magnitude feature per segment and channel.

    Expected input shape is (..., segment_len, channels), so the default output
    shape is (..., channels).
    """
    data = np.asarray(data)
    if method == "rms":
        return np.sqrt(np.mean(data ** 2, axis=axis))
    elif method == "l2":
        return np.sqrt(np.sum(data ** 2, axis=axis))
    elif method == "mean_abs":
        return np.mean(np.abs(data), axis=axis)
    else:
        raise ValueError(f"Unsupported magnitude method: {method}")

def transform_magnitude_features(
    magnitude,
    eps=1e-8,
    log_transform=True,
    normalize_method="zscore",
    normalize_axis="channel",
):
    """
    Convert segment magnitudes to contrastive-prior features.

    Input and output shape:
        (batch_size, seq_num, channels)
    """
    features = np.asarray(magnitude)
    if features.ndim != 3:
        raise ValueError(
            "magnitude should have shape (batch_size, seq_num, channels)"
        )

    if log_transform:
        features = np.log(features + eps)
    return _normalize(
        features,
        axis=_feature_normalize_axis(normalize_axis),
        eps=eps,
        method=normalize_method,
    )

def cosine_similarity_matrix(features, eps=1e-8):
    """
    Compute all-pairs cosine similarity for features shaped (batch_size, dim).
    """
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError("features should have shape (batch_size, dim)")

    features = _normalize(features, axis=-1, eps=eps, method="l2")
    return np.matmul(features, features.T)

def fixed_seq_cosine_similarity(psd_features, seq_idx=0, eps=1e-8):
    """
    Compare samples using one fixed segment only.

    Input shape:
        (batch_size, seq_num, n_freq_bins * channels)

    Output shape:
        (batch_size, batch_size)
    """
    psd_features = np.asarray(psd_features)
    if psd_features.ndim != 3:
        raise ValueError(
            "psd_features should have shape "
            "(batch_size, seq_num, n_freq_bins * channels)"
        )
    if not -psd_features.shape[1] <= seq_idx < psd_features.shape[1]:
        raise IndexError(
            f"seq_idx={seq_idx} is out of range for seq_num={psd_features.shape[1]}"
        )

    return cosine_similarity_matrix(psd_features[:, seq_idx, :], eps=eps)

def overall_seq_cosine_similarity(psd_features, eps=1e-8):
    """
    Compare samples using all segments flattened together.

    Input shape:
        (batch_size, seq_num, n_freq_bins * channels)

    Output shape:
        (batch_size, batch_size)
    """
    psd_features = np.asarray(psd_features)
    if psd_features.ndim != 3:
        raise ValueError(
            "psd_features should have shape "
            "(batch_size, seq_num, n_freq_bins * channels)"
        )

    batch_size = psd_features.shape[0]
    flattened = np.reshape(psd_features, (batch_size, -1))
    return cosine_similarity_matrix(flattened, eps=eps)

def save_computed_prior(save_path, prior_dict=None, compressed=True, **prior_arrays):
    """
    Save computed prior arrays to a NumPy .npz file.

    Examples:
        save_computed_prior(
            "priors/pamap2_train_prior.npz",
            mod1_psd=mod1_psd,
            mod2_psd=mod2_psd,
            mod1_features=mod1_features,
            overall_sim=overall_sim,
        )

        save_computed_prior("prior.npz", {"mod1_psd": mod1_psd})
    """
    if prior_dict is not None:
        duplicated_keys = set(prior_dict).intersection(prior_arrays)
        if duplicated_keys:
            raise ValueError(f"Duplicated prior names: {sorted(duplicated_keys)}")
        prior_arrays = {**prior_dict, **prior_arrays}

    if not prior_arrays:
        raise ValueError("No prior arrays were provided to save")

    parent_dir = os.path.dirname(save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    save_fn = np.savez_compressed if compressed else np.savez
    save_fn(save_path, **prior_arrays)
    return save_path

def extract_priors(data, dataset_name, segment_len=4, sampling_rate=1.0):
    modality_data = channel_separation(data, dataset_name)

    # convert to (batch size, seq_len, channel)
    modality_data = [np.transpose(data, (0, 2, 1)) for data in modality_data]

    # reshape to (batch size, segment_num, segment_len, channel)
    modality_data = [_segment_modality(data, segment_len) for data in modality_data]

    # if PAMAP2: every 3-axis sensor form a group
    sensor_nums = []
    if dataset_name in ["PAMAP2", "PAMAP2_3"]:
        sensor_nums = [data.shape[-1] // 3 for data in modality_data]
        modality_data = [_group_pamap2_sensors(data) for data in modality_data]
    
    # compute the psd for each segment
    modality_psd = [compute_psd(data, sampling_rate=sampling_rate) for data in modality_data]

    if dataset_name in ["PAMAP2", "PAMAP2_3"]:
        modality_psd = [
            _ungroup_pamap2_psd(psd, sensor_num)
            for psd, sensor_num in zip(modality_psd, sensor_nums)
        ]

    return tuple(modality_psd)

def extract_magnitude_priors(data, dataset_name, segment_len=4, method="rms"):
    modality_data = channel_separation(data, dataset_name)

    modality_data = [np.transpose(data, (0, 2, 1)) for data in modality_data]

    modality_data = [_segment_modality(data, segment_len) for data in modality_data]

    modality_magnitude = [
        compute_segment_magnitude(data, method=method)
        for data in modality_data
    ]

    return tuple(modality_magnitude)

def extract_prior_features(
    data,
    dataset_name,
    segment_len=4,
    sampling_rate=1.0,
    eps=1e-8,
    normalize_method="zscore",
    normalize_axis="channel",
):
    modality_psd = extract_priors(
        data,
        dataset_name,
        segment_len=segment_len,
        sampling_rate=sampling_rate,
    )
    modality_features = [
        log_normalize_psd(
            psd,
            eps=eps,
            normalize_method=normalize_method,
            normalize_axis=normalize_axis,
        )
        for psd in modality_psd
    ]

    return tuple(modality_features)

def extract_magnitude_features(
    data,
    dataset_name,
    segment_len=4,
    method="rms",
    eps=1e-8,
    log_transform=True,
    normalize_method="zscore",
    normalize_axis="channel",
):
    modality_magnitude = extract_magnitude_priors(
        data,
        dataset_name,
        segment_len=segment_len,
        method=method,
    )
    modality_features = [
        transform_magnitude_features(
            magnitude,
            eps=eps,
            log_transform=log_transform,
            normalize_method=normalize_method,
            normalize_axis=normalize_axis,
        )
        for magnitude in modality_magnitude
    ]

    return tuple(modality_features)
