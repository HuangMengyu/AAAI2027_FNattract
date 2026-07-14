#!/usr/bin/env python3
"""Visualize samples stored in a processed time-series ``.npy`` file.

The preprocessing scripts in this directory save data with shape
``(samples, channels, time)``.  This utility also supports
``(samples, time, channels)`` and single-sample 1-D/2-D arrays.

Examples
--------
Visualize 6 UCI-HAR samples, showing the first 80 time steps.  This creates
one clean image for every sample-channel pair::

    python3 visualize_dataset.py /path/to/UCI-HAR \
        --subject 01 --num-samples 6 --sample-length 80 \
        --output uci_har_subject01

Visualize selected samples and channels from one file::

    python3 visualize_dataset.py /path/to/101/data/0.npy \
        --sample-indices 2,10,15 --channels 0-2,9-11 \
        --time-start 32 --sample-length 128 --output pamap2_figures
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


def positive_int(value):
    """Argparse type for integers greater than zero."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value):
    """Argparse type for integers greater than or equal to zero."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def parse_indices(spec, upper_bound, argument_name):
    # type: (str, int, str) -> List[int]
    """Parse comma-separated indices and inclusive ranges such as ``0,2-4``."""
    result = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            if "-" in item:
                start_text, end_text = item.split("-", 1)
                start, end = int(start_text), int(end_text)
                if start > end:
                    raise ValueError("range start is greater than its end")
                result.extend(range(start, end + 1))
            else:
                result.append(int(item))
        except ValueError as exc:
            raise ValueError(
                "Invalid {} value {!r}: {}".format(argument_name, item, exc)
            )

    # Preserve the requested order, but avoid plotting an index twice.
    result = list(dict.fromkeys(result))
    if not result:
        raise ValueError("{} did not contain any indices".format(argument_name))
    invalid = [index for index in result if index < 0 or index >= upper_bound]
    if invalid:
        raise ValueError(
            "{} contains out-of-range indices {} (valid range: 0-{})".format(
                argument_name, invalid, upper_bound - 1
            )
        )
    return result


def resolve_data_file(path, subject=None, data_file="0.npy"):
    # type: (Path, Optional[str], str) -> Path
    """Resolve either a direct .npy path or a processed dataset directory."""
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".npy":
            raise ValueError("The input file must be a .npy file: {}".format(path))
        return path
    if not path.is_dir():
        raise FileNotFoundError("Input path does not exist: {}".format(path))

    candidates = []
    if subject is not None:
        candidates.extend(
            [path / subject / "data" / data_file, path / subject / data_file]
        )
    else:
        candidates.extend([path / "data" / data_file, path / data_file])

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    if subject is not None:
        raise FileNotFoundError(
            "Could not find data file {!r} for subject {!r} under {}".format(
                data_file, subject, path
            )
        )

    matches = sorted(path.glob("*/data/{}".format(data_file)))
    if not matches:
        matches = sorted(path.rglob(data_file))
    if not matches:
        raise FileNotFoundError(
            "Could not find {!r} in dataset directory {}".format(data_file, path)
        )
    if len(matches) > 1:
        print(
            "No --subject was given; using the first of {} matching files: {}".format(
                len(matches), matches[0]
            )
        )
    return matches[0].resolve()


def find_label_file(data_file, explicit_label_file=None):
    # type: (Path, Optional[Path]) -> Optional[Path]
    """Find the label file that mirrors ``.../data/name.npy``."""
    if explicit_label_file is not None:
        label_file = explicit_label_file.expanduser().resolve()
        if not label_file.is_file():
            raise FileNotFoundError("Label file does not exist: {}".format(label_file))
        return label_file

    if data_file.parent.name == "data":
        candidate = data_file.parent.parent / "label" / data_file.name
        if candidate.is_file():
            return candidate
    return None


def to_samples_channels_time(array, layout="auto"):
    # type: (np.ndarray, str) -> np.ndarray
    """Return a view of the input with shape (samples, channels, time)."""
    if array.ndim == 1:
        return array[np.newaxis, np.newaxis, :]

    if array.ndim == 2:
        selected_layout = layout
        if selected_layout == "auto":
            selected_layout = "CT" if array.shape[0] <= array.shape[1] else "TC"
        if selected_layout == "CT":
            return array[np.newaxis, :, :]
        if selected_layout == "TC":
            return array.T[np.newaxis, :, :]
        raise ValueError(
            "A 2-D array requires --layout auto, CT, or TC; got {}".format(layout)
        )

    if array.ndim == 3:
        selected_layout = layout
        if selected_layout == "auto":
            selected_layout = "NCT" if array.shape[1] <= array.shape[2] else "NTC"
        if selected_layout == "NCT":
            return array
        if selected_layout == "NTC":
            return array.transpose(0, 2, 1)
        raise ValueError(
            "A 3-D array requires --layout auto, NCT, or NTC; got {}".format(layout)
        )

    raise ValueError(
        "Expected a 1-D, 2-D, or 3-D array, but found shape {}".format(array.shape)
    )


def parse_channel_names(spec, channel_count):
    # type: (Optional[str], int) -> List[str]
    if spec is None:
        return ["channel {}".format(index) for index in range(channel_count)]
    names = [name.strip() for name in spec.split(",")]
    if len(names) != channel_count or any(not name for name in names):
        raise ValueError(
            "--channel-names must contain exactly {} non-empty comma-separated "
            "names".format(channel_count)
        )
    return names


def format_label(labels, sample_index):
    # type: (Optional[np.ndarray], int) -> Optional[str]
    if labels is None:
        return None
    if sample_index >= len(labels):
        return "unavailable"
    value = np.asarray(labels[sample_index])
    if value.ndim == 0:
        scalar = value.item()
        if isinstance(scalar, float) and np.isfinite(scalar):
            return "{:g}".format(scalar)
        return str(scalar)
    return np.array2string(value, threshold=8, edgeitems=3)


def filename_safe_label(labels, sample_index):
    # type: (Optional[np.ndarray], int) -> str
    """Return a compact label that is safe to include in a filename."""
    label = format_label(labels, sample_index)
    if label is None:
        return "unknown"
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_.-")
    return (safe_label or "unknown")[:80]


def make_separate_output_path(output_dir, sample_index, channel_index, labels=None):
    # type: (Path, int, int, Optional[np.ndarray]) -> Path
    """Create a filename containing channel, sample, and label information."""
    label = filename_safe_label(labels, sample_index)
    filename = "channel_{:02d}_sample_{:04d}_label_{}.png".format(
        channel_index, sample_index, label
    )
    return output_dir / filename


def plot_separate_samples(
    data,
    sample_indices,
    channel_indices,
    time_start,
    sample_length,
    output_dir,
    labels=None,
    dpi=150,
):
    # type: (np.ndarray, Sequence[int], Sequence[int], int, int, Path, Optional[np.ndarray], int) -> List[Path]
    """Save each selected sample-channel pair as a clean, axis-free image."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError(
            "matplotlib is required to create the images. Install it with "
            "'python3 -m pip install matplotlib' or load an environment that provides it."
        )

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stop = time_start + sample_length
    saved_files = []

    for sample_index in sample_indices:
        for channel_index in channel_indices:
            figure, axis = plt.subplots(figsize=(4.0, 1.5))
            values = data[sample_index, channel_index, time_start:stop]
            axis.plot(values, linewidth=1.8, color="black")
            axis.margins(x=0.01, y=0.08)
            axis.set_axis_off()
            figure.subplots_adjust(left=0, right=1, bottom=0, top=1)

            image_path = make_separate_output_path(
                output_dir, sample_index, channel_index, labels
            )
            figure.savefig(
                str(image_path),
                dpi=dpi,
                facecolor="none",
                edgecolor="none",
                pad_inches=0,
                transparent=True,
            )
            plt.close(figure)
            saved_files.append(image_path)

    return saved_files


def plot_combined_samples(
    data,
    sample_indices,
    channel_indices,
    time_start,
    sample_length,
    output_file,
    labels=None,
    channel_names=None,
    sampling_rate=None,
    title=None,
    dpi=150,
):
    # type: (np.ndarray, Sequence[int], Sequence[int], int, int, Path, Optional[np.ndarray], Optional[Sequence[str]], Optional[float], Optional[str], int) -> None
    """Plot selected samples with labeled axes and save them in one image."""
    # Import lazily so data/layout inspection and --help work without matplotlib.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError(
            "matplotlib is required to create the image. Install it with "
            "'python3 -m pip install matplotlib' or load an environment that provides it."
        )

    row_count = len(sample_indices)
    figure_width = 12.0
    figure_height = max(3.0, 2.6 * row_count)
    figure, axes = plt.subplots(
        row_count, 1, figsize=(figure_width, figure_height), squeeze=False
    )

    stop = time_start + sample_length
    raw_x = np.arange(time_start, stop)
    if sampling_rate is None:
        x_values = raw_x
        x_label = "Time step"
    else:
        x_values = raw_x / sampling_rate
        x_label = "Time (seconds)"

    names = channel_names or [
        "channel {}".format(index) for index in range(data.shape[1])
    ]
    for row, sample_index in enumerate(sample_indices):
        axis = axes[row, 0]
        for channel_index in channel_indices:
            values = data[sample_index, channel_index, time_start:stop]
            axis.plot(
                x_values,
                values,
                linewidth=1.0,
                label=names[channel_index],
            )
        label_text = format_label(labels, sample_index)
        subplot_title = "Sample {}".format(sample_index)
        if label_text is not None:
            subplot_title += " (label: {})".format(label_text)
        axis.set_title(subplot_title, loc="left", fontsize=10)
        axis.set_ylabel("Value")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right", fontsize=7, ncol=min(4, len(channel_indices)))

    axes[-1, 0].set_xlabel(x_label)
    if title:
        figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.98) if title else None)

    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_file), dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def build_parser():
    # type: () -> argparse.ArgumentParser
    parser = argparse.ArgumentParser(
        description=(
            "Read processed time-series data from a NumPy file and save selected "
            "samples as line-plot images. By default, every sample-channel pair "
            "is saved separately without axes or a grid."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="A data .npy file, subject directory, or processed dataset root",
    )
    parser.add_argument(
        "--subject",
        help="Subject directory name when data_path is a dataset root (for example 01 or 101)",
    )
    parser.add_argument(
        "--data-file",
        default="0.npy",
        help="Data filename to use when data_path is a directory",
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        help="Optional label .npy file; by default a matching label file is detected",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=positive_int,
        default=5,
        help="Number of consecutive samples/signals to visualize",
    )
    parser.add_argument(
        "--start-sample",
        type=nonnegative_int,
        default=0,
        help="Index of the first consecutive sample",
    )
    parser.add_argument(
        "--sample-indices",
        help="Specific sample indices/ranges, e.g. 0,4,10-12 (overrides sample count/start)",
    )
    parser.add_argument(
        "-l",
        "--sample-length",
        type=positive_int,
        help="Number of time steps shown from every sample (default: all remaining steps)",
    )
    parser.add_argument(
        "--time-start",
        type=nonnegative_int,
        default=0,
        help="First time step shown within every sample",
    )
    parser.add_argument(
        "--channels",
        help="Channel indices/ranges to plot, e.g. 0-2,5 (default: all or --num-channels)",
    )
    parser.add_argument(
        "--num-channels",
        type=positive_int,
        help="Plot only the first N channels; ignored when --channels is used",
    )
    parser.add_argument(
        "--channel-names",
        help="Comma-separated names for every channel in the input array",
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "NCT", "NTC", "CT", "TC"),
        default="auto",
        help="Input dimension order: N=samples, C=channels, T=time",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        help="Samples per second; when supplied the x-axis is shown in seconds",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("signal_visualizations"),
        help=(
            "Directory in which generated images are saved; separate-image "
            "filenames contain channel, sample, and label"
        ),
    )
    parser.add_argument(
        "--output-mode",
        choices=("separate", "combined"),
        default="separate",
        help=(
            "Save every sample-channel pair as an axis-free image, or retain "
            "the original combined diagnostic plot"
        ),
    )
    parser.add_argument("--title", help="Optional title at the top of the image")
    parser.add_argument("--dpi", type=positive_int, default=150, help="Output resolution")
    return parser


def main():
    # type: () -> int
    parser = build_parser()
    args = parser.parse_args()

    if args.sampling_rate is not None and args.sampling_rate <= 0:
        parser.error("--sampling-rate must be greater than zero")

    try:
        data_file = resolve_data_file(args.data_path, args.subject, args.data_file)
        label_file = find_label_file(data_file, args.label_file)
        original = np.load(str(data_file), mmap_mode="r")
        data = to_samples_channels_time(original, args.layout)
        labels = np.load(str(label_file), mmap_mode="r") if label_file else None

        sample_count, channel_count, time_count = data.shape
        if args.sample_indices:
            sample_indices = parse_indices(
                args.sample_indices, sample_count, "--sample-indices"
            )
        else:
            sample_stop = min(args.start_sample + args.num_samples, sample_count)
            sample_indices = list(range(args.start_sample, sample_stop))
            if not sample_indices:
                raise ValueError(
                    "--start-sample {} is outside the dataset ({} samples)".format(
                        args.start_sample, sample_count
                    )
                )

        if args.channels:
            channel_indices = parse_indices(args.channels, channel_count, "--channels")
        else:
            plotted_channels = args.num_channels or channel_count
            if plotted_channels > channel_count:
                raise ValueError(
                    "--num-channels {} exceeds the available {} channels".format(
                        plotted_channels, channel_count
                    )
                )
            channel_indices = list(range(plotted_channels))

        if args.time_start >= time_count:
            raise ValueError(
                "--time-start {} is outside each sample ({} time steps)".format(
                    args.time_start, time_count
                )
            )
        available_length = time_count - args.time_start
        sample_length = args.sample_length or available_length
        if sample_length > available_length:
            print(
                "Requested {} time steps, but only {} remain; clipping to {}.".format(
                    sample_length, available_length, available_length
                )
            )
            sample_length = available_length

        channel_names = parse_channel_names(args.channel_names, channel_count)
        print("Data file: {}".format(data_file))
        print(
            "Original shape: {}; interpreted as (samples, channels, time) = {}".format(
                original.shape, data.shape
            )
        )
        print("Samples plotted: {}".format(sample_indices))
        print("Channels plotted: {}".format(channel_indices))

        if args.output_mode == "separate":
            saved_files = plot_separate_samples(
                data=data,
                sample_indices=sample_indices,
                channel_indices=channel_indices,
                time_start=args.time_start,
                sample_length=sample_length,
                output_dir=args.output,
                labels=labels,
                dpi=args.dpi,
            )
            print(
                "Saved {} clean images in: {}".format(
                    len(saved_files), saved_files[0].parent
                )
            )
            print("First image: {}".format(saved_files[0]))
        else:
            combined_output = args.output.expanduser().resolve() / "combined.png"
            combined_output.parent.mkdir(parents=True, exist_ok=True)
            plot_combined_samples(
                data=data,
                sample_indices=sample_indices,
                channel_indices=channel_indices,
                time_start=args.time_start,
                sample_length=sample_length,
                output_file=combined_output,
                labels=labels,
                channel_names=channel_names,
                sampling_rate=args.sampling_rate,
                title=args.title,
                dpi=args.dpi,
            )
            print("Saved image: {}".format(combined_output))
        if label_file:
            print("Labels loaded from: {}".format(label_file))
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
