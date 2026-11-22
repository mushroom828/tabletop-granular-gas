"""Quality-gated container velocity from short position windows.

Each four-position fit is centered on the middle two positions.  Its result
is stored at the later middle frame, where a ball's two-frame velocity uses
exactly that middle interval.  The CSV timestamp remains the video timestamp
of the stored frame; it is not the effective midpoint of the velocity sample.
"""

import numpy as np


FIT_POINTS = 4
MAX_INTERVAL_FRAMES = 1.5
MAX_JUMP_RATIO = 4.0
ACCEL_RESIDUAL_RATIO = 6.0


def fit_container_velocities(positions):
    """Return ``[(velocity_px_s, ball_end_frame), ...]`` and quality counts.

    ``positions`` contains only valid tracked container centers as
    ``(frame, timestamp_seconds, center_xy)``.  Missing frames stay missing:
    no speed window may cross a gap, a reacquisition, or an isolated glitch.
    """
    diagnostics = {"positions": len(positions), "isolated_outliers": 0,
                   "discontinuous_steps": 0, "misaligned_windows": 0,
                   "velocity_samples": 0}
    if len(positions) < FIT_POINTS:
        return [], diagnostics

    frames = np.asarray([item[0] for item in positions], dtype=np.int64)
    times = np.asarray([item[1] for item in positions], dtype=float)
    centers = np.asarray([item[2] for item in positions], dtype=float)
    if (centers.shape != (len(positions), 2)
            or not np.isfinite(times).all() or not np.isfinite(centers).all()
            or np.any(np.diff(frames) <= 0)):
        raise ValueError("Container positions must be finite and in increasing frame order")

    intervals = np.diff(times)
    consecutive_frames = np.diff(frames) == 1
    ordinary_periods = intervals[consecutive_frames & (intervals > 0)]
    if not ordinary_periods.size:
        return [], diagnostics
    frame_period = float(np.median(ordinary_periods))
    adjacent = (consecutive_frames & (intervals > 0)
                & (intervals <= MAX_INTERVAL_FRAMES * frame_period))
    jumps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    ordinary_jumps = jumps[adjacent]
    median_jump = float(np.median(ordinary_jumps)) if ordinary_jumps.size else 0.0
    jump_limit = max(2.0, MAX_JUMP_RATIO * median_jump)

    # Reject a one-frame excursion only when the following position returns
    # close to the preceding trajectory.  Other large jumps split the run.
    residual = np.zeros(len(positions), dtype=float)
    triples = np.zeros(len(positions), dtype=bool)
    triples[1:-1] = adjacent[:-1] & adjacent[1:]
    neighbor_spans = intervals[:-1] + intervals[1:]
    fraction = np.divide(intervals[:-1], neighbor_spans,
                         out=np.zeros_like(neighbor_spans), where=neighbor_spans > 0)
    expected = centers[:-2] + fraction[:, None] * (centers[2:] - centers[:-2])
    residual[1:-1] = np.where(
        triples[1:-1], np.linalg.norm(centers[1:-1] - expected, axis=1), 0.0)
    median_residual = float(np.median(residual[triples])) if triples.any() else 0.0
    residual_limit = max(2.0, ACCEL_RESIDUAL_RATIO * median_residual)
    outlier = np.zeros(len(positions), dtype=bool)
    neighborhood = np.maximum(residual[:-2], residual[2:])
    outlier[1:-1] = (triples[1:-1]
                     & (residual[1:-1] > residual_limit)
                     & (residual[1:-1] > 1.4 * neighborhood))
    diagnostics["isolated_outliers"] = int(outlier.sum())

    valid_indices = np.flatnonzero(~outlier)
    if len(valid_indices) < FIT_POINTS:
        return [], diagnostics
    previous, current = valid_indices[:-1], valid_indices[1:]
    breaks = (current != previous + 1) | ~adjacent[previous] | (jumps[previous] > jump_limit)
    diagnostics["discontinuous_steps"] = int(breaks.sum())
    run_starts = np.r_[0, np.flatnonzero(breaks) + 1]
    run_ends = np.r_[run_starts[1:], len(valid_indices)]

    velocity_samples = []
    for start, end in zip(run_starts, run_ends):
        run = valid_indices[start:end]
        if len(run) < FIT_POINTS:
            continue
        windows = run[np.arange(len(run) - FIT_POINTS + 1)[:, None]
                      + np.arange(FIT_POINTS)[None, :]]
        window_times = times[windows]
        centered_times = window_times - window_times.mean(axis=1, keepdims=True)
        denominator = np.sum(centered_times ** 2, axis=1)
        slopes = (np.sum(centered_times[:, :, None] * centers[windows], axis=1)
                  / denominator[:, None])
        # For VFR, retain only windows whose fit center is close to the ball
        # interval midpoint between the two middle positions.
        midpoint = (window_times[:, 1] + window_times[:, 2]) / 2.0
        aligned = np.abs(window_times.mean(axis=1) - midpoint) <= 0.25 * frame_period
        diagnostics["misaligned_windows"] += int((~aligned).sum())
        for frame, slope, is_aligned in zip(frames[windows[:, 2]], slopes, aligned):
            if is_aligned and np.isfinite(slope).all():
                velocity_samples.append((slope, int(frame)))

    diagnostics["velocity_samples"] = len(velocity_samples)
    return velocity_samples, diagnostics
