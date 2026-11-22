# SPDX-License-Identifier: AGPL-3.0-only
# See ../LICENSE for the full license terms.

"""Load finite laboratory ball velocities or paired particle/container samples."""

import numpy as np
import pandas as pd


def load_lab_components(csv_file, factor):
    """Return all finite laboratory-frame ball velocities, independent of container data."""
    data = pd.read_csv(csv_file)
    required = {"obj_id", "vx", "vy"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")
    balls = data.loc[data["obj_id"] != "container", ["vx", "vy"]]
    laboratory = balls.to_numpy(dtype=float) * factor
    # Keep zero speeds and high-speed tails; only nonfinite components are excluded.
    valid = np.isfinite(laboratory).all(axis=1)
    print(f"Valid laboratory particle samples: {int(valid.sum())} / {len(balls)}")
    return laboratory[valid]


def load_paired_components(csv_file, factor):
    data = pd.read_csv(csv_file)
    required = {"frame", "obj_id", "vx", "vy"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")

    container = data.loc[data["obj_id"] == "container", ["frame", "vx", "vy"]]
    if container["frame"].duplicated().any():
        raise ValueError("More than one container velocity for the same frame")
    container = container.rename(columns={"vx": "container_vx", "vy": "container_vy"})
    balls = data.loc[data["obj_id"] != "container", ["frame", "obj_id", "vx", "vy"]]
    paired = balls.merge(container, on="frame", how="inner", validate="many_to_one")

    camera = paired[["vx", "vy"]].to_numpy(dtype=float) * factor
    relative = camera - paired[["container_vx", "container_vy"]].to_numpy(dtype=float) * factor
    # Keep identical finite samples in both frames, including the high-speed tail.
    valid = np.isfinite(camera).all(axis=1) & np.isfinite(relative).all(axis=1)

    print(f"Paired particle samples: {int(valid.sum())} / {len(balls)} "
          f"({len(paired)} with container velocity)")
    return camera[valid], relative[valid]
