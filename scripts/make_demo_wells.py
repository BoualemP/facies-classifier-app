"""Create the two synthetic demonstration wells shipped with the application.

The contest dataset is not openly licensed, so the application ships simulated
wells instead of real ones. They are built only from summary statistics of the
training wells (facies transition frequencies, and the mean and covariance of
the logs in each facies):

1. a facies sequence is drawn from the facies-to-facies transition matrix;
2. for each sample, the logs are drawn from the facies' multivariate normal;
3. the logs are smoothed along depth, as real logs are;
4. NM_M and RELPOS are derived from the sequence.

Usage: python scripts/make_demo_wells.py path/to/facies_vectors.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOGS = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]
TRAIN_WELLS = ["CHURCHMAN BIBLE", "CROSS H CATTLE", "LUKE G U", "NOLAN", "SHRIMPLIN"]
OUT = Path(__file__).resolve().parent.parent / "app" / "demo"


def statistics(data):
    wells = data[data["Well Name"].isin(TRAIN_WELLS)].sort_values(["Well Name", "Depth"])
    transitions = np.full((9, 9), 0.5)  # small prior so every transition is possible
    for _, well in wells.groupby("Well Name"):
        codes = well["Facies"].to_numpy() - 1
        for a, b in zip(codes[:-1], codes[1:]):
            transitions[a, b] += 1
    transitions /= transitions.sum(axis=1, keepdims=True)
    means = wells.groupby("Facies")[LOGS].mean().to_numpy()
    covs = [np.cov(wells.loc[wells["Facies"] == k, LOGS].to_numpy(), rowvar=False) for k in range(1, 10)]
    return transitions, means, covs


def simulate(name, samples, top, seed, transitions, means, covs):
    rng = np.random.default_rng(seed)
    codes = [rng.integers(0, 9)]
    for _ in range(samples - 1):
        codes.append(rng.choice(9, p=transitions[codes[-1]]))
    codes = np.array(codes)
    logs = np.array([rng.multivariate_normal(means[c], covs[c] * 0.5) for c in codes])
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    smooth = np.column_stack([np.convolve(np.pad(logs[:, j], 2, mode="edge"), kernel, mode="valid")
                              for j in range(len(LOGS))])
    well = pd.DataFrame(smooth, columns=LOGS)
    well["PE"] = well["PE"].clip(lower=0.5)
    well.insert(0, "Depth", top + 0.5 * np.arange(samples))
    well.insert(0, "Well Name", name)
    well["NM_M"] = np.where(codes < 3, 1, 2)
    # RELPOS: 1 at the top of each interval of constant NM_M, 0 at its base
    interval = np.cumsum(np.r_[0, np.diff(well["NM_M"]) != 0])
    position = well.groupby(interval).cumcount()
    size = well.groupby(interval)["Depth"].transform("size")
    well["RELPOS"] = np.where(size > 1, 1 - position / (size - 1).clip(lower=1), 1.0)
    well["Facies"] = codes + 1
    return well.round({"GR": 2, "ILD_log10": 3, "DeltaPHI": 2, "PHIND": 2, "PE": 2, "RELPOS": 3})


if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1])
    stats = statistics(data)
    OUT.mkdir(exist_ok=True)
    for path in OUT.glob("*.csv"):
        path.unlink()
    for name, samples, top, seed in [("SYNTHETIC A", 460, 2800.0, 11), ("SYNTHETIC B", 440, 2900.0, 23)]:
        well = simulate(name, samples, top, seed, *stats)
        well.to_csv(OUT / f"{name.lower().replace(' ', '_')}.csv", index=False)
        print(name, len(well), "samples")
