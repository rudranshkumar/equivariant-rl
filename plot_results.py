import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 22,          # default text size
    "axes.titlesize": 26,     # title
    "axes.labelsize": 24,     # x/y labels
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.titlesize": 28,
})

def load_and_align_csvs(csv_paths):
    """Load multiple CSV files and align them on a shared step grid."""
    dfs = [pd.read_csv(p) for p in csv_paths]
    # normalize column names just in case
    for df in dfs:
        df.columns = [c.strip().lower() for c in df.columns]

    # expect 'step' and 'value'
    all_steps = sorted(set().union(*[df["step"].tolist() for df in dfs]))

    aligned = []
    for df in dfs:
        # interpolate this run's values onto the union of all steps
        aligned_values = np.interp(all_steps, df["step"], df["value"])
        aligned.append(aligned_values)

    arr = np.vstack(aligned)          # shape: [n_runs, n_steps]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return np.array(all_steps), mean, std

def smooth_moving_average(y, window):
    if window is None or window <= 1:
        return y
    window = min(window, len(y))
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)

    half = window // 2
    n = len(y)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        out[i] = y[start:end].mean()
    return out


def plot_experiments(
    experiment_folders,
    title="Training curves",
    output_file="plot.png",
    smoothing_window=1000,
):
    """
    experiment_folders: list of folder names, each containing multiple CSV runs.
                        One line + shaded band per folder.
    """
    plt.figure(figsize=(10, 6))

    for folder in experiment_folders:
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        if not csv_files:
            print(f"[WARN] No CSV files found in {folder}, skipping.")
            continue

        print(f"[INFO] Loading {len(csv_files)} runs from {folder}")
        steps, mean, std = load_and_align_csvs(csv_files)

        # smooth after averaging
        mean_s = smooth_moving_average(mean, smoothing_window)
        std_s = smooth_moving_average(std, smoothing_window)

        label = os.path.basename(folder.rstrip("/"))
        plt.plot(steps, mean_s, linewidth=2, label=label)
        plt.fill_between(steps, mean_s - std_s, mean_s + std_s, alpha=0.15)

    plt.xlabel("Environment Steps")
    plt.ylabel("Reward/Episode")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()
    print(f"[INFO] Saved figure to {output_file}")


if __name__ == "__main__":
    # === EXAMPLES ===
    # PPO figure
    ppo_folders = [
        "SAC",
        "D4 Equivariant SAC",
    ]
    plot_experiments(
        experiment_folders=ppo_folders,
        title="Mujoco Push Block",
        output_file="Fetch.png",
        smoothing_window=250,  # tune this as you like
    )

