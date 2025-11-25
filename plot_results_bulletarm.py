import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def smooth_moving_average(y, window):
    """Centered moving average without zero-padding at the edges."""
    if window is None or window <= 1:
        return y
    y = np.asarray(y, dtype=float)
    n = len(y)
    window = min(window, n)

    out = np.empty_like(y)
    half = window // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        out[i] = y[start:end].mean()
    return out

def load_policy_runs(outputs_root, policy_name, task_pattern):
    """
    For a given policy (e.g., 'curl_sac_cnn') and task substring,
    find all runs and load their eval_rewards arrays.
    """
    policy_dir = os.path.join(outputs_root, policy_name)
    if not os.path.isdir(policy_dir):
        return []

    eval_arrays = []
    run_dirs = sorted(
        d for d in os.listdir(policy_dir)
        if task_pattern in d and os.path.isdir(os.path.join(policy_dir, d))
    )

    for run in run_dirs:
        base = os.path.join(policy_dir, run)
        # Prefer result/ if it exists, otherwise info/
        result_path = os.path.join(base, "result", "eval_rewards.npy")
        info_path = os.path.join(base, "info", "eval_rewards.npy")

        if os.path.isfile(result_path):
            path = result_path
        elif os.path.isfile(info_path):
            path = info_path
        else:
            print(f"[WARN] No eval_rewards.npy found in {base}, skipping.")
            continue

        arr = np.load(path)
        eval_arrays.append(arr)
        print(f"[INFO] Loaded {path} with shape {arr.shape}")

    return eval_arrays


def aggregate_runs(eval_arrays):
    """
    Stack runs that have the same length and compute mean & std.
    We keep only runs with the most common length to avoid partial runs.
    """
    if not eval_arrays:
        return None, None

    lengths = [len(a) for a in eval_arrays]
    # most common length
    unique, counts = np.unique(lengths, return_counts=True)
    L = unique[np.argmax(counts)]

    filtered = [a for a in eval_arrays if len(a) == L]
    dropped = len(eval_arrays) - len(filtered)
    if dropped > 0:
        print(f"[WARN] Dropping {dropped} run(s) with non-dominant length.")

    arr = np.vstack(filtered)  # shape: [n_runs, L]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-root", type=str, default="outputs",
        help="Root directory where policy folders live."
    )
    parser.add_argument(
        "--task", type=str, required=True,
        help="Substring to match in run folder names, "
             "e.g. 'close_loop_drawer_opening'."
    )
    parser.add_argument(
        "--eval-freq", type=int, default=500,
        help="Number of training steps between evaluations."
    )
    parser.add_argument(
        "--smooth-window", type=int, default=1,
        help="Moving-average window size for smoothing (in eval indices). "
             "Use 1 or 0 to disable."
    )
    parser.add_argument(
        "--output-file", type=str, default="eval_plot.png",
        help="Filename for the saved plot."
    )
    parser.add_argument(
        "--policies", nargs="*", default=None,
        help="Optional list of policy subdirectories under outputs/. "
             "If omitted, use all first-level dirs inside outputs/."
    )
    args = parser.parse_args()

    # Big, paper-friendly fonts
    plt.rcParams.update({
        "font.size": 22,
        "axes.titlesize": 26,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "figure.titlesize": 28,
    })

    # Determine which policies to use
    if args.policies is None:
        args.policies = sorted(
            d for d in os.listdir(args.outputs_root)
            if os.path.isdir(os.path.join(args.outputs_root, d))
        )
        print("[INFO] Auto-detected policies:", ", ".join(args.policies))
    else:
        print("[INFO] Using specified policies:", ", ".join(args.policies))

    plt.figure(figsize=(12, 8))

    any_plotted = False

    for policy in args.policies:
        eval_arrays = load_policy_runs(args.outputs_root, policy, args.task)
        if not eval_arrays:
            print(f"[INFO] No runs found for policy '{policy}' and task '{args.task}'.")
            continue

        mean, std = aggregate_runs(eval_arrays)
        if mean is None:
            continue

        # x-axis: training steps
        steps = np.arange(len(mean)) * args.eval_freq

        # smooth after averaging
        mean_s = smooth_moving_average(mean, args.smooth_window)
        std_s = smooth_moving_average(std, args.smooth_window)

        label = policy
        plt.plot(steps, mean_s, linewidth=2, label=label)
        plt.fill_between(steps, mean_s - std_s, mean_s + std_s, alpha=0.15)

        any_plotted = True

    if not any_plotted:
        print("[ERROR] No data plotted. Check task name / policy names.")
        return

    plt.xlabel("Training Steps")
    plt.ylabel("Evaluation Reward")
    plt.title("Household Picking")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Saved figure to {args.output_file}")


if __name__ == "__main__":
    main()

