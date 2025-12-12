import argparse
import torch
from sac_continuous_action import Args, train_and_eval

# ---------------------------------------------------------
# Recommended hyperparameters for each task
# ---------------------------------------------------------

EXPERIMENT_CONFIGS = {
    "slide": dict(
        env_id="FetchSlideDense-v4",
        policy_lr=1e-3,
        q_lr=2e-3,
        gamma=0.97,
        tau=0.002,
        batch_size=256,
    ),
    "pick": dict(
        env_id="FetchPickAndPlaceDense-v4",
        policy_lr=6e-4,
        q_lr=1.2e-3,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
    ),
    "push": dict(
        env_id="FetchPushDense-v4",
        policy_lr=4e-4,
        q_lr=1e-3,
        gamma=0.97,
        tau=0.005,
        batch_size=256,
    ),
}


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        choices=["slide", "pick", "push"],
        help="Which experiment to run: slide | pick | push",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1_000_000,
        help="Total timesteps (default: 1e6)"
    )

    args_cli = parser.parse_args()

    # Load experiment-specific hyperparameters
    cfg = EXPERIMENT_CONFIGS[args_cli.exp]

    # Build Args for SAC
    args = Args(
        env_id=cfg["env_id"],
        total_timesteps=args_cli.steps,
        policy_lr=cfg["policy_lr"],
        q_lr=cfg["q_lr"],
        gamma=cfg["gamma"],
        tau=cfg["tau"],
        batch_size=cfg["batch_size"],
        track=False,
        capture_video=False,
    )

    # ---- Run training ----
    final_score = train_and_eval(args, trial=None)
    print(f"[{args_cli.exp}] Completed run with final score: {final_score}")


if __name__ == "__main__":
    main()
