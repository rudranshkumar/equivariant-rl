import argparse
from sac_continuous_action import Args, train_and_eval


# ---------------------------------------------------------
# Non-equivariant SAC baselines for Fetch tasks
# ---------------------------------------------------------

BASELINE_CONFIGS = {
    "push": dict(
        env_id="FetchPushDense-v4",
        policy_lr=5e-4,
        q_lr=3e-4,
        gamma=0.98,
        tau=0.005,
        batch_size=128,
    ),
    "pick": dict(
        env_id="FetchPickAndPlaceDense-v4",
        policy_lr=3e-4,
        q_lr=5e-4,
        gamma=0.98,
        tau=0.001,
        batch_size=256,
    ),
    "slide": dict(
        env_id="FetchSlideDense-v4",
        policy_lr=1e-4,
        q_lr=6e-4,
        gamma=0.97,
        tau=0.02,
        batch_size=256,
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        choices=["push", "pick", "slide"],
        help="Which baseline to run: push | pick | slide",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1_000_000,
        help="Total timesteps (default: 1e6)",
    )
    args_cli = parser.parse_args()

    cfg = BASELINE_CONFIGS[args_cli.exp]

    # ---- Build Args object ----
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
    score = train_and_eval(args, trial=None)

    print(f"[{args_cli.exp}] Completed baseline run with final score: {score}")
    return score   # Optuna can maximize this if you call main() inside an objective


if __name__ == "__main__":
    main()

