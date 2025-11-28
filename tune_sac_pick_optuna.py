import optuna
import torch
from sac_continuous_action import Args, train_and_eval   # <-- adjust filename if needed


def objective(trial: optuna.Trial) -> float:
    # ---- Hyperparameter search space ----
    policy_lr = trial.suggest_float("policy_lr", 1e-5, 3e-3, log=True)
    q_lr      = trial.suggest_float("q_lr", 1e-5, 3e-3, log=True)
    gamma     = trial.suggest_float("gamma", 0.95, 0.999)
    tau       = trial.suggest_float("tau", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])


    # ---- Build Args object ----
    args = Args(
        env_id="FetchPickAndPlaceDense-v4",
        total_timesteps=1_000_000,   # shorter for tuning
        policy_lr=policy_lr,
        q_lr=q_lr,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,

        # IMPORTANT: disable slow logging
        track=False,
        capture_video=False,
    )

    # ---- Run training ----
    score = train_and_eval(args, trial=trial)

    return score   # Optuna maximizes this


def main():
    STUDY_NAME = "sac_fetch_pick_dense"
    STORAGE_URL = "sqlite:///sac_fetch_pick_dense.db"

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        load_if_exists=True,        # <-- KEY: resume if DB exists
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=8,
        ),
    )

    study.optimize(
        objective,
        n_trials=30,                # runs *new* trials only
    )

    print("✅ Best value:", study.best_value)
    print("✅ Best params:", study.best_params)


if __name__ == "__main__":
    main()
