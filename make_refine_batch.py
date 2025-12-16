#!/usr/bin/env python3
import json, uuid, argparse
from pathlib import Path

WINNER_PARAMS = {
    "policy_lr": 4e-4,
    "q_lr": 1e-3,
    "tau": 0.005179,
    "gamma": 0.970006,
    "batch_size": 256,
    "dihedral_N": 16,
    "reg_rep_N": 64,
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="push_final_batch.jsonl")
    p.add_argument("--n_seeds", type=int, default=4)
    p.add_argument("--seed0", type=int, default=1)
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--env_id", default="FetchPushDense-v4")
    p.add_argument("--exp", default="push_final")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for k in range(args.n_seeds):
            seed = args.seed0 + k
            item = {
                "trial_uid": str(uuid.uuid4()),
                "exp": args.exp,
                "env_id": args.env_id,
                "params": WINNER_PARAMS,
                "seed": seed,
                "steps": args.steps,
            }
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {args.n_seeds} lines -> {out}")

if __name__ == "__main__":
    main()

