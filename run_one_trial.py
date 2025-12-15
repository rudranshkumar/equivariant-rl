import json, os, time, argparse, socket
from sac_continuous_action_equivariant import Args, train_and_eval

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--steps", type=int, default=1_000_000)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # read the indexed row
    item = None
    with open(args.batch, "r") as f:
        for i, line in enumerate(f):
            if i == args.index:
                item = json.loads(line)
                break
    if item is None:
        raise SystemExit(f"Index {args.index} out of range")

    trial_uid = item["trial_uid"]
    params = item["params"]
    env_id = item["env_id"]
    seed = int(item.get("seed", 1))

    a = Args(
        env_id=env_id,
        total_timesteps=args.steps,
        seed=seed,
        track=False,
        capture_video=False,
        eval_n_episodes=25,
        eval_interval=100_000,
        **params,
    )

    start = time.time()
    status, score, err = "ok", None, None
    try:
        score = float(train_and_eval(a, trial=None))
    except Exception as e:
        status, err = "error", repr(e)

    result = {
        "trial_uid": trial_uid,
        "exp": item.get("exp"),
        "env_id": env_id,
        "seed": seed,
        "params": params,
        "score": score,
        "status": status,
        "error": err,
        "hostname": socket.gethostname(),
        "time_sec": time.time() - start,
    }

    tmp = os.path.join(args.outdir, f".{trial_uid}.tmp")
    final = os.path.join(args.outdir, f"{trial_uid}.json")
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, final)

if __name__ == "__main__":
    main()

