import json, uuid, random, math

EXPERIMENT_CONFIGS = {
    "pick":  dict(env_id="FetchPickAndPlaceDense-v4", policy_lr=6e-4, q_lr=5e-4, gamma=0.99,  tau=0.005, batch_size=256, dihedral_N=16, reg_rep_N=64),
    "slide": dict(env_id="FetchSlideDense-v4",       policy_lr=1e-3, q_lr=2e-3, gamma=0.97,  tau=0.002, batch_size=256, dihedral_N=16, reg_rep_N=128),
}

# --- sweep size ---
N_PER_TASK = 40          # 60 pick + 60 slide = 120 trials (nice for a 0-119 array)

# --- jitter scales (tuneable) ---
SIGMA_LR    = 0.60       # broader than before
SIGMA_TAU   = 0.60
SIGMA_GAMMA = 0.015      # broader gamma search for harder tasks

def clip(x, lo, hi): return max(lo, min(hi, x))

def log_jitter(x, sigma, lo, hi):
    y = x * math.exp(random.gauss(0.0, sigma))
    return clip(y, lo, hi)

def jitter_gamma(g):
    return clip(g + random.gauss(0.0, SIGMA_GAMMA), 0.94, 0.999)

def main():
    batch = []
    for exp, base in EXPERIMENT_CONFIGS.items():
        for _ in range(N_PER_TASK):
            # Wider search for pick/slide: include capacity + symmetry as knobs
            dihedral_N = random.choice([8, 16]) if exp == "slide" else random.choice([4, 8, 16])
            reg_rep_N  = random.choice([64, 128]) if exp == "pick" else random.choice([128, 192])

            params = {
                "policy_lr":  log_jitter(base["policy_lr"], SIGMA_LR, 1e-5, 3e-3),
                "q_lr":       log_jitter(base["q_lr"],      SIGMA_LR, 1e-5, 3e-3),
                "tau":        log_jitter(base["tau"],       SIGMA_TAU, 1e-4, 2e-2),
                "gamma":      jitter_gamma(base["gamma"]),
                "batch_size": random.choice([128, 256, 512]),
                "dihedral_N": dihedral_N,
                "reg_rep_N":  reg_rep_N,
            }

            batch.append({
                "trial_uid": str(uuid.uuid4()),
                "exp": exp,
                "env_id": base["env_id"],
                "params": params,
                "seed": random.choice([1, 2]),  # keep 2 seeds for stability signal
            })

    out = "pick_slide_sweep.jsonl"
    with open(out, "w") as f:
        for row in batch:
            f.write(json.dumps(row) + "\n")

    print("Wrote", out, "with", len(batch), "trials")

if __name__ == "__main__":
    main()

