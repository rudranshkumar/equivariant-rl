import json, uuid, random, math

EXPERIMENT_CONFIGS = {
    "slide": dict(env_id="FetchSlideDense-v4", policy_lr=1e-3, q_lr=2e-3, gamma=0.97,  tau=0.002, batch_size=256, dihedral_N=16, reg_rep_N=128),
    "pick":  dict(env_id="FetchPickAndPlaceDense-v4", policy_lr=6e-4, q_lr=5e-4, gamma=0.99, tau=0.005, batch_size=256, dihedral_N=16, reg_rep_N=64),
    "push":  dict(env_id="FetchPushDense-v4", policy_lr=4e-4, q_lr=1e-3, gamma=0.957, tau=0.005, batch_size=256, dihedral_N=16, reg_rep_N=64),
}

# how tight?
N_PER_TASK = 30
SIGMA_LR   = 0.45   # log-jitter: ~x/÷1.6 typical, rarely > x/÷3
SIGMA_TAU  = 0.45
SIGMA_GAMMA = 0.008

def clip(x, lo, hi): return max(lo, min(hi, x))

def log_jitter(x, sigma, lo, hi):
    y = x * math.exp(random.gauss(0.0, sigma))
    return clip(y, lo, hi)

def jitter_gamma(g):
    return clip(g + random.gauss(0.0, SIGMA_GAMMA), 0.94, 0.999)

batch = []
for exp, base in EXPERIMENT_CONFIGS.items():
    for _ in range(N_PER_TASK):
        params = {
            "policy_lr": log_jitter(base["policy_lr"], SIGMA_LR, 1e-5, 3e-3),
            "q_lr":      log_jitter(base["q_lr"],      SIGMA_LR, 1e-5, 3e-3),
            "tau":       log_jitter(base["tau"],       SIGMA_TAU, 1e-4, 2e-2),
            "gamma":     jitter_gamma(base["gamma"]),
            "batch_size": random.choice([128, 256]),            # keep tight
            "dihedral_N": base["dihedral_N"],                   # fixed
            "reg_rep_N":  base["reg_rep_N"],                    # fixed (or random.choice([64,128]))
        }

        batch.append({
            "trial_uid": str(uuid.uuid4()),
            "exp": exp,
            "env_id": base["env_id"],
            "params": params,
            "seed": random.choice([1, 2]),  # optional: two-seed stability
        })

with open("refine_batch.jsonl", "w") as f:
    for row in batch:
        f.write(json.dumps(row) + "\n")

print("Wrote refine_batch.jsonl with", len(batch), "trials")

