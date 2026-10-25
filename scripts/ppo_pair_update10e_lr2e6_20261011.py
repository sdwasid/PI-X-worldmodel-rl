from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import torch
from stable_baselines3 import PPO


ROOT = Path("outputs/rl_shadow")

T1 = ROOT / "PPO_BATCH_B375_SCALE050_T1_20261009_181257"
T2 = ROOT / "PPO_BATCH_B375_SCALE050_T2_20261009_182638"

# 之前那次重复 T2，不进入训练
OLD_T2 = ROOT / "PPO_BATCH_B375_SCALE050_T2_20261009_182029"

MODEL_BASE = Path(
    "outputs/residual_ppo_obs2054_safe_init_20261009"
)

MODEL_ZIP = Path(str(MODEL_BASE) + ".zip")

OUT_MODEL = Path(
    "outputs/residual_ppo_obs2054_pair_update10e_lr2e6_20261011"
)

METRICS = Path(
    "outputs/rl_training/"
    "ppo_pair_update10e_lr2e6_metrics_20261011.json"
)

GAMMA = 0.99
GAE_LAMBDA = 0.95

# 极小更新
LEARNING_RATE = 2e-6
EPOCHS = 10
CLIP_RANGE = 0.10
VF_COEF = 0.25
MAX_GRAD_NORM = 0.5
MAX_APPROX_KL = 0.002


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def update_result(path, updates):
    p = path / "result.txt"

    if not p.exists():
        raise RuntimeError(f"Missing result: {p}")

    data = {}
    order = []

    for line in p.read_text().splitlines():
        if "=" not in line:
            continue

        k, v = line.split("=", 1)

        if k not in data:
            order.append(k)

        data[k] = v

    for k, v in updates.items():
        if k not in data:
            order.append(k)

        data[k] = str(v)

    p.write_text(
        "\n".join(
            f"{k}={data[k]}"
            for k in order
        )
        + "\n"
    )


# ============================================================
# 1. Human labels
# ============================================================

update_result(
    T1,
    {
        "human_result": "SUCCESS",
        "motion_status": "NORMAL",
        "grasp_quality": "MARGINAL",
        "training_use": "YES",
        "training_label_source": "HUMAN",
        "training_terminal_reward": "1",
    },
)

update_result(
    T2,
    {
        "human_result": "FAILURE",
        "motion_status": "NORMAL",
        "grasp_quality": "FAIL",
        "failure_reason":
            "GRIPPER_COLLISION_TIPPED_TISSUE",
        "training_use": "YES",
        "training_label_source": "HUMAN",
        "training_terminal_reward": "-1",
        "reward_detector_disagreement": "TRUE",
    },
)

if OLD_T2.exists():
    update_result(
        OLD_T2,
        {
            "training_use": "NO",
            "training_exclusion_reason":
                "DUPLICATE_T2_WITHOUT_CONFIRMED_HUMAN_LABEL",
        },
    )


# ============================================================
# 2. Safety gate
# ============================================================

control_text = Path(
    "src/lerobot/rollout/residual_control.py"
).read_text()

if "ACTIVE_SCALE = 0.0" not in control_text:
    raise RuntimeError(
        "STOP: ACTIVE_SCALE is not 0.0"
    )

if not MODEL_ZIP.exists():
    raise RuntimeError(
        f"Missing original model: {MODEL_ZIP}"
    )

original_hash_before = sha256(MODEL_ZIP)


# ============================================================
# 3. Load one real episode
# ============================================================

def load_episode(path, terminal_reward):

    obs_path = path / "shadow_macro_obs2054.f32"
    onpolicy_path = path / "shadow_macro_onpolicy.f32"
    csv_path = path / "shadow_macro_decisions.csv"

    for p in [obs_path, onpolicy_path, csv_path]:
        if not p.exists():
            raise RuntimeError(f"Missing: {p}")

    obs_flat = np.fromfile(
        obs_path,
        dtype=np.float32,
    )

    onpolicy_flat = np.fromfile(
        onpolicy_path,
        dtype=np.float32,
    )

    if obs_flat.size % 2054 != 0:
        raise RuntimeError(
            f"Bad obs size: {path}"
        )

    if onpolicy_flat.size % 8 != 0:
        raise RuntimeError(
            f"Bad onpolicy size: {path}"
        )

    obs = obs_flat.reshape(-1, 2054)
    onpolicy = onpolicy_flat.reshape(-1, 8)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)

    if not (
        obs.shape[0]
        == onpolicy.shape[0]
        == n
        == 35
    ):
        raise RuntimeError(
            f"Expected 35 macro decisions: {path}; "
            f"obs={obs.shape[0]}, "
            f"onpolicy={onpolicy.shape[0]}, "
            f"csv={n}"
        )

    actions = onpolicy[:, :6].copy()
    old_values = onpolicy[:, 6].copy()
    old_logprob = onpolicy[:, 7].copy()

    rewards = np.zeros(n, dtype=np.float32)
    rewards[-1] = float(terminal_reward)

    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n)):

        if t == n - 1:
            next_value = 0.0
            nonterminal = 0.0
        else:
            next_value = old_values[t + 1]
            nonterminal = 1.0

        delta = (
            rewards[t]
            + GAMMA * next_value * nonterminal
            - old_values[t]
        )

        last_gae = (
            delta
            + GAMMA
            * GAE_LAMBDA
            * nonterminal
            * last_gae
        )

        advantages[t] = last_gae

    returns = advantages + old_values

    return {
        "obs": obs,
        "actions": actions,
        "old_values": old_values,
        "old_logprob": old_logprob,
        "advantages": advantages,
        "returns": returns,
    }


success = load_episode(T1, +1.0)
failure = load_episode(T2, -1.0)


obs = np.concatenate(
    [success["obs"], failure["obs"]],
    axis=0,
)

actions = np.concatenate(
    [success["actions"], failure["actions"]],
    axis=0,
)

old_values = np.concatenate(
    [success["old_values"], failure["old_values"]],
    axis=0,
)

old_logprob = np.concatenate(
    [success["old_logprob"], failure["old_logprob"]],
    axis=0,
)

advantages_raw = np.concatenate(
    [success["advantages"], failure["advantages"]],
    axis=0,
)

returns = np.concatenate(
    [success["returns"], failure["returns"]],
    axis=0,
)

adv_mean = float(advantages_raw.mean())
adv_std = float(advantages_raw.std())

if adv_std < 1e-8:
    raise RuntimeError(
        "Advantage variance is zero"
    )

advantages = (
    advantages_raw - adv_mean
) / (adv_std + 1e-8)


print("===== TRAINING PAIR =====")
print("success_steps =", len(success["obs"]))
print("failure_steps =", len(failure["obs"]))
print("batch_steps   =", len(obs))
print()
print(
    "success_adv_mean =",
    float(advantages[:35].mean()),
)
print(
    "failure_adv_mean =",
    float(advantages[35:].mean()),
)


# ============================================================
# 4. On-policy contract
# ============================================================

model = PPO.load(
    str(MODEL_BASE),
    device="cpu",
)

obs_t = torch.as_tensor(
    obs,
    dtype=torch.float32,
    device=model.device,
)

actions_t = torch.as_tensor(
    actions,
    dtype=torch.float32,
    device=model.device,
)

old_logprob_t = torch.as_tensor(
    old_logprob,
    dtype=torch.float32,
    device=model.device,
)

returns_t = torch.as_tensor(
    returns,
    dtype=torch.float32,
    device=model.device,
)

advantages_t = torch.as_tensor(
    advantages,
    dtype=torch.float32,
    device=model.device,
)


with torch.no_grad():

    values_check, logprob_check, _ = (
        model.policy.evaluate_actions(
            obs_t,
            actions_t,
        )
    )

    values_check = values_check.flatten()

    logprob_max_diff = float(
        torch.max(
            torch.abs(
                logprob_check
                - old_logprob_t
            )
        ).cpu()
    )

    value_max_diff = float(
        torch.max(
            torch.abs(
                values_check
                - torch.as_tensor(
                    old_values,
                    dtype=torch.float32,
                )
            )
        ).cpu()
    )


print()
print("===== ON-POLICY CONTRACT =====")
print(
    "old_logprob_max_diff =",
    logprob_max_diff,
)
print(
    "old_value_max_diff   =",
    value_max_diff,
)

if logprob_max_diff > 1e-4:
    raise RuntimeError(
        "STOP: stored logprob does not match "
        "the safe-init policy"
    )

if value_max_diff > 1e-4:
    raise RuntimeError(
        "STOP: stored values do not match "
        "the safe-init policy"
    )


# ============================================================
# 5. Tiny PPO update on COPY
# ============================================================

candidate = PPO.load(
    str(MODEL_BASE),
    device="cpu",
)

# 第一批只有两个 episode：
# 不允许改变 exploration std。
candidate.policy.log_std.requires_grad_(False)

for group in candidate.policy.optimizer.param_groups:
    group["lr"] = LEARNING_RATE


def evaluate(m):

    with torch.no_grad():

        values, new_logprob, _ = (
            m.policy.evaluate_actions(
                obs_t,
                actions_t,
            )
        )

        values = values.flatten()

        ratio = torch.exp(
            new_logprob - old_logprob_t
        )

        surr1 = ratio * advantages_t

        surr2 = (
            torch.clamp(
                ratio,
                1.0 - CLIP_RANGE,
                1.0 + CLIP_RANGE,
            )
            * advantages_t
        )

        surrogate = torch.mean(
            torch.minimum(
                surr1,
                surr2,
            )
        )

        approx_kl = torch.mean(
            old_logprob_t
            - new_logprob
        )

        clip_fraction = torch.mean(
            (
                torch.abs(ratio - 1.0)
                > CLIP_RANGE
            ).float()
        )

        value_mse = torch.mean(
            (values - returns_t) ** 2
        )

        success_delta = torch.mean(
            new_logprob[:35]
            - old_logprob_t[:35]
        )

        failure_delta = torch.mean(
            new_logprob[35:]
            - old_logprob_t[35:]
        )

    return {
        "surrogate": float(
            surrogate.cpu()
        ),
        "approx_kl": float(
            approx_kl.cpu()
        ),
        "clip_fraction": float(
            clip_fraction.cpu()
        ),
        "value_mse": float(
            value_mse.cpu()
        ),
        "success_logprob_delta": float(
            success_delta.cpu()
        ),
        "failure_logprob_delta": float(
            failure_delta.cpu()
        ),
    }


before = evaluate(candidate)

epoch_metrics = []

for epoch in range(EPOCHS):

    values, new_logprob, entropy = (
        candidate.policy.evaluate_actions(
            obs_t,
            actions_t,
        )
    )

    values = values.flatten()

    ratio = torch.exp(
        new_logprob - old_logprob_t
    )

    policy_loss_1 = (
        advantages_t * ratio
    )

    policy_loss_2 = (
        advantages_t
        * torch.clamp(
            ratio,
            1.0 - CLIP_RANGE,
            1.0 + CLIP_RANGE,
        )
    )

    policy_loss = -torch.mean(
        torch.minimum(
            policy_loss_1,
            policy_loss_2,
        )
    )

    value_loss = 0.5 * torch.mean(
        (values - returns_t) ** 2
    )

    loss = (
        policy_loss
        + VF_COEF * value_loss
    )

    candidate.policy.optimizer.zero_grad()

    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        candidate.policy.parameters(),
        MAX_GRAD_NORM,
    )

    candidate.policy.optimizer.step()

    m = evaluate(candidate)

    m["epoch"] = epoch + 1
    m["policy_loss"] = float(
        policy_loss.detach().cpu()
    )
    m["value_loss"] = float(
        value_loss.detach().cpu()
    )
    m["grad_norm_before_clip"] = float(
        grad_norm.detach().cpu()
    )

    epoch_metrics.append(m)

    print()
    print(
        f"===== EPOCH {epoch + 1} ====="
    )
    print(
        "surrogate =",
        m["surrogate"],
    )
    print(
        "approx_kl =",
        m["approx_kl"],
    )
    print(
        "clip_fraction =",
        m["clip_fraction"],
    )
    print(
        "success_logprob_delta =",
        m["success_logprob_delta"],
    )
    print(
        "failure_logprob_delta =",
        m["failure_logprob_delta"],
    )

    if abs(m["approx_kl"]) > MAX_APPROX_KL:
        print(
            "EARLY STOP: KL limit reached"
        )
        break


after = evaluate(candidate)


# ============================================================
# 6. Safety acceptance
# ============================================================

std_before = (
    model.policy.log_std
    .detach()
    .cpu()
    .numpy()
    .copy()
)

std_after = (
    candidate.policy.log_std
    .detach()
    .cpu()
    .numpy()
    .copy()
)

std_max_diff = float(
    np.max(
        np.abs(
            std_after - std_before
        )
    )
)

surrogate_improved = (
    after["surrogate"]
    > before["surrogate"]
)

kl_safe = (
    abs(after["approx_kl"])
    <= MAX_APPROX_KL
)

clip_safe = (
    after["clip_fraction"]
    <= 0.10
)

std_fixed = (
    std_max_diff == 0.0
)


# 原始 checkpoint 必须完全不变
original_hash_after = sha256(MODEL_ZIP)

original_untouched = (
    original_hash_before
    == original_hash_after
)


passed = all(
    [
        surrogate_improved,
        kl_safe,
        clip_safe,
        std_fixed,
        original_untouched,
    ]
)


print()
print("===== UPDATE SUMMARY =====")
print(
    "surrogate_before =",
    before["surrogate"],
)
print(
    "surrogate_after  =",
    after["surrogate"],
)
print(
    "approx_kl        =",
    after["approx_kl"],
)
print(
    "clip_fraction    =",
    after["clip_fraction"],
)
print(
    "success_logprob_delta =",
    after["success_logprob_delta"],
)
print(
    "failure_logprob_delta =",
    after["failure_logprob_delta"],
)
print(
    "std_max_diff =",
    std_max_diff,
)
print(
    "original_model_untouched =",
    original_untouched,
)

print()
print(
    "surrogate_improved =",
    surrogate_improved,
)
print(
    "kl_safe            =",
    kl_safe,
)
print(
    "clip_safe          =",
    clip_safe,
)
print(
    "std_fixed          =",
    std_fixed,
)


metrics = {
    "success_episode": str(T1),
    "failure_episode": str(T2),
    "failure_reason":
        "GRIPPER_COLLISION_TIPPED_TISSUE",
    "batch_steps": 70,
    "gamma": GAMMA,
    "gae_lambda": GAE_LAMBDA,
    "learning_rate": LEARNING_RATE,
    "epochs_requested": EPOCHS,
    "clip_range": CLIP_RANGE,
    "vf_coef": VF_COEF,
    "max_grad_norm": MAX_GRAD_NORM,
    "before": before,
    "after": after,
    "epoch_metrics": epoch_metrics,
    "onpolicy_logprob_max_diff":
        logprob_max_diff,
    "onpolicy_value_max_diff":
        value_max_diff,
    "std_max_diff": std_max_diff,
    "original_model_untouched":
        original_untouched,
    "passed": passed,
}

METRICS.write_text(
    json.dumps(
        metrics,
        indent=2,
    )
)


if passed:

    candidate.save(
        str(OUT_MODEL)
    )

    print()
    print("PASS = True")
    print(
        "CANDIDATE SAVED =",
        str(OUT_MODEL) + ".zip",
    )

else:

    print()
    print("PASS = False")
    print(
        "Candidate NOT approved for robot."
    )


# 最终再检查一次安全门
control_text_after = Path(
    "src/lerobot/rollout/residual_control.py"
).read_text()

if "ACTIVE_SCALE = 0.0" not in control_text_after:
    raise RuntimeError(
        "FINAL SAFETY ERROR: ACTIVE_SCALE changed"
    )


print()
print(
    "metrics =",
    METRICS,
)

print("ACTIVE_SCALE = 0.0")

print()
print(
    "FIRST REAL PAIR PPO UPDATE FINISHED"
)
