"""
Reward Model from Scratch — RLHF Stage 2
==========================================
Trains a scalar reward head on top of a frozen LLM using Bradley-Terry loss
over human preference pairs. This is the reward model that InstructGPT / RLHF
uses to drive PPO; skipping it is exactly what DPO does, so building one
explicitly shows the full alignment pipeline end-to-end.

Model   : Qwen/Qwen2.5-1.5B-Instruct  (1.54B params, float16 backbone)
Dataset : Anthropic HH-RLHF harmless-base  (200 train / 40 eval pairs)
Technique: LoRA backbone + linear reward head — from scratch (no TRL / PEFT)
Hardware : Apple M4 MPS  (peak ~6 GB unified memory)

Architecture:
  input_ids ──► [frozen Qwen backbone + LoRA(q,v)] ──► last_hidden (1,T,H)
                                                             │
                                                    last non-pad token
                                                             ▼
                                             reward_head: Linear(H, 1)
                                                             ▼
                                                    scalar reward r(x)

Bradley-Terry loss (Christiano et al. 2017):
  L = -E[log σ(r(x, y_chosen) - r(x, y_rejected))]

Why this is faster than DPO:
  DPO  : 4 forward passes / step  (policy + reference on both chosen+rejected)
  RM   : 2 forward passes / step  (only reward head — no reference needed)
  → roughly 2× the throughput on the same hardware

Key measurements:
  1. BT loss over training
  2. Reward margin r(chosen) - r(rejected)
  3. Pairwise preference accuracy on held-out eval set
  4. Reward distribution histograms (chosen vs rejected)
"""
import os, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE        = torch.float16
MAX_LEN      = 128
N_TRAIN      = 200
N_EVAL       = 40
LR_LORA      = 2e-4
LR_HEAD      = 1e-3        # head is tiny + untrained, benefits from larger LR
N_STEPS      = 350
GRAD_ACCUM   = 2
LORA_RANK    = 4
LORA_ALPHA   = 8
LORA_DROP    = 0.05
SEED         = 42
PLOTS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

torch.manual_seed(SEED); np.random.seed(SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── LoRA from scratch ─────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with trainable low-rank adapters."""
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base    = base
        self.rank    = rank
        self.scale   = alpha / rank
        in_f, out_f  = base.in_features, base.out_features
        dev = base.weight.device
        self.A = nn.Parameter(torch.randn(rank, in_f,  dtype=torch.float32, device=dev) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_f, rank, dtype=torch.float32, device=dev))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out   = self.base(x)
        x32   = self.drop(x.to(torch.float32))
        delta = (x32 @ self.A.T @ self.B.T) * self.scale
        return out + delta.to(out.dtype)

def inject_lora(model, rank, alpha, dropout, targets=("q_proj", "v_proj")):
    trainable = []
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear): continue
        if not any(full_name.endswith(t) for t in targets): continue
        parts  = full_name.split(".")
        parent = model.get_submodule(".".join(parts[:-1]))
        layer  = LoRALinear(module, rank, alpha, dropout)
        setattr(parent, parts[-1], layer)
        trainable += [layer.A, layer.B]
    return model, trainable

# ── Reward head ───────────────────────────────────────────────────────────────
class RewardHead(nn.Module):
    """Linear projection from last non-pad token hidden state to scalar reward."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, T, H)  attention_mask: (B, T)
        last_idx  = attention_mask.sum(-1) - 1                 # (B,)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        h_last    = hidden_states[batch_idx, last_idx]         # (B, H)
        return self.proj(h_last.float()).squeeze(-1)           # (B,)

# ── Dataset ───────────────────────────────────────────────────────────────────
def load_data():
    print("Loading Anthropic/hh-rlhf (harmless-base)…")
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
    ds = ds.filter(lambda x: len(x["chosen"]) < 600 and len(x["rejected"]) < 600)
    total = min(N_TRAIN + N_EVAL, len(ds))
    ds    = ds.shuffle(seed=SEED).select(range(total))
    return ds.select(range(N_TRAIN)), ds.select(range(N_TRAIN, total))

# ── Forward / loss ────────────────────────────────────────────────────────────
def reward_score(backbone, head, tokenizer, text: str,
                 requires_grad: bool) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", max_length=MAX_LEN,
                    truncation=True, padding=False).to(DEVICE)
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        out = backbone.model(input_ids=enc["input_ids"],
                             attention_mask=enc["attention_mask"])
        r   = head(out.last_hidden_state, enc["attention_mask"])
    return r.squeeze(0)                                        # scalar tensor

def bt_step(backbone, head, tokenizer, chosen: str, rejected: str):
    r_c = reward_score(backbone, head, tokenizer, chosen,   True)
    r_r = reward_score(backbone, head, tokenizer, rejected, True)
    margin = r_c - r_r
    loss   = -F.logsigmoid(margin)
    return loss, r_c.item(), r_r.item(), margin.item()

# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(backbone, head, tokenizer, eval_ds, n: int):
    correct, r_c_all, r_r_all = 0, [], []
    for i in range(min(n, len(eval_ds))):
        ex = eval_ds[i]
        r_c = reward_score(backbone, head, tokenizer, ex["chosen"],   False).item()
        r_r = reward_score(backbone, head, tokenizer, ex["rejected"], False).item()
        correct += int(r_c > r_r)
        r_c_all.append(r_c); r_r_all.append(r_r)
        if DEVICE == "mps":
            torch.mps.empty_cache()
    acc = correct / min(n, len(eval_ds))
    return acc, r_c_all, r_r_all

# ── Plots ─────────────────────────────────────────────────────────────────────
def save_plots(losses, margins, eval_curve, r_c_final, r_r_final,
               n_base, n_lora, n_head):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    pct = 100 * (n_lora + n_head) / n_base
    fig.suptitle(
        f"Reward Model (RLHF Stage 2) — Qwen2.5-1.5B-Instruct  |  "
        f"LoRA rank-{LORA_RANK} + scalar head  |  "
        f"{n_lora + n_head:,} trainable ({pct:.3f}% of {n_base/1e9:.2f}B)",
        fontsize=10)

    def smooth(v, w=15):
        w = min(w, len(v)//3 or 1)
        return np.convolve(v, np.ones(w)/w, mode="valid"), w

    ax = axes[0, 0]
    s, w = smooth(losses)
    ax.plot(losses, alpha=0.25, color="steelblue")
    ax.plot(range(w-1, len(losses)), s, color="steelblue", lw=2)
    ax.set_title("Bradley-Terry Loss")
    ax.set_xlabel("Step"); ax.set_ylabel("-log σ(r_c - r_r)")

    ax = axes[0, 1]
    s, w = smooth(margins)
    ax.plot(margins, alpha=0.25, color="green")
    ax.plot(range(w-1, len(margins)), s, color="green", lw=2)
    ax.axhline(0, color="red", ls="--", alpha=0.5, label="margin = 0")
    ax.set_title("Reward Margin  r(chosen) − r(rejected)")
    ax.set_xlabel("Step"); ax.legend()

    ax = axes[1, 0]
    if eval_curve:
        steps, accs = zip(*eval_curve)
        ax.plot(steps, accs, "o-", color="purple", lw=2, ms=6)
        ax.axhline(0.5, color="red", ls="--", alpha=0.5, label="random baseline")
        ax.set_ylim(0, 1)
    ax.set_title("Pairwise Preference Accuracy (eval)")
    ax.set_xlabel("Step"); ax.legend()

    ax = axes[1, 1]
    bins = np.linspace(min(min(r_c_final), min(r_r_final)),
                       max(max(r_c_final), max(r_r_final)), 20)
    ax.hist(r_r_final, bins=bins, alpha=0.6, color="crimson",    label="rejected")
    ax.hist(r_c_final, bins=bins, alpha=0.6, color="seagreen",   label="chosen")
    ax.set_title("Final Reward Distribution (eval set)")
    ax.set_xlabel("reward r(x)"); ax.set_ylabel("count"); ax.legend()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "reward_model_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device : {DEVICE}")
    print(f"Model  : {MODEL_NAME}")
    print("Loading model (float16)… using local cache if available.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=DTYPE, low_cpu_mem_usage=True)
    backbone = backbone.to(DEVICE)
    backbone.eval()
    hidden_size = backbone.config.hidden_size

    n_base = sum(p.numel() for p in backbone.parameters())
    backbone, lora_params = inject_lora(backbone, LORA_RANK, LORA_ALPHA, LORA_DROP)

    # Freeze base; only LoRA A/B train in the backbone
    for n, p in backbone.named_parameters():
        p.requires_grad = n.endswith(".A") or n.endswith(".B")

    # Reward head in float32 for stable scalar regression
    head = RewardHead(hidden_size).to(DEVICE)
    n_lora = sum(p.numel() for p in lora_params)
    n_head = sum(p.numel() for p in head.parameters())
    print(f"Base params    : {n_base/1e9:.3f}B  (frozen)")
    print(f"LoRA trainable : {n_lora:,}  ({100*n_lora/n_base:.4f}%)")
    print(f"Head trainable : {n_head:,}  ({hidden_size} → 1)")

    train_ds, eval_ds = load_data()
    print(f"Train pairs : {len(train_ds)}   Eval pairs : {len(eval_ds)}")

    optimizer = torch.optim.AdamW([
        {"params": lora_params,             "lr": LR_LORA},
        {"params": list(head.parameters()), "lr": LR_HEAD},
    ], weight_decay=0.01)
    optimizer.zero_grad()

    losses, margins, eval_curve = [], [], []
    idx = list(range(len(train_ds))); np.random.shuffle(idx)
    t0  = time.time()

    print(f"\nTraining for {N_STEPS} steps  (GRAD_ACCUM={GRAD_ACCUM})…\n")
    for step in range(N_STEPS):
        ex = train_ds[idx[step % len(idx)]]
        loss, r_c, r_r, margin = bt_step(backbone, head, tokenizer,
                                         ex["chosen"], ex["rejected"])
        (loss / GRAD_ACCUM).backward()
        losses.append(loss.item()); margins.append(margin)

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                lora_params + list(head.parameters()), 1.0)
            optimizer.step(); optimizer.zero_grad()
            if DEVICE == "mps":
                torch.mps.empty_cache()

        if step % 20 == 0:
            print(f"  step {step:3d} | loss={loss.item():.4f} | "
                  f"r_c={r_c:+.3f} r_r={r_r:+.3f} margin={margin:+.3f} | "
                  f"{(time.time()-t0)/60:.1f} min")

        if step > 0 and step % 50 == 0:
            acc, _, _ = evaluate(backbone, head, tokenizer, eval_ds, n=20)
            eval_curve.append((step, acc))
            print(f"           eval accuracy = {acc:.3f}")

    final_acc, r_c_final, r_r_final = evaluate(
        backbone, head, tokenizer, eval_ds, n=N_EVAL)
    eval_curve.append((N_STEPS, final_acc))
    elapsed = (time.time() - t0) / 60

    print(f"\n{'─'*55}")
    print(f"Final BT loss             : {losses[-1]:.4f}")
    print(f"Final reward margin       : {margins[-1]:+.4f}")
    print(f"Mean r(chosen)  on eval   : {np.mean(r_c_final):+.3f}")
    print(f"Mean r(rejected) on eval  : {np.mean(r_r_final):+.3f}")
    print(f"Final preference accuracy : {final_acc:.3f}  (0.5 = random)")
    print(f"Training time             : {elapsed:.1f} min")
    print(f"{'─'*55}")

    save_plots(losses, margins, eval_curve, r_c_final, r_r_final,
               n_base, n_lora, n_head)

    results = {
        "model": MODEL_NAME, "device": DEVICE,
        "base_params_B": round(n_base/1e9, 3),
        "lora_trainable_params": n_lora,
        "head_trainable_params": n_head,
        "total_trainable_pct": round(100*(n_lora+n_head)/n_base, 4),
        "lora_rank": LORA_RANK,
        "n_train": len(train_ds), "n_eval": len(eval_ds),
        "n_steps": N_STEPS, "lr_lora": LR_LORA, "lr_head": LR_HEAD,
        "final_loss": round(losses[-1], 4),
        "final_reward_margin": round(margins[-1], 4),
        "mean_reward_chosen":  round(float(np.mean(r_c_final)), 4),
        "mean_reward_rejected": round(float(np.mean(r_r_final)), 4),
        "reward_gap":           round(float(np.mean(r_c_final) - np.mean(r_r_final)), 4),
        "final_preference_accuracy": round(final_acc, 4),
        "training_minutes": round(elapsed, 1),
        "portfolio_note": (
            "Stage 2 of the classical RLHF pipeline (Christiano 2017 / InstructGPT). "
            "Trains an explicit scalar reward model with Bradley-Terry loss, which "
            "would drive PPO in stage 3. DPO collapses stages 2+3 — building this "
            "from scratch shows the full pipeline DPO short-circuits."
        )
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"\nResults → {out}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

