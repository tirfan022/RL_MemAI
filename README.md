# RL_MemAI — Cache Replacement with Deep Reinforcement Learning

**A Double DQN learns a cache-eviction policy from scratch and is benchmarked
against three classical algorithms (FIFO, LRU, CLOCK) on skewed, realistic
access patterns — documented in full below, since there's no live dashboard
to demo this with.**

> **TL;DR:** trained a Double DQN with Belady-distance reward shaping, tuned
> it with Optuna, then stress-tested the result across 8 independent
> held-out traces instead of trusting a single number. Final,
> honestly-reported result: **the trained agent beats LRU on 6 of 8
> held-out traces** (mean hit rate 55.6% vs. LRU's 54.4%), and loses
> narrowly on the other 2. Full methodology and results below.

---

## Table of contents
1. [What this project does](#1-what-this-project-does)
2. [Architecture](#2-architecture)
3. [The MDP: state, action, reward](#3-the-mdp-state-action-reward)
4. [Hyperparameter tuning with Optuna](#4-hyperparameter-tuning-with-optuna)
5. [Results](#5-results)
6. [Honest limitations — what this does *not* show](#6-honest-limitations--what-this-does-not-show)
7. [Why isn't this just used in production caches?](#7-why-isnt-this-just-used-in-production-caches)
8. [Project structure](#8-project-structure)
9. [Running it](#9-running-it)
10. [Skills this project demonstrates](#10-skills-this-project-demonstrates)
11. [References](#11-references)

---

## 1. What this project does

Cache eviction is normally handled by a fixed heuristic: FIFO evicts the
oldest entry, LRU evicts the least-recently-used one, CLOCK approximates LRU
cheaply with a circular buffer. This project instead treats eviction as a
**sequential decision problem**: an agent watches the cache fill up under a
skewed (Zipfian) access pattern and, every time it needs to evict something,
chooses *which slot* to evict — trained with a Double Deep Q-Network to
maximize long-run hit rate.

The interesting design choice is *how* the agent is taught: rather than a
sparse hit/miss signal, training rewards are shaped using **Belady's
offline-optimal algorithm** as a coach — the agent is told how good or bad
each eviction was by comparing it to what the *optimal* offline algorithm
would have done, using knowledge of the future access sequence. This is only
valid during training (the full trace is available in advance); at
inference time the trained policy sees none of that — it only ever sees the
current cache state, just like a real classical policy would.

This is not a novel idea invented here — it's the same core technique behind
Meta and Princeton's **Learning Relaxed Belady (LRB)**, published at NSDI
2020, which used an approximation of Belady's algorithm to reduce WAN traffic
by 5–24% in real production CDN traces. This project is a small-scale,
from-scratch reproduction of that idea.

---

## 2. Architecture

![System architecture](assets/architecture_diagram.png)

**Pipeline, left to right / top to bottom:** a `WorkloadGenerator` produces a
page-access trace (Zipfian-distributed — a small number of "hot" pages get
most of the traffic, which is realistic for real memory/cache access
patterns) → the same trace drives both the custom `CacheEnv` (a Gymnasium
environment) and the classical baselines, so every policy is compared on
identical input → `CacheEnv` turns the DQN agent's eviction choices into
reward → the `DQNAgent` (online network + target network, replay memory,
ε-greedy exploration) learns from that reward stream → `metrics/benchmark.py`
and `multi_trace_benchmark.py` replay held-out traces through the trained
agent and the three baselines side by side → `plots/*.py` turns the logs into
the charts in this document. `main.py` runs the whole pipeline end to end in
one command.

---

## 3. The MDP: state, action, reward

| Component | Definition |
|---|---|
| **State** | For each of the 32 cache slots: `[page_id, access_frequency, recency]`, normalized to comparable ranges (empty slots padded), plus the incoming page id. |
| **Action** | `Discrete(32)` — which cache slot to evict. Only consulted on a miss when the cache is full. |
| **Reward** | `+1` on a hit · `+0.1` on a miss with a free slot · on a full-cache miss, a reward derived from **Belady's look-ahead distance** for the evicted page (evicting a page that won't be needed for a long time is rewarded; evicting one needed again soon is penalized). |

**Network:** a 4-layer MLP (`128 → 128 → 64 → action_dim`), trained as a
**Double DQN** — the online network selects the next action, the target
network evaluates it, which reduces the value-overestimation bias that
plagues vanilla DQN. Target network updates are **soft (Polyak averaging)**
applied every learning step, rather than periodic hard copies — chosen for
training stability, since infrequent hard copies let the target lag too far
behind the online network given this environment's reward scale.

---

## 4. Hyperparameter tuning with Optuna

With training stabilized, a proper hyperparameter search became worth doing.
`tune.py` runs an Optuna study (TPE sampler + median pruner) over `gamma`,
learning rate, `tau` (soft-update rate), `epsilon_decay`, and `batch_size`.
Each trial trains on a shorter proxy budget (25 episodes, not the full 150)
using **the same fixed set of workloads across every trial**, so differences
in score reflect the hyperparameters, not a lucky workload draw — and each
trial is scored the same way as `EvalHitRate` above: greedy evaluation on one
fixed held-out trace. The study is resumable (SQLite-backed), so it doesn't
need one long uninterrupted session.

25 trials found:

```json
{
  "gamma": 0.899,
  "lr": 2.56e-05,
  "tau": 0.00176,
  "epsilon_decay": 0.9535,
  "batch_size": 128
}
```

Worth being direct about the limits of this: with a 25-episode proxy budget
and 25 trials, this is a search that narrows the space efficiently — it is
not an exhaustive or rigorous sweep, and shouldn't be described as finding a
global optimum.

---

## 5. Results

### Training dynamics

| Reward vs. episode | Hit rate vs. episode (noisy, per-episode) | Epsilon decay |
|---|---|---|
| ![Reward](assets/reward_curve.png) | ![Hit rate](assets/hitrate_curve.png) | ![Epsilon](assets/epsilon_curve.png) |

The per-episode hit rate above is measured on a **freshly randomized
workload every episode**, so it's noisy and not by itself evidence of
learning. A second metric, `EvalHitRate`, evaluates the policy greedily
(no exploration) on one **fixed, seeded** held-out trace every 5 episodes —
that's the number that actually trends upward as training progresses:

![Eval hit rate vs. episode, compared to the three baselines](assets/eval_hitrate_curve.png)

### Benchmark — single held-out trace

![Benchmark bar chart](assets/benchmark_comparison.png)

| Policy | Hit Rate |
|---|---|
| FIFO | 46.40% |
| CLOCK | 50.90% |
| LRU | 52.65% |
| **DQN** | **56.45%** |

### Robustness — 8 independent held-out traces (no retraining)

A single trace can be a lucky draw, so the trained model (unchanged) was
re-evaluated against 8 more independently-drawn traces from the same
workload family:

| Policy | Mean | Std | Min | Max |
|---|---|---|---|---|
| FIFO | 48.19% | 1.41 | 45.00% | 50.20% |
| CLOCK | 52.54% | 1.27 | 49.75% | 54.50% |
| LRU | 54.36% | 1.30 | 51.40% | 55.95% |
| **DQN** | **55.62%** | 1.94 | 52.30% | 58.45% |

**DQN beat all three baselines on 6 of 8 traces**, and was narrowly
outperformed by LRU on the remaining 2 (53.70% vs. 54.70%, and 54.50% vs.
55.00%). The honest summary is "beats LRU on most held-out draws by roughly a
point on average," not "always wins" — and that's the claim this project
actually makes, not a stronger one.

---

## 6. Honest limitations — what this does *not* show

- **One workload family.** Everything above is Zipfian with a single skew
  parameter (`alpha=1.1`, 500 pages). There's no evidence yet this
  generalizes to a different skew, a scan-heavy pattern, or a real trace.
- **Toy scale.** 32 cache slots, 500 pages. Real caches are orders of
  magnitude larger.
- **Training-run variance not yet measured.** The 8-trace robustness check
  varies the *test* trace, holding the trained policy fixed. It does not yet
  show how much the *learned policy itself* varies across independent
  training runs (different seeds) — that would need 2–3 full retrains,
  flagged as follow-up work.
- **Inference cost not measured.** The benchmark only reports hit rate, never
  decision latency — which matters a lot in practice (see §7).

---

## 7. Why isn't this just used in production caches?

A fair question after a positive benchmark. Three compounding reasons real
systems mostly stick to classical heuristics:

1. **Inference latency.** A neural forward pass costs microseconds —
   irrelevant for a CDN cache making a few thousand decisions/sec, but
   disqualifying for CPU L1/L2/L3 caches or OS page tables, which decide in
   nanoseconds, billions of times/sec. (Meta's LRB, mentioned above, uses
   lightweight gradient-boosted trees rather than a deep network for exactly
   this reason — inference cost was a first-class constraint.)
2. **Predictability and drift.** LRU/CLOCK/ARC have no training data to go
   stale, and their worst-case behavior is well understood. A learned policy
   trained on one traffic period can silently degrade as patterns shift, and
   diagnosing why is harder than diagnosing a heuristic.
3. **The heuristics are already very good.** Modern non-ML policies (ARC,
   W-TinyLFU) get close to Belady-optimal on real workloads at O(1) cost —
   which is why even LRB's production numbers, against strong baselines, are
   a meaningful-but-not-dramatic improvement (5-24%) rather than a blowout.

---

## 8. Project structure

```
RL_MemAI/
├── main.py                    # train → benchmark → plots, end-to-end
├── tune.py                    # Optuna hyperparameter search (§4)
├── multi_trace_benchmark.py   # robustness check across held-out traces (§5)
├── requirements.txt
├── agent/
│   ├── agent.py                # DQNAgent: Double DQN, soft target updates, Huber loss
│   ├── network.py               # CacheQNetwork (4-layer MLP)
│   └── train.py                 # training loop: seeded, fixed-trace EvalHitRate, CSV logging
├── cache/
│   ├── base.py                  # CacheBase ABC
│   ├── cache_env.py             # Gymnasium CacheEnv: normalized state, scaled/oracle reward
│   └── fifo.py / lru.py / clock.py   # classical baselines
├── metrics/
│   └── benchmark.py             # DQN vs. FIFO/LRU/CLOCK, single trace
├── plots/
│   ├── generate_plots.py        # reward/hit-rate/epsilon curves
│   └── benchmark_plot.py        # baseline-vs-DQN bar chart
├── workload/
│   └── generator.py             # Zipfian / locality / mixed synthetic traces
└── assets/                      # every image in this README
```

---

## 9. Running it

```bash
pip install -r requirements.txt

# full pipeline: train -> benchmark -> plots (~15 min on CPU)
python main.py

# optional: hyperparameter search before training
python tune.py --trials 25 --episodes 25
# -> writes best_hparams.json; copy values into the DQNAgent(...) call in agent/train.py

# optional: robustness check across multiple held-out traces (no retraining)
python multi_trace_benchmark.py --traces 8
```

`models/` and `logs/` are gitignored and regenerated by the commands above.

---

## 10. Skills this project demonstrates

- Designing a custom Gymnasium RL environment (state/action/reward design, reward shaping via an offline oracle)
- Double DQN implementation (online/target networks, soft target updates, experience replay)
- **Diagnosing training instability from first principles** (deriving the Q-value scale from `gamma` and reward magnitude, rather than guessing at hyperparameters)
- Catching a metric bug that was masking the real state of a system — the kind of bug that "looks like it's working"
- Hyperparameter optimization with Optuna (resumable studies, pruning, proxy objectives)
- Statistical validation discipline: not trusting a single benchmark number, testing across independent traces, and reporting a result that's honest about its limits (6/8, not "always wins")
- Technical writing: this document itself, as the primary artifact in place of a live demo

---

## 11. References

- Song, Z., Berger, D. S., Li, K., & Lloyd, W. (2020). **Learning Relaxed Belady for Content Distribution Network Caching.** *17th USENIX Symposium on Networked Systems Design and Implementation (NSDI 20)*, pp. 529–544. [usenix.org/conference/nsdi20/presentation/song](https://www.usenix.org/conference/nsdi20/presentation/song) — the production-scale precedent for the Belady-distance reward shaping used in §1/§3.
- Belady, L. A. (1966). **A Study of Replacement Algorithms for a Virtual-Storage Computer.** *IBM Systems Journal*, 5(2), 78–101. — the offline-optimal algorithm this project's reward function is shaped against.
- Mnih, V., et al. (2015). **Human-level control through deep reinforcement learning.** *Nature*, 518, 529–533. — the original DQN architecture (online network, target network, experience replay).
- van Hasselt, H., Guez, A., & Silver, D. (2016). **Deep Reinforcement Learning with Double Q-learning.** *AAAI 2016*. [arXiv:1509.06461](https://arxiv.org/abs/1509.06461) — Double DQN, used here to reduce Q-value overestimation.
- Breslau, L., Cao, P., Fan, L., Phillips, G., & Shenker, S. (1999). **Web Caching and Zipf-like Distributions: Evidence and Implications.** *IEEE INFOCOM 1999*. — justification for using Zipfian-distributed synthetic workloads to approximate real cache/web access skew.
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). **Optuna: A Next-generation Hyperparameter Optimization Framework.** *KDD 2019*. [optuna.org](https://optuna.org) — used in `tune.py` (§4).
- **Gymnasium** (Farama Foundation), the maintained successor to OpenAI Gym — [gymnasium.farama.org](https://gymnasium.farama.org) — RL environment API used by `cache/cache_env.py`.
- Paszke, A., et al. (2019). **PyTorch: An Imperative Style, High-Performance Deep Learning Library.** *NeurIPS 2019*. [pytorch.org](https://pytorch.org)
