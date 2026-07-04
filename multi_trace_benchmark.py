"""
Multi-trace robustness check for the trained DQN model.

Runs the SAME already-trained model (models/best_model.pth) against several
independently-drawn held-out Zipfian traces (same distribution family/params
as training, different random draws), plus the same traces through FIFO/LRU/
CLOCK. This isolates "does the win generalize across which specific access
sequence you test on" from training-run variance -- no retraining needed.

Usage:
    python multi_trace_benchmark.py --traces 8

Writes logs/multi_trace_results.csv and prints a mean +/- std summary table.
"""
import os
import sys
import csv
import argparse

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cache.cache_env import CacheEnv
from workload.generator import WorkloadGenerator
from agent.agent import DQNAgent
from cache.fifo import FIFO
from cache.lru import LRU
from cache.clock import Clock


def run_one_trace(workload, capacity, agent):
    results = {}

    baselines = {"FIFO": FIFO(capacity), "LRU": LRU(capacity), "CLOCK": Clock(capacity)}
    for name, policy in baselines.items():
        policy.reset()
        for page in workload:
            policy.access(page)
        stats = policy.get_stats()
        total = stats["Hits"] + stats["Misses"]
        results[name] = (stats["Hits"] / total) if total > 0 else 0.0

    env = CacheEnv(workload=workload, capacity=capacity)
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.act(state, eval_mode=True)
        state, _, term, trunc, _ = env.step(action)
        done = term or trunc
    results["DQN"] = env.hits / (env.hits + env.misses)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=int, default=8,
                         help="number of independent held-out traces to test on")
    parser.add_argument("--length", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=1.1)
    parser.add_argument("--base-seed", type=int, default=1000,
                         help="traces are seeded base_seed, base_seed+1, ... for reproducibility")
    args = parser.parse_args()

    capacity = 32
    os.makedirs("logs", exist_ok=True)

    temp_gen = WorkloadGenerator(num_pages=500)
    probe = temp_gen.zipfian(length=10, alpha=args.alpha)
    env_probe = CacheEnv(workload=probe, capacity=capacity)

    agent = DQNAgent(env_probe.observation_space.shape[0], env_probe.action_space.n, epsilon_start=0.0)
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        print("[ERROR] models/best_model.pth not found -- run training first.")
        sys.exit(1)
    agent.model.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.model.eval()

    all_results = {"FIFO": [], "LRU": [], "CLOCK": [], "DQN": []}
    rows = []

    for i in range(args.traces):
        seed = args.base_seed + i
        np.random.seed(seed)
        gen = WorkloadGenerator(num_pages=500)
        workload = gen.zipfian(length=args.length, alpha=args.alpha)

        results = run_one_trace(workload, capacity, agent)
        rows.append({"trace_seed": seed, **results})
        for k, v in results.items():
            all_results[k].append(v)

        print(f"trace seed={seed:<6} " + "  ".join(f"{k}={v*100:5.2f}%" for k, v in results.items()))

    print("\n" + "=" * 70)
    print(f"{'Policy':<10}{'Mean':<10}{'Std':<10}{'Min':<10}{'Max':<10}{'Wins':<10}")
    print("=" * 70)
    n_traces = args.traces
    dqn_wins = sum(1 for r in rows if r["DQN"] == max(r["FIFO"], r["LRU"], r["CLOCK"], r["DQN"]))
    for policy in ["FIFO", "CLOCK", "LRU", "DQN"]:
        vals = np.array(all_results[policy]) * 100
        wins = dqn_wins if policy == "DQN" else ""
        print(f"{policy:<10}{vals.mean():<10.2f}{vals.std():<10.2f}{vals.min():<10.2f}{vals.max():<10.2f}{wins!s:<10}")
    print("=" * 70)
    print(f"DQN beat all three baselines on {dqn_wins}/{n_traces} traces.")

    with open("logs/multi_trace_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trace_seed", "FIFO", "LRU", "CLOCK", "DQN"])
        for r in rows:
            w.writerow([r["trace_seed"], r["FIFO"], r["LRU"], r["CLOCK"], r["DQN"]])
    print("\nSaved per-trace results to logs/multi_trace_results.csv")


if __name__ == "__main__":
    main()