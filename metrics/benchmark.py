import os
import csv
import sys
import torch
import numpy as np

# Path injection to guarantee cross-directory safety
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache.cache_env import CacheEnv   # Fixed directory mapping
from workload.generator import WorkloadGenerator
from agent.agent import DQNAgent

from cache.fifo import FIFO
from cache.lru import LRU
from cache.clock import Clock

def run_benchmark():
    os.makedirs("logs", exist_ok=True)
    capacity = 32
    generator = WorkloadGenerator(num_pages=500)
    test_workload = generator.zipfian(length=2000, alpha=1.1)
    
    results = {}
    
    # 1. Classical Baselines Setup
    baselines = {
        "FIFO": FIFO(capacity),
        "LRU": LRU(capacity),
        "CLOCK": Clock(capacity)
    }
    
    for name, policy in baselines.items():
        policy.reset()
        for page in test_workload:
            policy.access(page)
        stats = policy.get_stats()
        total = stats["Hits"] + stats["Misses"]
        results[name] = (stats["Hits"] / total) if total > 0 else 0.0

    # 2. DQN Baseline Execution
    env = CacheEnv(workload=test_workload, capacity=capacity)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, epsilon_start=0.0)
    
    model_path = "models/best_model.pth"
    if os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path, map_location=agent.device))
        agent.model.eval()
        
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.act(state, eval_mode=True)
            state, _, term, trunc, _ = env.step(action)
            done = term or trunc
        results["DQN"] = env.get_stats()["Hit Rate"]
    else:
        print("[WARNING] Target best_model.pth missing; skipping tracking comparison.")
        results["DQN"] = np.nan

    print(f"\n================ BENCHMARK RESULTS (Zipfian Workload) ================")
    for policy_name, hr in results.items():
        print(f" Policy: {policy_name:<10} | Target Hit Rate: {f'{hr*100:.2f}%' if not np.isnan(hr) else 'N/A'}")
    print("======================================================================")
    
    with open("logs/benchmark_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Policy", "HitRate"])
        for k, v in results.items():
            w.writerow([k, v])

if __name__ == "__main__":
    run_benchmark()