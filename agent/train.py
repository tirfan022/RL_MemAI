import os
import csv
import sys
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Ensure the root path is recognized before anything else executes
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

try:
    from cache.cache_env import CacheEnv   
    from workload.generator import WorkloadGenerator
    from agent.agent import DQNAgent
except Exception as e:
    print(f"\n[CRITICAL ERROR] Import stage failed: {e}")
    print("Please check that your 'cache' and 'workload' folders exist in the root directory.")
    sys.exit(1)

def train():
    try:
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        generator = WorkloadGenerator(num_pages=500)
        
        # Get dimensions using a quick initial environment setup
        temp_workload = generator.zipfian(length=10, alpha=1.1)
        temp_env = CacheEnv(capacity=32, workload=temp_workload)
        state_dim = temp_env.observation_space.shape[0]
        action_dim = temp_env.action_space.n
        
        agent = DQNAgent(
            state_dim=state_dim, 
            action_dim=action_dim, 
            lr=2.558234541754176e-05,
            gamma=0.8989356522040539,
            epsilon_decay=0.953460418033084,
            tau=0.0017598387368335318
        )
        agent.batch_size = 128  # from Optuna search (tune.py, 25 trials -> best_hparams.json)
        
        episodes = 150
        best_hit_rate = -1.0

        # Fixed, seeded held-out trace for evaluation. Training HitRate below is
        # measured on a *fresh random* workload every episode, so it's noisy and
        # not directly comparable episode-to-episode. EvalHitRate is measured
        # greedily (epsilon=0) on the SAME trace every time, so its trend is the
        # real signal of whether the policy is actually improving.
        eval_workload = generator.zipfian(length=1500, alpha=1.1)

        log_file = open("logs/training_log.csv", "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["Episode", "Reward", "Hits", "Misses", "HitRate", "EvalHitRate", "Epsilon", "AvgLoss"])
        
        print("\n" + "="*77)
        print(f"{'Ep':<5}{'Reward':<10}{'Hits':<8}{'Misses':<8}{'HitRate':<10}{'EvalHR':<12}{'Epsilon':<10}{'AvgLoss':<10}")
        print("="*77)

        for ep in range(1, episodes + 1):
            workload = generator.zipfian(length=1000, alpha=1.1)
            env = CacheEnv(capacity=32, workload=workload)
            state, _ = env.reset()
            
            total_reward = 0
            losses = []
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                loss_val = agent.learn()
                if loss_val > 0:
                    losses.append(loss_val)
                    agent.update_target_network()  # soft/Polyak update every learning step

            agent.decay_epsilon()
            
            # Ground-truth hit/miss counts come directly from the environment's own
            # counters (incremented inside handle_hit/handle_miss), not from reward
            # sign. Reward is a *shaped* signal (a miss with free cache space still
            # scores +1, and a full-cache miss can score up to +10 via the oracle
            # term), so "reward > 0" is NOT a valid proxy for "was a hit" and was
            # previously inflating the logged hit rate to ~97-99% versus the true
            # ~47% measured at benchmark time.
            ep_hits = env.hits
            ep_misses = env.misses

            total_accesses = ep_hits + ep_misses
            hit_rate = (ep_hits / total_accesses) if total_accesses > 0 else 0.0
            avg_loss = np.mean(losses) if losses else 0.0

            # Greedy evaluation on the fixed held-out trace (every 5 episodes to
            # keep runtime reasonable) -- this is the number that should trend
            # upward, unlike the noisy per-episode training HitRate.
            eval_hit_rate = ""
            if ep % 5 == 0 or ep == 1:
                eval_env = CacheEnv(capacity=32, workload=eval_workload)
                eval_state, _ = eval_env.reset()
                eval_done = False
                while not eval_done:
                    eval_action = agent.act(eval_state, eval_mode=True)
                    eval_state, _, eval_term, eval_trunc, _ = eval_env.step(eval_action)
                    eval_done = eval_term or eval_trunc
                eval_hit_rate = eval_env.hits / (eval_env.hits + eval_env.misses)

                if eval_hit_rate > best_hit_rate:
                    best_hit_rate = eval_hit_rate
                    torch.save(agent.model.state_dict(), "models/best_model.pth")

            log_writer.writerow([ep, total_reward, ep_hits, ep_misses, f"{hit_rate:.4f}",
                                  f"{eval_hit_rate:.4f}" if eval_hit_rate != "" else "",
                                  f"{agent.epsilon:.4f}", f"{avg_loss:.4f}"])
            log_file.flush()
            
            if ep % 5 == 0 or ep == 1:
                eval_str = f"{eval_hit_rate:.4f}" if eval_hit_rate != "" else "  -   "
                print(f"{ep:<5}{total_reward:<10.1f}{ep_hits:<8}{ep_misses:<8}{hit_rate:<10.4f}{eval_str:<12}{agent.epsilon:<10.3f}{avg_loss:<10.4f}")
                
        torch.save(agent.model.state_dict(), "models/final_model.pth")
        log_file.close()
        print("\nTraining execution complete. Models safely cached.")
        
    except Exception as e:
        print(f"\n[RUNTIME ERROR] Training loop crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train()