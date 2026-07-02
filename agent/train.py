import os
import csv
import sys
import numpy as np
import torch

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
            lr=3e-4, 
            epsilon_decay=0.965
        )
        
        episodes = 150
        best_hit_rate = -1.0
        
        log_file = open("logs/training_log.csv", "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["Episode", "Reward", "Hits", "Misses", "HitRate", "Epsilon", "AvgLoss"])
        
        print("\n" + "="*65)
        print(f"{'Ep':<5}{'Reward':<10}{'Hits':<8}{'Misses':<8}{'HitRate':<10}{'Epsilon':<10}{'AvgLoss':<10}")
        print("="*65)

        for ep in range(1, episodes + 1):
            workload = generator.zipfian(length=1000, alpha=1.1)
            env = CacheEnv(capacity=32, workload=workload)
            state, _ = env.reset()
            
            total_reward = 0
            losses = []
            done = False
            
            # Direct tracking variables inside the loop
            ep_hits = 0
            ep_misses = 0
            
            while not done:
                action = agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Safely capture hit/miss status from the info dict or environment attributes
                is_hit = False
                if isinstance(info, dict):
                    if info.get("hit") is True or info.get("result") == "hit":
                        is_hit = True
                    elif "hit" in str(info).lower() and info.get("hit") not in [False, None]:
                        is_hit = True
                
                # Alternative fallback: if your environment uses standard positive rewards for hits
                if not is_hit and reward > 0:
                    is_hit = True
                
                if is_hit:
                    ep_hits += 1
                else:
                    ep_misses += 1
                
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                loss_val = agent.learn()
                if loss_val > 0:
                    losses.append(loss_val)
                    
            agent.decay_epsilon()
            
            if ep % 10 == 0:
                agent.update_target_network()
            
            # Double check fallback using potential direct environment attributes if loop counts stayed 0
            if ep_hits == 0 and ep_misses == 0:
                ep_hits = getattr(env, "hits", getattr(env, "hit_count", 0))
                ep_misses = getattr(env, "misses", getattr(env, "miss_count", 0))
                
            total_accesses = ep_hits + ep_misses
            hit_rate = (ep_hits / total_accesses) if total_accesses > 0 else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
            
            log_writer.writerow([ep, total_reward, ep_hits, ep_misses, f"{hit_rate:.4f}", f"{agent.epsilon:.4f}", f"{avg_loss:.4f}"])
            log_file.flush()
            
            if ep % 5 == 0 or ep == 1:
                print(f"{ep:<5}{total_reward:<10.1f}{ep_hits:<8}{ep_misses:<8}{hit_rate:<10.4f}{agent.epsilon:<10.3f}{avg_loss:<10.4f}")
                
            if hit_rate > best_hit_rate and hit_rate > 0:
                best_hit_rate = hit_rate
                torch.save(agent.model.state_dict(), "models/best_model.pth")
                
        torch.save(agent.model.state_dict(), "models/final_model.pth")
        log_file.close()
        print("\nTraining execution complete. Models safely cached.")
        
    except Exception as e:
        print(f"\n[RUNTIME ERROR] Training loop crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train()