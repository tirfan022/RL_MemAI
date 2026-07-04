import os
import sys
import json
import random
import argparse

import numpy as np
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from cache.cache_env import CacheEnv
from workload.generator import WorkloadGenerator
from agent.agent import DQNAgent

CAPACITY = 32
EPISODE_LENGTH = 1000
EVAL_EVERY = 5

# One fixed set of workloads for ALL trials, so differences in score reflect
# hyperparameters, not lucky/unlucky random traces.
_gen = WorkloadGenerator(num_pages=500)
random.seed(123)
np.random.seed(123)
TRAIN_WORKLOADS = None  # populated in main() once --episodes is known
EVAL_WORKLOAD = _gen.zipfian(length=1500, alpha=1.1)


def evaluate(agent, workload):
    env = CacheEnv(capacity=CAPACITY, workload=workload)
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.act(state, eval_mode=True)
        state, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return env.hits / (env.hits + env.misses)


def objective(trial):
    gamma = trial.suggest_float("gamma", 0.80, 0.99)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    tau = trial.suggest_float("tau", 0.001, 0.05, log=True)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.94, 0.995)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    temp_env = CacheEnv(capacity=CAPACITY, workload=TRAIN_WORKLOADS[0][:10])
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim,
                      lr=lr, gamma=gamma, epsilon_decay=epsilon_decay, tau=tau)
    agent.batch_size = batch_size

    best_eval = 0.0
    for ep, workload in enumerate(TRAIN_WORKLOADS, start=1):
        env = CacheEnv(capacity=CAPACITY, workload=workload)
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            loss = agent.learn()
            if loss > 0:
                agent.update_target_network()
        agent.decay_epsilon()

        if ep % EVAL_EVERY == 0 or ep == len(TRAIN_WORKLOADS):
            eval_hr = evaluate(agent, EVAL_WORKLOAD)
            best_eval = max(best_eval, eval_hr)
            trial.report(eval_hr, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_eval


def main():
    global TRAIN_WORKLOADS

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=40,
                         help="episodes per trial (shorter than the full 150-episode run, just for search)")
    args = parser.parse_args()

    TRAIN_WORKLOADS = [_gen.zipfian(length=EPISODE_LENGTH, alpha=1.1) for _ in range(args.episodes)]

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3 * EVAL_EVERY),
    )
    study.optimize(objective, n_trials=args.trials)

    print("\n" + "=" * 60)
    print(f"Best eval hit rate : {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")
    print("=" * 60)

    with open("best_hparams.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("\nSaved to best_hparams.json — plug these into the DQNAgent(...) "
          "call in agent/train.py for your full 150-episode run.")


if __name__ == "__main__":
    main()