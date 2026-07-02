# MemAI

## Overview

MemAI is a Reinforcement Learning based cache replacement system built using Deep Q Networks (DQN). It learns cache eviction policies and compares them against classical algorithms such as FIFO, LRU, and CLOCK.

---

## Features

- Deep Q Network (PyTorch)
- Custom Gymnasium Environment
- Experience Replay
- Epsilon-Greedy Exploration
- Workload Generator
- FIFO
- LRU
- CLOCK
- Automatic Benchmarking
- Training Logs
- Performance Graphs

---

## Project Structure

(project tree)

---

## Installation

pip install -r requirements.txt

---

## Training

python -m agent.train

---

## Benchmark

python -m metrics.benchmark

---

## Results

(Add the generated graphs here)

---

## Future Work

- Double DQN
- Prioritized Experience Replay
- Transformer-based State Encoder
- Real Memory Traces
