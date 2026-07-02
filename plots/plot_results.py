import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Create output directory
# -----------------------------
os.makedirs("plots", exist_ok=True)

# -----------------------------
# Read training log
# -----------------------------
df = pd.read_csv("logs/training_log.csv")

# -----------------------------
# Reward vs Episode
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(df["Episode"], df["Reward"], linewidth=2)
plt.title("Reward vs Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)

plt.savefig("plots/reward_curve.png", dpi=300)
plt.close()

# -----------------------------
# Hit Rate vs Episode
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(df["Episode"], df["HitRate"]*100, linewidth=2)
plt.title("Hit Rate vs Episode")
plt.xlabel("Episode")
plt.ylabel("Hit Rate (%)")
plt.grid(True)

plt.savefig("plots/hitrate_curve.png", dpi=300)
plt.close()

# -----------------------------
# Epsilon Decay
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(df["Episode"], df["Epsilon"], linewidth=2)
plt.title("Epsilon Decay")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.grid(True)

plt.savefig("plots/epsilon_curve.png", dpi=300)
plt.close()

print("Graphs saved in plots/")