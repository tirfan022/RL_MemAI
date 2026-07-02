import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("logs/training_log.csv")

# Reward
plt.figure(figsize=(8,5))
plt.plot(df["Episode"], df["Reward"])
plt.title("Reward vs Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.tight_layout()
plt.savefig("plots/reward_curve.png")
plt.close()

# Hit Rate
plt.figure(figsize=(8,5))
plt.plot(df["Episode"], df["HitRate"]*100)
plt.title("Hit Rate vs Episode")
plt.xlabel("Episode")
plt.ylabel("Hit Rate (%)")
plt.grid()
plt.tight_layout()
plt.savefig("plots/hitrate_curve.png")
plt.close()

# Epsilon
plt.figure(figsize=(8,5))
plt.plot(df["Episode"], df["Epsilon"])
plt.title("Exploration (Epsilon)")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.grid()
plt.tight_layout()
plt.savefig("plots/epsilon_curve.png")
plt.close()

print("Graphs Generated!")