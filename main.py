import subprocess
import sys
import time

steps = [
    ("Training MemAI", ["-m", "agent.train"]),
    ("Running Benchmark", ["-m", "metrics.benchmark"]),
    ("Generating Training Graphs", ["plots/generate_plots.py"]),
    ("Generating Benchmark Graph", ["plots/benchmark_plot.py"]),
]

print("=" * 60)
print("MemAI Complete Pipeline")
print("=" * 60)

start = time.time()

for title, command in steps:

    print(f"\n>>> {title}")

    result = subprocess.run([sys.executable] + command)

    if result.returncode != 0:
        print(f"\n[FAILED] {title} Failed")
        sys.exit(1)

    print(f"[SUCCESS] {title} Completed")

print("\n" + "=" * 60)
print("Everything Finished Successfully!")
print("=" * 60)

print(f"\nTotal Time : {time.time()-start:.2f} seconds")