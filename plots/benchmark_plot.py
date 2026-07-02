import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

try:
    from workload.generator import WorkloadGenerator
except Exception as e:
    print(f"[CRITICAL ERROR] Failed to import workload generator: {e}")
    sys.exit(1)

def generate_benchmark_chart():
    csv_path = "logs/benchmark_results.csv"
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] Cannot find target file: {csv_path}. Run 'python -m metrics.benchmark' first.")
        return

    try:
        # Load the saved data metrics
        df = pd.read_csv(csv_path)
        
        
        if df['HitRate'].max() <= 1.0:
            df['HitRate'] = df['HitRate'] * 100

        
        plt.figure(figsize=(9, 5))
        
        
        colors = ['#dc3545' if policy == 'DQN' else '#6c757d' for policy in df['Policy']]
        
        if 'DQN' in df['Policy'].values:
            colors = ['#007bff' if policy == 'DQN' else '#6c757d' for policy in df['Policy']]

        bars = plt.bar(df['Policy'], df['HitRate'], color=colors, edgecolor='black', width=0.5)
        
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

        plt.title("Comparative Cache Hit Rates (Hard Zipfian Workload)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Replacement Strategy Policy", fontsize=12, labelpad=10)
        plt.ylabel("Hit Rate Percentage (%)", fontsize=12, labelpad=10)
        
        
        max_rate = df['HitRate'].max()
        plt.ylim(0, min(100, max_rate + 15))
        
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        
        chart_output_path = os.path.join(output_dir, "benchmark_comparison.png")
        plt.savefig(chart_output_path, dpi=300)
        plt.close()
        
        print("\n" + "="*50)
        print(f" SUCCESS: Chart safely generated at: {chart_output_path}")
        print("="*50)

    except Exception as e:
        print(f"[RUNTIME ERROR] Plot generation failed: {e}")

if __name__ == "__main__":
    generate_benchmark_chart()