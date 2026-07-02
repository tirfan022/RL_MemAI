import csv
import os


class Evaluator:

    def __init__(self):
        self.results = {}

    def add_result(self, name, hits, misses):

        total = hits + misses
        hit_rate = hits / total if total else 0

        self.results[name] = {
            "Hits": hits,
            "Misses": misses,
            "Hit Rate": hit_rate
        }

    def print_results(self):

        print("\n==============================")
        print("Benchmark Results")
        print("==============================")

        os.makedirs("logs", exist_ok=True)

        with open("logs/benchmark_results.csv", "w", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                "Algorithm",
                "Hits",
                "Misses",
                "HitRate"
            ])

            for name, result in self.results.items():

                print(
                    f"{name:10s}"
                    f" Hits={result['Hits']:4d}"
                    f" Misses={result['Misses']:4d}"
                    f" HitRate={result['Hit Rate']:.2%}"
                )

                writer.writerow([
                    name,
                    result["Hits"],
                    result["Misses"],
                    result["Hit Rate"] * 100
                ])