class MetricsCollector:

    def print_result(self, policy, name):

        stats = policy.get_stats()

        total = stats["Hits"] + stats["Misses"]

        hit_rate = 0

        if total != 0:
            hit_rate = stats["Hits"] / total

        print("=" * 40)
        print(f"Policy      : {name}")
        print(f"Hits        : {stats['Hits']}")
        print(f"Misses      : {stats['Misses']}")
        print(f"Evictions   : {stats['Evictions']}")
        print(f"Hit Rate    : {hit_rate:.2%}")
        print("=" * 40)