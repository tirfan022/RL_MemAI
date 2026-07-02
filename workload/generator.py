import numpy as np

class WorkloadGenerator:
    def __init__(self, num_pages=1000):
        self.num_pages = num_pages

    def locality(self, length=1000, hot_pages_count=3, hot_ratio=0.8):
        """Original workload: highly localized."""
        accesses = []
        hot_pages = list(range(hot_pages_count))
        cold_pages = list(range(hot_pages_count, self.num_pages))
        
        for _ in range(length):
            if np.random.rand() < hot_ratio:
                accesses.append(np.random.choice(hot_pages))
            else:
                accesses.append(np.random.choice(cold_pages))
        return accesses

    def zipfian(self, length=1000, alpha=1.1):
        """Harder, power-law workload where working sets bleed out of cache limits."""
        ranks = np.arange(1, self.num_pages + 1)
        probabilities = 1.0 / (ranks ** alpha)
        probabilities /= probabilities.sum()
        return list(np.random.choice(np.arange(self.num_pages), size=length, p=probabilities))

    def mixed(self, length=1000):
        """Phase-shifting workload combining locality spikes and sudden noise."""
        part1 = self.locality(length=length // 2, hot_pages_count=5, hot_ratio=0.7)
        part2 = self.zipfian(length=length - (length // 2), alpha=1.0)
        return part1 + part2