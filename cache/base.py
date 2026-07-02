from abc import ABC, abstractmethod

class CacheBase(ABC):

    def __init__(self, capacity):
        self.capacity = capacity

        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @abstractmethod
    def access(self, page):
        pass

    @abstractmethod
    def reset(self):
        pass

    def get_stats(self):
        return {
            "Hits": self.hits,
            "Misses": self.misses,
            "Evictions": self.evictions
        }