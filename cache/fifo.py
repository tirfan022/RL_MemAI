# cache/fifo.py

from collections import deque
from cache.base import CacheBase


class FIFO(CacheBase):

    def __init__(self, capacity):
        super().__init__(capacity)

        self.queue = deque()
        self.cache = set()

    def access(self, page):

        # HIT
        if page in self.cache:
            self.hits += 1
            return True

        # MISS
        self.misses += 1

        # Cache Full
        if len(self.cache) == self.capacity:

            old = self.queue.popleft()

            self.cache.remove(old)

            self.evictions += 1

        self.queue.append(page)

        self.cache.add(page)

        return False

    def reset(self):

        self.queue.clear()

        self.cache.clear()

        self.hits = 0
        self.misses = 0
        self.evictions = 0