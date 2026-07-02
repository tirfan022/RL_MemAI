from collections import OrderedDict
from cache.base import CacheBase


class LRU(CacheBase):

    def __init__(self, capacity):

        super().__init__(capacity)

        self.cache = OrderedDict()

    def access(self, page):

        if page in self.cache:

            self.hits += 1

            self.cache.move_to_end(page)

            return True

        self.misses += 1

        if len(self.cache) == self.capacity:

            self.cache.popitem(last=False)

            self.evictions += 1

        self.cache[page] = True

        return False

    def reset(self):

        self.cache.clear()

        self.hits = 0
        self.misses = 0
        self.evictions = 0