from cache.base import CacheBase

class Clock(CacheBase):

    def __init__(self, capacity):
        super().__init__(capacity)

        self.frames = [None] * capacity
        self.reference = [0] * capacity
        self.page_to_index = {}

        self.hand = 0

    def access(self, page):
        #hit
        if page in self.page_to_index:
            self.hits += 1

            index = self.page_to_index[page]
            self.reference[index] = 1

            return True
        
        #miss

        self.misses += 1

        #empty frame 
        if len(self.page_to_index) < self.capacity:

            index = len(self.page_to_index)

            self.frames[index] = page
            self.reference[index] = 1
            self.page_to_index[page] = index

            return False

        # cache full
        while True:

            if self.reference[self.hand] == 0:

                old_page = self.frames[self.hand]

                del self.page_to_index[old_page]

                self.frames[self.hand] = page
                self.reference[self.hand] = 1
                self.page_to_index[page] = self.hand

                self.evictions += 1

                self.hand = (self.hand + 1) % self.capacity

                return False

            else:

                self.reference[self.hand] = 0
                self.hand = (self.hand + 1) % self.capacity

    def reset(self):

        self.frames = [None] * self.capacity
        self.reference = [0] * self.capacity
        self.page_to_index.clear()

        self.hand = 0

        self.hits = 0
        self.misses = 0
        self.evictions = 0