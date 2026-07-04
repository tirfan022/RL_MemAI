import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CacheEnv(gym.Env):

    def __init__(self, workload, capacity=32):

        super().__init__()

        # Workload (page access sequence)
        self.workload = workload

        # Maximum cache size
        self.capacity = capacity

        # Current cache
        self.cache = []

        # Current position in workload
        self.index = 0

        # Current page request
        self.current_page = None

        # Current timestamp
        self.time = 0

        # Frequency of every page
        self.frequency = {}

        # Last access time of every page
        self.last_used = {}

        # Statistics
        self.hits = 0
        self.misses = 0

        # RL Action Space
        # Action = cache slot to evict
        self.action_space = spaces.Discrete(capacity)

        # State:
        # cache contents
        # +
        # current page
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(capacity*3 + 1,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Empty cache
        self.cache = []

        # Start from beginning of workload
        self.index = 0

        # Reset time
        self.time = 0

        # Clear statistics
        self.frequency = {}
        self.last_used = {}

        self.hits = 0
        self.misses = 0

        # First page in workload
        self.current_page = self.workload[self.index]

        # Return initial state
        return self.get_state(), {}
    
    
    def handle_hit(self):

        # Reward for cache hit (scaled down from 10 -> keeps Q-value
        # magnitudes small/stable for the network; see oracle_reward too)
        reward = 1.0

        self.hits += 1

        # Update frequency of current page
        if self.current_page in self.frequency:
            self.frequency[self.current_page] += 1
        else:
            self.frequency[self.current_page] = 1

        # Update last access time
        self.last_used[self.current_page] = self.time

        # Increase time
        self.time += 1

        return reward
    
    def handle_miss(self, action):

        # ----------------------------
        # Case 1 : Cache has free space
        # ----------------------------
        self.misses += 1

        if len(self.cache) < self.capacity:

            self.cache.append(self.current_page)

            reward = 0.1

        # ----------------------------
        # Case 2 : Cache Full
        # ----------------------------
        else:
            # Calculate reward BEFORE replacing
            reward = self.oracle_reward(action)

            # Replace the page
            self.cache[action] = self.current_page

        # ----------------------------
        # Update frequency
        # ----------------------------
        self.frequency[self.current_page] = (
            self.frequency.get(self.current_page, 0) + 1
        )

        # ----------------------------
        # Update last used time
        # ----------------------------
        self.last_used[self.current_page] = self.time

        self.time += 1

        return reward
    
    def step(self, action):

        # -----------------------------
        # Check Hit or Miss
        # -----------------------------
        if self.current_page in self.cache:

            reward = self.handle_hit()

        else:

            reward = self.handle_miss(action)

        # -----------------------------
        # Move to next page
        # -----------------------------
        self.index += 1

        done = self.index >= len(self.workload)

        # -----------------------------
        # Update current page
        # -----------------------------
        if not done:
            self.current_page = self.workload[self.index]
        else:
            self.current_page = -1

        # -----------------------------
        # Return
        # -----------------------------
        next_state = self.get_state()

        info = {
            "cache": self.cache.copy(),
            "current_page": self.current_page if not done else None,
            "time": self.time
        }

        return next_state, reward, done, False, info
    
    def next_use_distance(self, page):

        for i in range(self.index + 1, len(self.workload)):

            if self.workload[i] == page:
                return i - self.index

        return float("inf")
    
    def find_best_page_to_evict(self):

        """
        Oracle:
        Returns the cache slot that should be
        evicted according to Belady's algorithm.
        """

        best_slot = 0

        farthest_distance = -1

        for i, page in enumerate(self.cache):

            distance = self.next_use_distance(page)

            if distance > farthest_distance:

                farthest_distance = distance

                best_slot = i

        return best_slot
    
    def oracle_reward(self, action):

        evicted_page = self.cache[action]

        distance = self.next_use_distance(evicted_page)

        if distance == float("inf"):
            return 1.0

        reward = distance - 5

        reward = max(-10, min(10, reward))

        # Scaled to roughly match the +1 hit reward's magnitude (was -10..+10)
        return reward / 10.0
    
    def get_state(self):

        state = []

        # Normalization constants. page_id is a categorical index, not a
        # meaningful ordinal quantity -- feeding it in raw (0..num_pages)
        # alongside small frequency/recency values lets it dominate a
        # randomly-initialized network's input norm and slows learning.
        page_scale = float(max(self.workload)) + 1.0 if len(self.workload) > 0 else 1.0
        recency_scale = 1000.0  # ~episode length; keeps recency in a small range
        freq_scale = 50.0       # typical max access count for hot pages

        # Store Page ID, Frequency and Recency (all normalized to ~[0,1]-ish ranges)
        for page in self.cache:

            # Page ID
            state.append(page / page_scale)

            # Frequency
            state.append(self.frequency.get(page, 0) / freq_scale)

            # Recency
            if page in self.last_used:
                recency = self.time - self.last_used[page]
            else:
                recency = 0

            state.append(recency / recency_scale)

        # Fill remaining cache slots (empty slot marker stays -1 for page id)
        while len(state) < self.capacity * 3:
            state.extend([-1.0 / page_scale, 0.0, 0.0])

        # Current requested page
        state.append(self.current_page / page_scale if self.current_page is not None and self.current_page >= 0 else -1.0 / page_scale)

        return np.array(state, dtype=np.float32)

    def get_stats(self):

        total = self.hits + self.misses

        hit_rate = 0.0

        if total > 0:
            hit_rate = self.hits / total

        return {
            "Hits": self.hits,
            "Misses": self.misses,
            "Hit Rate": hit_rate
        }