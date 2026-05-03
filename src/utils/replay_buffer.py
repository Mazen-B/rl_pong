from __future__ import annotations

import torch
import numpy as np

from collections import namedtuple

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], device: torch.device | str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        # state and next_state come from ImageToPyTorch already in [0, 1] float32,
        # so a direct cast is sufficient — no clip needed.
        self.states[self.ptr] = (state * 255.0).astype(np.uint8)
        self.next_states[self.ptr] = (next_state * 255.0).astype(np.uint8)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Transition:
        idxs = np.random.randint(0, self.size, size=batch_size)
        states = torch.from_numpy(self.states[idxs]).float().to(self.device) / 255.0
        next_states = torch.from_numpy(self.next_states[idxs]).float().to(self.device) / 255.0
        actions = torch.from_numpy(self.actions[idxs]).long().to(self.device)
        rewards = torch.from_numpy(self.rewards[idxs]).to(self.device)
        dones = torch.from_numpy(self.dones[idxs]).to(self.device)
        return Transition(states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        return self.size
