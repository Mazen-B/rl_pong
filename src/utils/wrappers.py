from __future__ import annotations

import cv2
import numpy as np
import gymnasium as gym

from gymnasium import spaces
from collections import deque

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(int(noops)):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if terminated or truncated:
                # episode ended early; fill both slots with the latest obs so the
                # max projection does not mix in stale frames from a previous episode.
                self._obs_buffer[0] = obs
                self._obs_buffer[1] = obs
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

class EpisodicLifeEnv(gym.Wrapper):
    """Treat life loss as done. For Pong this is effectively a no-op because ALE Pong reports 0 lives."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info

class FireResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        assert meanings[1] == "FIRE"
        assert len(meanings) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        _, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized[:, :, None]

class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward):
        return float(np.sign(reward))

class FrameStack(gym.Wrapper):
    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames: deque[np.ndarray] = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames), axis=-1)

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32)

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1)).astype(np.float32) / 255.0

def make_raw_atari(env_id: str, render_mode: str | None = None) -> gym.Env:
    # Best path: use PongNoFrameskip-v4. If someone passes ALE/*-v5, disable v5's built-in skip/sticky actions.
    if env_id.startswith("ALE/") and env_id.endswith("-v5"):
        return gym.make(env_id, frameskip=1, repeat_action_probability=0.0, render_mode=render_mode)
    return gym.make(env_id, render_mode=render_mode)

def make_atari_env(env_id: str = "PongNoFrameskip-v4",
    seed: int = 0, clip_rewards: bool = True,
    episodic_life: bool = True, pytorch_format: bool = True,
    render_mode: str | None = None) -> gym.Env:
    env = make_raw_atari(env_id, render_mode=render_mode)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = FrameStack(env, k=4)
    if pytorch_format:
        env = ImageToPyTorch(env)
    return env
