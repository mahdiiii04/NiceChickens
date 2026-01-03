import json
import os
import random
from collections import defaultdict

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch as th
from pettingzoo.utils import parallel_to_aec
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from supersuit import pad_observations_v0, pad_action_space_v0

# Assuming your environment class is in a file called chicks.py
from chicks import NiceChickens

def make_env():
    parallel_env = NiceChickens(
        num_chickens=num_chickens,
        smax=smax,
        v=v,
        c=c,
        alpha=alpha,
        beta1=beta1,
        beta2=beta2,
        empathy_as_reward=empathy_as_reward,
    )
    # No need for parallel_to_aec or padding anymore
    # Directly wrap the ParallelEnv
    wrapped = NiceChickensWrapper(parallel_env)
    return wrapped

class NiceChickensWrapper(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.possible_agents = env.possible_agents
        self.agents = env.agents

        # Single flat observation space
        single_obs_space = env.observation_space(env.agents[0])
        self.observation_space = spaces.Box(
            low=np.tile(single_obs_space.low, len(self.agents)),
            high=np.tile(single_obs_space.high, len(self.agents)),
            dtype=np.float32,
        )

        # MultiDiscrete action space (one Discrete(2) per agent)
        self.action_space = spaces.MultiDiscrete([2] * len(self.agents))

        self.paired_actions = []

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed)
        self.agents = self.env.agents
        self.paired_actions = []  # <-- Critical: reset here!
        return self._get_concat_obs(observations), infos

    def step(self, actions):  # actions: np.array of shape (num_agents,)
        action_dict = {agent: int(actions[i]) for i, agent in enumerate(self.agents)}
        observations, rewards, terminations, truncations, infos = self.env.step(action_dict)

        # Track paired actions for metrics
        paired_this_step = []
        for occ in self.env.resource_occupants.values():
            if len(occ) == 2:
                a1, a2 = occ
                act1, act2 = action_dict.get(a1), action_dict.get(a2)
                if act1 is not None and act2 is not None:
                    paired_this_step.extend([act1, act2])
        self.paired_actions.extend(paired_this_step)

        done = all(terminations.values()) or all(truncations.values())
        reward = np.mean(list(rewards.values()))  # shared policy → average reward

        return self._get_concat_obs(observations), reward, done, False, infos

    def _get_concat_obs(self, obs_dict):
        if not obs_dict:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return np.concatenate([obs_dict[agent] for agent in self.agents])

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.env_ref = None
        self.metrics_log = []  # ← Re-added to collect history for JSON save

    def _on_training_start(self) -> None:
        self.env_ref = self.training_env.envs[0]  # NiceChickensWrapper

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.env_ref is None:
            return

        env = self.env_ref.env  # Original NiceChickens

        final_scores = list(env.scores.values())
        paired_actions = getattr(self.env_ref, "paired_actions", [])
        coop_rate = (sum(a == 0 for a in paired_actions) / len(paired_actions)) if paired_actions else 0.0

        scores_sorted = sorted(final_scores)
        n = len(scores_sorted)
        if n > 0 and sum(scores_sorted) > 0:
            index = np.arange(1, n + 1)
            gini = np.sum((2 * index - n - 1) * scores_sorted) / (n * sum(scores_sorted))
        else:
            gini = 0.0

        max_score = max(final_scores) if final_scores else 0.0
        num_winners = sum(1 for s in final_scores if s >= env.smax)
        tie = 1 if num_winners > 1 else 0

        # Approximate episode reward from SB3 logger
        episode_reward = self.logger.name_to_value.get("train/episode_reward", 0.0) if self.logger else 0.0

        metrics = {
            "timesteps": self.num_timesteps,
            "cooperation_rate": coop_rate,
            "final_gini": gini,
            "winner_score": max_score,
            "num_winners": num_winners,
            "tie": tie,
            "episode_reward_mean": episode_reward,
        }

        self.metrics_log.append(metrics)

        print(f"\n=== Episode Metrics (at {self.num_timesteps:,} steps) ===")
        print(f"Cooperation rate: {coop_rate:.3f}")
        print(f"Gini coefficient: {gini:.3f}")
        print(f"Winner score: {max_score:.1f}")
        print(f"Winners: {num_winners} | Tie: {tie}")
        print(f"Episode reward (avg): {episode_reward:.2f}")
        print("=" * 50)

        # TensorBoard logging
        if self.logger:
            self.logger.record("metrics/cooperation_rate", coop_rate)
            self.logger.record("metrics/final_gini", gini)
            self.logger.record("metrics/winner_score", max_score)
            self.logger.record("metrics/num_winners", num_winners)
            self.logger.record("metrics/tie", tie)

        # Reset for next episode
        self.env_ref.paired_actions = []
        
def train_marl(
    num_timesteps=500_000,
    num_chickens=6,
    smax=100.0,
    v=10.0,
    c=2.0,
    alpha=1.0,
    beta1=1.0,
    beta2=1.0,
    empathy_as_reward=False,
    log_dir="./sb3_logs",
    checkpoint_dir="./sb3_checkpoints",
):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    def make_env():
        parallel_env = NiceChickens(
            num_chickens=num_chickens,
            smax=smax,
            v=v,
            c=c,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            empathy_as_reward=empathy_as_reward,
        )
        # No need for parallel_to_aec or padding anymore
        # Directly wrap the ParallelEnv
        wrapped = NiceChickensWrapper(parallel_env)
        return wrapped

    vec_env = DummyVecEnv([make_env])
    vec_env = VecMonitor(vec_env, filename=os.path.join(log_dir, "monitor.csv"))

    policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[256, 256])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=256,
        n_steps=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
    )

    callback = MetricsCallback()

    model.learn(total_timesteps=num_timesteps, callback=callback)

    # Save model
    model_path = os.path.join(checkpoint_dir, "final_model")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save custom metrics
    metrics_path = os.path.join(log_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(callback.metrics_log, f, indent=2)
    print(f"Training metrics saved to {metrics_path}")


if __name__ == "__main__":
    train_marl(
        num_timesteps=1_000_000,
        num_chickens=6,
        smax=100.0,
        v=10.0,
        c=2.0,
        alpha=1.0,
        beta1=1.0,
        beta2=1.0,
        empathy_as_reward=False,
    )