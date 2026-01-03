import json
import os

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import AgentID

from chicks import NiceChickens


class NiceChickensCallback(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker,
        base_env: BaseEnv,
        policies: dict[Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):

        # Actual PettingZoo env
        env = base_env.get_sub_environments()[env_index]

        final_scores = list(env.scores.values())

        paired_actions = episode.user_data.get("paired_actions", [])
        if paired_actions:
            coop_rate = sum(a == 0 for a in paired_actions) / len(paired_actions)
        else:
            coop_rate = 0.0

        scores_sorted = sorted(final_scores)
        n = len(scores_sorted)
        if n > 0 and sum(scores_sorted) > 0:
            index = np.arange(1, n + 1)
            gini = np.sum((2 * index - n - 1) * scores_sorted) / (n * sum(scores_sorted))
        else:
            gini = 0.0

        max_score = max(final_scores)
        num_winners = sum(1 for s in final_scores if s >= env.smax)

        episode.custom_metrics["cooperation_rate"] = coop_rate
        episode.custom_metrics["final_gini"] = gini
        episode.custom_metrics["winner_score"] = max_score
        episode.custom_metrics["num_winners"] = num_winners
        episode.custom_metrics["tie"] = 1 if num_winners > 1 else 0

    def on_episode_step(
        self,
        *,
        worker,
        base_env: BaseEnv,
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        env = base_env.get_sub_environments()[env_index]
        actions = episode.last_info_for().get("actions", {})
        paired_this_step = []
        for occ in env.resource_occupants.values():
            if len(occ) == 2:
                a1, a2 = occ
                act1, act2 = actions.get(a1), actions.get(a2)
                if act1 is not None and act2 is not None:
                    paired_this_step.extend([act1, act2])
        if "paired_actions" not in episode.user_data:
            episode.user_data["paired_actions"] = []
        episode.user_data["paired_actions"].extend(paired_this_step)

def train_marl(
    num_iterations=100,
    num_env_runners=4,
    num_chickens=6,
    smax=100.0,
    v=10.0,
    c=2.0,
    alpha=1.0,
    beta1=1.0,
    beta2=1.0,
    empathy_as_reward=False,
    wandb_project="NiceChickens",
    checkpoint_dir="./checkpoints"
):
    def env_creator(_):
        return ParallelPettingZooEnv(
            NiceChickens(
                num_chickens=num_chickens,
                smax=smax,
                v=v,
                c=c,
                alpha=alpha,
                beta1=beta1,
                beta2=beta2,
                empathy_as_reward=empathy_as_reward
            )
        )

        register_env("NiceChickens", env_creator)

        if wandb_project:
            wandb.init(project=wandb_project, config=locals())

        config = (
            PPOConfig()
            .environment("NiceChickens")
            .env_runners(num_env_runners=num_env_runners, rollout_fragment_length=100)
            .training(lr=3e-4, train_batch_size=4000)
            .multi_agent(
                policies={"shared_policy": (None, None, None, {})},
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
            )
            .callbacks(NiceChickensCallback)
            .framework("torch")
        )

        algo = config.build()

        log_file = os.path.join(checkpoint_dir, "training_metrics.json")
        training_log = []

        for i in range(num_iterations):
            result = algo.train()
            print(f"Iter {i+1}/{num_iterations}:")
            print(f"  Reward: {result['episode_reward_mean']:.2f}")
            print(f"  Length: {result['episode_len_mean']:.1f}")
            print(f"  Coop rate: {result['custom_metrics']['cooperation_rate_mean']:.3f}")
            print(f"  Gini: {result['custom_metrics']['final_gini_mean']:.3f}")
            print(f"  Winner score: {result['custom_metrics']['winner_score_mean']:.1f}")
            print(f"  Tie rate: {result['custom_metrics']['tie_mean']:.3f}")

            training_log.append({
                "iteration": i + 1,
                "episode_reward_mean": result['episode_reward_mean'],
                "episode_len_mean": result['episode_len_mean'],
                "cooperation_rate": result['custom_metrics']['cooperation_rate_mean'],
                "final_gini": result['custom_metrics']['final_gini_mean'],
                "winner_score": result['custom_metrics']['winner_score_mean'],
                "tie_rate": result['custom_metrics']['tie_mean'],
            })

            if wandb_project:
                wandb.log({
                    "episode_reward_mean": result['episode_reward_mean'],
                    "episode_len_mean": result['episode_len_mean'],
                    "cooperation_rate": result['custom_metrics']['cooperation_rate_mean'],
                    "final_gini": result['custom_metrics']['final_gini_mean'],
                    "winner_score": result['custom_metrics']['winner_score_mean'],
                    "tie_rate": result['custom_metrics']['tie_mean'],
                    "iteration": i + 1
                })

        checkpoint = algo.save(checkpoint_dir)
        print(f"Checkpoint saved: {checkpoint}")

        with open(log_file, "w") as f:
            json.dump(training_log, f, indent=2)
        print(f"Training metrics saved to {log_file}")
        
        algo.stop()
        ray.shutdown()

