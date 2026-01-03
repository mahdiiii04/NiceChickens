import random
import numpy as np
from collections import defaultdict

from gymnasium import spaces
from pettingzoo import ParallelEnv

class NiceChickens(ParallelEnv):
    metadata = {
        "name" : "nice_chickens_v0",
    }

    def __init__(self, num_chickens=5, smax=100.0, v=10.0, c=2.0, alpha=1.0, beta1=1.0, beta2=1.0, epsilon=1e-5, empathy_as_reward=False):
        super().__init__()

        self._num_chickens = int(num_chickens)
        self.smax = float(smax)
        self.v = float(v)
        self.c = float(c)
        self.alpha = float(alpha)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.EAR = bool(empathy_as_reward)

        self.possible_agents = [f"chicken_{i}" for i in range(self._num_chickens)]
        self.agents = self.possible_agents.copy() 

        self.agent_positions = {agent: None for agent in self.agents}
        self.agent_indexes = {agent: i for i, agent in enumerate(self.agents)}
        self.scores = None
        self.resource_occupants = defaultdict(list)
        self.num_resources = None
        self.resources = None

    def reset_resources(self):
        self.num_resources = random.randint(int(np.ceil(self._num_chickens / 2)), self._num_chickens)
        self.resources = [i for i in range(self.num_resources)] + [i for i in range(self.num_resources)]
        self.resource_occupants = defaultdict(list)

    def assign_resources(self):
        for agent in self.agents:
            self.agent_positions[agent] = self.resources.pop(random.randint(0, len(self.resources) - 1))
            self.resource_occupants[self.agent_positions[agent]].append(agent)

    def _get_obs(self, agent):

        scores = np.array(
            [self.scores[a] for a in self.agents],
            dtype=np.float32
        ) / self.smax

        self_norm_idx = self.agent_indexes[agent] / max(1, self._num_chickens - 1)

        occupants = self.resource_occupants[self.agent_positions[agent]]

        if len(occupants) == 1:
            opp_norm_idx = -1
        else:
            opponent = occupants[1] if occupants[0] == agent else occupants[0]
            opp_norm_idx = self.agent_indexes[opponent] / max(1, self._num_chickens - 1)
        
        res_occ = np.array([self_norm_idx, opp_norm_idx], dtype=np.float32)
            
        return (scores, res_occ)
    
    def observation_space(self, agent):
        return spaces.Tuple(
            spaces.Box(low=0.0, high=1.0, shape=(self._num_chickens,), dtype=np.float32),
            spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        )
    
    def action_space(self, agent):
        return spaces.Discrete(2)
    
    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)

        self.current_step = 0

        self.reset_resources()
        self.scores = {agent: 0.0 for agent in self.agents}
        self.assign_resources()

        observations = {agent : self._get_obs(agent) for agent in self.agents}
        infos = {agent : {} for agent in self.agents}

        return observations, infos
    
    def step(self, actions):
        
        for agent in self.agents:
            occupants = self.resource_occupants[self.agent_positions[agent]]
            if len(occupants) == 2:
                assert agent in actions, f"Missing action for {agent}"
                assert self.action_space(agent).contains(actions[agent])

        vanilla_outcomes = {agent: 0.0 for agent in self.agents}
        rewards = {agent: 0.0 for agent in self.agents}

        self.current_step += 1

        for occupation in self.resource_occupants.values():

            if len(occupation) == 0: # Empty ressource
                continue

            elif len(occupation) == 1:  # Alone in resource
                chicken = occupation[0]
                vanilla_outcomes[chicken] = self.v
                rewards[chicken] = self.v + self.alpha * (self.scores[chicken] / self.smax)

            elif len(occupation) == 2:
                chicken_1 = occupation[0]
                chicken_2 = occupation[1]

                # Cooperate -> 0 / Defect -> 1

                # classic chicken game outcomes
                if actions[chicken_1] == 0 and actions[chicken_2] == 0:
                    vanilla_outcomes[chicken_1], vanilla_outcomes[chicken_2] = self.v / 2, self.v / 2
                elif actions[chicken_1] == 1 and actions[chicken_2] == 0:
                    vanilla_outcomes[chicken_1], vanilla_outcomes[chicken_2] = self.v, 0.0
                elif actions[chicken_1] == 0 and actions[chicken_2] == 1:
                    vanilla_outcomes[chicken_1], vanilla_outcomes[chicken_2] = 0.0, self.v
                else:
                    vanilla_outcomes[chicken_1], vanilla_outcomes[chicken_2] = (self.v - self.c) / 2, (self.v - self.c) / 2

                # adding motivation incentive
                reward_1 = vanilla_outcomes[chicken_1] + self.alpha * (self.scores[chicken_1] / self.smax)
                reward_2 = vanilla_outcomes[chicken_2] + self.alpha * (self.scores[chicken_2] / self.smax)
            
                # empathy sign
                emp_1_2, emp_2_1 = 1 if actions[chicken_1] == 0 else -1, 1 if actions[chicken_2] == 0 else -1

                # normalization denominators
                normalizer_towards_1 = (self.scores[chicken_1] / self.smax) + self.epsilon
                normalizer_towards_2 = (self.scores[chicken_2] / self.smax) + self.epsilon

                # calculating local empathies
                local_empathy_1_2 = np.log(1 + (max(0.0, self.scores[chicken_1] - self.scores[chicken_2]) / normalizer_towards_2))
                local_empathy_2_1 = np.log(1 + (max(0.0, self.scores[chicken_2] - self.scores[chicken_1]) / normalizer_towards_1))

                # calculating global empathy towards 1
                upper_norm = 1 / max(1, self._num_chickens - 1)
                score_differences_with_1 = [max(0.0, self.scores[chicken] - self.scores[chicken_1]) for chicken in self.agents]
                nominator_1 = upper_norm * np.sum(score_differences_with_1)
                global_empathy_1 = np.log(1 + (nominator_1 / normalizer_towards_1))

                # calculating global empathy towards 2
                score_differences_with_2 = [max(0.0, self.scores[chicken] - self.scores[chicken_2]) for chicken in self.agents]
                nominator_2 = upper_norm * np.sum(score_differences_with_2)
                global_empathy_2 = np.log(1 + nominator_2 / normalizer_towards_2)
                
                # combining everything
                rewards[chicken_1] = reward_1 + (self.beta1 * local_empathy_1_2 + self.beta2 * global_empathy_2) * emp_1_2
                rewards[chicken_2] = reward_2 + (self.beta1 * local_empathy_2_1 + self.beta2 * global_empathy_1) * emp_2_1
    
        # Adding in the scores
        for agent in self.agents:
            if self.EAR:
                self.scores[agent] += rewards[agent]
            else:
                self.scores[agent] += vanilla_outcomes[agent]

        game_over = any(self.scores[agent] >= self.smax for agent in self.agents)
        terminations = {agent: game_over for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        self.reset_resources()
        self.assign_resources()

        observations = {agent : self._get_obs(agent) for agent in self.agents}
        infos = {agent : {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        print(f"Step {getattr(self, 'current_step', '?')}")
        print("Scores:", {a: round(self.scores[a], 1) for a in self.agents})
        pairs = []
        solos = []
        for occ in self.resource_occupants.values():
            if len(occ) == 1:
                solos.append(occ[0])
            elif len(occ) == 2:
                pairs.append(tuple(sorted(occ)))
        print(f"Pairs: {pairs} | Solos: {solos}")
        if any(self.scores[a] >= self.smax for a in self.agents):
            winners = [a for a in self.agents if self.scores[a] >= self.smax]
            print(f"WINNER(S): {winners} 🎉")
        print("-" * 40)
        
