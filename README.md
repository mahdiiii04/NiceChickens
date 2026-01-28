# Nice Chickens 🐔  
*A Multi-Agent Reinforcement Learning environment with empathy-based incentives*

This repository contains a **multi-agent Chicken Game environment** implemented with **PettingZoo**, along with a **PPO training setup** using **Stable-Baselines3**.

The environment extends the classic Chicken Game by introducing **local and global empathy terms** that influence agent rewards and long-term score dynamics, allowing the study of **cooperation, inequality, and non-stationarity** in multi-agent systems.

---

## 📁 Project Structure

```

.
├── chicks.py      # PettingZoo ParallelEnv (NiceChickens)
├── eval.py        # PPO training and evaluation
├── models/        # Trained PPO models
└── README.md

````

---

## 🧠 Environment Overview

- **Agents**: Multiple chickens competing for shared resources
- **Game**: Chicken Game (Cooperate / Defect)
- **Resources**: Randomly reassigned every step (solo or paired)
- **Rewards**:
  - Vanilla Chicken Game payoff
  - Motivation incentive (score-based)
  - Local empathy (pairwise)
  - Global empathy (population-level)
- **Termination**: When any agent reaches `smax`

The environment is implemented as a **PettingZoo ParallelEnv** and wrapped into a **single-policy PPO setup**.

---

## ⚙️ Installation

```bash
pip install gymnasium pettingzoo stable-baselines3 supersuit torch numpy
````

---

## 🚀 Training

To train a PPO agent:

```bash
python eval.py
```

Training hyperparameters are defined directly in `eval.py`, for example:

```python
num_timesteps = 1_000_000
num_chickens = 6
v = 10.0
c = 2.0
beta1 = 1.0
beta2 = 1.0
```

---

## 🧪 Trained Models

Pretrained models are stored in the `models/` directory.

You can load a trained model using Stable-Baselines3:

```python
from stable_baselines3 import PPO

model = PPO.load("models/final_model")
```

---

## 📊 Evaluation Metrics

During training, the following metrics are computed at the end of each rollout:

* Cooperation rate (paired interactions)
* Gini coefficient (score inequality)
* Winner score
* Number of winners and ties
* Mean episode reward

Metrics are printed to the console and saved internally during training.

---

## 📖 Project Write-up

A detailed exploration of the motivation, environment design, and experimental results is available here:

👉 **Project post:**
[https://mahdiiii04.github.io/posts/NiceChickens/](https://mahdiiii04.github.io/posts/NiceChickens/)

---

## 🎯 Research Focus

This project explores:

* Empathy as a reward-shaping mechanism
* Emergent cooperation in competitive environments
* Inequality dynamics in multi-agent reinforcement learning
* Population-level feedback effects

It is intended as a **research prototype**, not a benchmark environment.

---
