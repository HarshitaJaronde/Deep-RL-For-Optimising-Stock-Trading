# Double DQN for Inverted Pendulum Control

## Overview

This project implements a **Double Deep Q-Network (Double DQN)** to solve the classic **inverted pendulum control problem** using the `Pendulum-v1` environment from Gymnasium.

Since the environment has a **continuous action space**, it is adapted for DQN by discretizing the torque values. The agent learns to **swing up and balance the pendulum upright** through reinforcement learning.

---

## Problem Setup

### Environment

* **Task:** Control an inverted pendulum
* **Action Space:** Continuous torque in `[-2.0, 2.0]`
* **Goal:** Stabilize the pendulum upright

### Discretization

* Continuous torque is divided into **21 discrete actions**
* Odd number ensures a **zero-torque action** for stable balancing

---

## State Representation

```
[cos(θ), sin(θ), θ̇]
```

**Why?**

* Avoids discontinuity at ±π
* Provides smooth representation of orientation

---

## Model Architecture

```
Input (3)
 → Dense (64, ReLU)
 → Dense (64, ReLU)
 → Output (21, Linear)
```

* Outputs **Q-values for each discrete action**

---

## Algorithm: Double DQN

Standard DQN overestimates Q-values. Double DQN fixes this:

* **Action selection (online network):**

```
a' = argmax Q_online(s')
```

* **Evaluation (target network):**

```
Q_target(s', a')
```

**Benefit:** More stable and accurate value estimation

---

## Target Network Updates (Soft Updates)

```
θ_target ← τ · θ_online + (1 − τ) · θ_target
```

* `τ = 0.005`

**Why?**

* Smooth updates
* Reduces training instability

---

## Experience Replay

* Buffer size: **50,000**
* Stores: `(s, a, r, s', done)`

**Benefits:**

* Breaks correlation between samples
* Improves learning efficiency

---

## Exploration Strategy

Epsilon-greedy with exponential decay:

* `epsilon_start = 1.0`
* `epsilon_end = 0.05`

**Behavior:**

* Early: more exploration
* Later: more exploitation

---

## Training Configuration

| Hyperparameter       | Value  |
| -------------------- | ------ |
| Discount Factor (γ)  | 0.99   |
| Learning Rate        | 1e-3   |
| Batch Size           | 64     |
| Loss Function        | MSE    |
| Gradient Clip Norm   | 1.0    |
| Replay Buffer Size   | 50,000 |
| Warm-up Steps        | 1,000  |
| Soft Update Rate (τ) | 0.005  |
| Episodes             | 300    |
| Max Steps/Episode    | 200    |

---

## Training Loop

For each episode:

1. Select action (epsilon-greedy)
2. Execute in environment
3. Store transition
4. Sample minibatch
5. Compute Double DQN target
6. Update network
7. Soft update target network
8. Decay epsilon

---

## Evaluation

* Set `epsilon = 0` (greedy policy)
* Run multiple episodes

**Metrics:**

* Episode return
* Training loss

**Extras:**

* Reward/loss plots
* Animation of pendulum behavior

---

## Key Highlights

* Converts continuous control → discrete DQN
* Uses **Double DQN** for stability
* Applies **soft target updates**
* Efficient learning via replay buffer
* Lightweight (runs on modest hardware)

---

## Future Improvements

* Prioritized Experience Replay
* Dueling DQN
* Noisy Networks
* Continuous RL methods (DDPG, SAC)

---

## Requirements

* Python 3.x
* PyTorch / TensorFlow
* Gymnasium
* NumPy
* Matplotlib

---

## Usage

```bash
python train.py
```

(Optional)

```bash
python evaluate.py
```

---

## Acknowledgements

* Gymnasium (OpenAI Gym)
* DeepMind DQN & Double DQN research
