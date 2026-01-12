# Privacy-Aware Distributed Deep Neural Networks via Reinforcement Learning (RL-PDNN)

This repository hosts a complete framework for distributing **Deep Learning** inference tasks across a network of resource-constrained **IoT devices**, optimized by a **Deep Reinforcement Learning (DRL)** agent.

The system is designed to minimize end-to-end latency while strictly respecting device constraints (Memory, Computing Power) and Data Privacy requirements.

---

## 📚 Table of Contents
1.  [Project Structure](#-project-structure)
2.  [The Logic Behind the Code](#-the-logic-behind-the-code)
    *   [The RL Environment (Simulation)](#1-the-rl-environment-rl_pdnn)
    *   [The CNN Model (Workload)](#2-the-cnn-model-split_inference)
    *   [The Inference Engine (Integration)](#3-the-inference-engine-integrated_system)
3.  [Algorithmic Deep Dive](#-algorithmic-deep-dive)
4.  [How to Run](#-how-to-run)

---

## 📂 Project Structure

```text
RL/
├── rl_pdnn/                 # THE BRAIN (Reinforcement Learning Module)
│   ├── agent.py             # The AI Agent (Deep Q-Network implementation)
│   ├── env.py               # The Simulation Environment (IoT Network & Constraints)
│   ├── main.py              # Training Script (Rule the Agent)
│   ├── evaluate.py          # Grading Script (Test the Agent without learning)
│   └── utils.py             # Data Structures (Device, Layer definitions)
│
├── split_inference/         # THE WORKLOAD (Deep Learning Module)
│   ├── cnn_model.py         # LeNet-5 Architecture (Split-ready)
│   └── train_cnn.py         # Script to train LeNet-5 on MNIST
│
├── integrated_system/       # THE ENGINE (Execution Module)
│   └── inference_engine.py  # Simulator that runs PyTorch layers on "devices"
│
├── full_demo.py             # END-TO-END DEMO (Integration of all above)
└── README.md                # This file
```

---

## 🧠 The Logic Behind the Code

Here is a detailed breakdown of what every critical file does and **why** it was coded that way.

### 1. The RL Environment (`rl_pdnn`)

#### `env.py` (The World)
This file defines the "game" the AI plays.
*   **State Normalization**: inside `_get_observation()`, we divide all values (CPU speed, Memory) by their maximums to get numbers between $0$ and $1$. Neural networks struggle with large numbers (like 500 MB), so normalization is crucial for fast convergence.
*   **The "Punishment"**: inside `step()`, if the AI assigns a layer to a device that is out of memory or not secure (`can_host() == False`), we return a reward of **-50**. This teaches the AI to respect physical constraints.
*   **Latency Calculation**: `reward = -(Computation Time + Transmission Time)`. The AI maximizes reward, which mathematically equals **minimizing latency**.

#### `agent.py` (The Learner)
This file implements the **Deep Q-Network (DQN)**.
*   **Replay Buffer**: The `remember()` and `replay()` functions allow the agent to save its experiences and learn from them later. This breaks the correlation between consecutive steps, stabilizing learning.
*   **Epsilon-Greedy**: The `act()` function behaves randomly at first (`epsilon=1.0`) to explore, and becomes deterministic (`epsilon=0.01`) to exploit the best strategy.
*   **Double Network**: We use `policy_net` (to act) and `target_net` (to evaluate). This trick prevents the "moving target" problem in RL training.

### 2. The CNN Model (`split_inference`)

#### `cnn_model.py` (The Structure)
*   **Why `nn.ModuleList`?**: Standard PyTorch models use `nn.Sequential`. We explicitly used `nn.ModuleList` to define layers independently.
*   **Effect**: This allows us to run `layer[0](x)` on Device A, stop, send data to Device B, and then run `layer[1](x)`. A standard model would run everything at once on one device.

### 3. The Inference Engine (`integrated_system`)

#### `inference_engine.py` (The Simulator)
*   **Hybrid Execution**: This script runs **real** PyTorch computations (convolutions, matrix multiplications) but **simulates** the network delay.
*   **Logic**:
    ```python
    if target_device != current_device:
        # Calculate size of tensor in MB
        # Simulate wait time (time.sleep) or log value
    ```
    This bridges the gap between our theoretical RL simulation and practical application execution.

---

## 🧮 Algorithmic Deep Dive

The core algorithm solves the **Constrained Optimization Problem**:

**Objective**: $\min \sum (T_{comp} + T_{trans})$

**Constraints**:
1.  $M_{required} \le M_{available}$ (Memory)
2.  $P_{required} \le P_{clearance}$ (Privacy)

**The Reinforcement Learning Approach**:
Instead of using a solver (like Integer Programming) which is slow, we treat this as a decision process (MDP).
1.  **Observation**: The agent sees "Step 3, Device 1 has 50% RAM".
2.  **Action**: It outputs "Assign to Device 1".
3.  **Feedback**: It immediately gets the cost (latency) or penalty.

Over 500 episodes, the Neural Network within `agent.py` approximates the optimal mapping function $f(State) \rightarrow Action$.

---

## 🚀 How to Run

1.  **Install Dependencies**:
    ```bash
    pip install gym torch numpy matplotlib torchvision
    ```

2.  **Train the Brain (RL)**:
    ```bash
    python -m rl_pdnn.main
    ```

3.  **Train the Workload (CNN)**:
    ```bash
    python -m split_inference.train_cnn
    ```

4.  **Run the Full Demo**:
    ```bash
    python -m full_demo
    ```

5.  **View Performance Report**:
    ```bash
    python -m rl_pdnn.evaluate
    ```
