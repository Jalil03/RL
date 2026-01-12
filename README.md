# Integrated RL-PDNN System

This repository contains a complete framework for **Privacy-Aware Distributed Deep Neural Network (DNN)** inference. It combines Reinforcement Learning decision-making with actual PyTorch model execution.

## 🌟 Modules

### 1. The Brain (`rl_pdnn/`)
Contains the Deep Reinforcement Learning agent responsible for making offloading decisions.
*   **Algorithm**: Deep Q-Network (DQN) with Experience Replay.
*   **Optimizations**: Normalized State Space, 256-neuron architecture.
*   **Output**: An "Allocation Map" deciding which Device ID (0-4) runs which Layer.

### 2. The Model (`split_inference/`)
Contains the real Deep Learning workload.
*   **Architecture**: LeNet-5 (Modified with `nn.ModuleList` for split execution).
*   **Dataset**: MNIST.
*   **Training**: Train locally to generate `mnist_lenet.pth`.

### 3. The Engine (`integrated_system/`)
Contains the logic to execute the model across simulated distributed devices.
*   **`DistributedRunner`**: Takes the RL's allocation map and the CNN, and runs inference while simulating network transfers.

## 🚀 Quick Start

### Step 1: Train the RL Agent
Train the "Brain" to learn how to optimize latency and privacy.
```bash
python -m rl_pdnn.main
```

### Step 2: Train the CNN
Train the "Workload" (LeNet-5) on MNIST data.
```bash
python -m split_inference.train_cnn
```

### Step 3: Run the Full Demo
Connect the Brain to the Workload. This script will:
1.  Load the trained RL Agent.
2.  Ask it to allocate the LeNet layers.
3.  Simulate the distributed execution.
```bash
python -m full_demo
```

### Step 4: Verify Performance
Run a dedicated evaluation of the RL agent's quality.
```bash
python -m rl_pdnn.evaluate
```

## 📊 Results

*   **RL Performance**: Achieved **GOOD** rating (Score > -150).
*   **Inference Accuracy**: Standard MNIST accuracy (~98%).
*   **Latency**: Minimized by the RL agent's offloading strategy.

---
*Implementation of "Reinforcement Learning for Privacy-Aware Distributed Neural Networks in IoT Systems"*
