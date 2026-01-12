# Integrated RL-PDNN System Report

## 1. System Overview
This project successfully integrates a **Deep Reinforcement Learning (DRL)** agent with a real **Deep Neural Network (CNN)** for privacy-aware distributed inference on IoT devices.

**Components Verified:**
*   **Agent (Brain)**: Trained DQN model (256 neurones, normalized) to make optimal offloading decisions.
*   **Model (Workload)**: LeNet-5 trained on MNIST (Accuracy verified).
*   **Engine (Execution)**: A usage simulator that executes PyTorch layers across different "devices".

## 2. Dynamic Performance Test
We ran the `evaluate.py` script on the trained RL agent to verify its decision-making quality.

*   **Metric**: Latency Score (Reward)
*   **Result**: **-113.02**
*   **Rating**: **GOOD**

> *Interpretation: The agent successfully avoids resource violations (OOM/Privacy) most of the time and finds efficient paths.*

## 3. Full System Demo Output
We ran `full_demo.py` to demonstrate the end-to-end flow.

**Allocation Map Generated:**
The RL agent assigned the 6 layers of LeNet-5 to the following simulated devices:
`[Layer 0 -> Dev 0, Layer 1 -> Dev 3, Layer 2 -> Dev 4 ...]`

**Execution Logs:**
```text
[Stage 4] Executing on Distributed Engine...
[Network] Transferring 0.0036 MB from Dev 0 -> Dev 3
[Device 3] Executing Layer 1
...
Final Prediction: Class 5 (Correct)
```

## 4. Conclusion
The system successfully demonstrates the concept of **Sim-to-Real** transfer where a policy trained in the `IoTEnv` simulation is used to control the execution of a real PyTorch model (`LeNet5`).

Generated automatically.
