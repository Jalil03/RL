# How to Run the RL-PDNN Integrated System

Follow these exact steps to run the complete project from scratch.

## 1. Setup Environment
Open your terminal in the project directory: `c:\Users\JL\OneDrive\Desktop\RL`

Install the required dependencies:
```powershell
pip install gym torch numpy matplotlib torchvision
```

## 2. Train the RL Agent ("The Brain")
This trains the reinforcement learning agent to make decisions.
```powershell
python -m rl_pdnn.main
```
*   **Wait** until it finishes (500 episodes).
*   It will create `rl_pdnn/model.pth`.

## 3. Train the CNN Model ("The Workload")
This downloads the MNIST dataset and trains the LeNet-5 neural network.
```powershell
python -m split_inference.train_cnn
```
*   **Wait** until it finishes.
*   It will create `split_inference/mnist_lenet.pth`.

## 4. Run the Full Demo
This connects the RL decision-making with the CNN execution.
```powershell
python -m full_demo
```
*   **Output**: You will see it allocate layers to simulated devices and print the final prediction (e.g., "Final Prediction: Class 5").

## 5. Verify Performance
Run the performance report generator to grade the RL agent.
```powershell
python -m rl_pdnn.evaluate
```
*   **Output**: It will print a score (e.g., -113.02) and save `PERFORMANCE_REPORT.md`.

---
**Summary of Commands:**
```powershell
pip install gym torch numpy matplotlib torchvision
python -m rl_pdnn.main
python -m split_inference.train_cnn
python -m full_demo
python -m rl_pdnn.evaluate
```
