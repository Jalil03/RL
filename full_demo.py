import torch
import torch
from split_inference.cnn_model import LeNet5
from integrated_system.inference_engine import DistributedRunner
from rl_pdnn.agent import DQNAgent
from rl_pdnn.env import IoTEnv
import numpy as np
import random
import os

def run_demo():
    print("=== RL-PDNN Integrated System Demo ===")
    
    # 1. Load Trained RL Agent
    print("\n[Stage 1] Loading RL Agent...")
    env = IoTEnv(num_devices=5, num_layers=4) # Assuming LeNet has ~4 separate trainable chunks + flattening
    # Note: Our Environment assumes 10 layers, but LeNet in our implementation has 6 blocks.
    # For this demo, we'll map the 6 LeNet blocks to the RL decision steps. 
    # We'll re-init env with match.
    
    env_demo = IoTEnv(num_devices=5, num_layers=6) 
    
    agent = DQNAgent(state_dim=env_demo.state_dim, action_dim=5)
    
    rl_pixels = "rl_pdnn/model.pth"
    if os.path.exists(rl_pixels):
        agent.load(rl_pixels)
        print("RL Agent loaded.")
    else:
        print("Warning: Trained RL model not found. Using random agent.")
    
    # 2. Get Allocation Map from RL
    print("\n[Stage 2] Asking RL Agent for Allocation Map...")
    state = env_demo.reset()
    allocation_map = []
    
    for i in range(6): # For each layer in LeNet5
        # Agent decides
        action = agent.act(state)
        allocation_map.append(action)
        
        # Advance env (dummy step for demo)
        next_state, _, _, _ = env_demo.step(action)
        state = next_state
        
    print(f"Generated Allocation Map: {allocation_map}")
    print(f"(Interpret: Layer 0 -> Dev {allocation_map[0]}, Layer 1 -> Dev {allocation_map[1]}...)")
    
    # 3. Load Trained CNN
    print("\n[Stage 3] Loading CNN Model...")
    cnn = LeNet5()
    cnn_path = "split_inference/mnist_lenet.pth"
    if os.path.exists(cnn_path):
        cnn.load_state_dict(torch.load(cnn_path))
        print("CNN Weights loaded.")
    else:
        print("Error: CNN weights not found! Run 'python -m split_inference.train_cnn' first.")
        return

    # 4. Perform Distributed Inference
    print("\n[Stage 4] Executing on Distributed Engine...")
    runner = DistributedRunner(cnn)
    
    # Create valid dummy input (1 example form MNIST format)
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # RUN!
    output = runner.run(dummy_input, allocation_map)
    
    predicted_class = torch.argmax(output).item()
    print(f"Final Prediction: Class {predicted_class}")
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    run_demo()
