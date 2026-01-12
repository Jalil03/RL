import numpy as np
from .env import IoTEnv
from .agent import DQNAgent
import os

def evaluate():
    NUM_EPISODES = 100
    NUM_DEVICES = 5
    NUM_LAYERS = 10
    MODEL_PATH = "rl_pdnn/model.pth"
    
    print("Starting Evaluation...")
    
    env = IoTEnv(num_devices=NUM_DEVICES, num_layers=NUM_LAYERS)
    state_dim = env.state_dim
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    
    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print("Model not found! Please run training first.")
        return

    # Turn off exploration for evaluation
    agent.epsilon = 0.0
    
    scores = []
    latencies = []
    
    for e in range(NUM_EPISODES):
        state = env.reset()
        done = False
        score = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
            
        scores.append(score)
        latencies.append(-score) # Since reward is negative latency

    avg_score = np.mean(scores)
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    
    # Qualitative Judgment
    judgment = "POOR"
    if avg_score > -100:
        judgment = "EXCELLENT"
    elif avg_score > -200:
        judgment = "GOOD"
    elif avg_score > -500:
        judgment = "FAIR"
        
    report_content = f"""# RL-PDNN Performance Report

## Judgment: **{judgment}**

### Summary Statistics (100 Test Episodes)
*   **Average Score**: {avg_score:.2f}
*   **Average Latency**: {avg_latency:.2f} ms
*   **Best Latency**: {min_latency:.2f} ms
*   **Worst Latency**: {max_latency:.2f} ms

### Interpretation
The agent was evaluated on 100 random inference descriptions.
A higher score (closer to 0) indicates lower latency and fewer constraint violations.

*   **Excellent**: > -100
*   **Good**: -100 to -200
*   **Fair**: -200 to -500
*   **Poor**: < -500

Generated automatically by `evaluate.py`.
"""

    with open("PERFORMANCE_REPORT.md", "w") as f:
        f.write(report_content)
        
    print("Evaluation Complete. Report saved to PERFORMANCE_REPORT.md")
    print(f"Average Score: {avg_score:.2f} ({judgment})")

if __name__ == "__main__":
    evaluate()
