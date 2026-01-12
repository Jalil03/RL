import gym
from gym import spaces
import numpy as np
from .utils import generate_dummy_dnn_model, generate_iot_network, IoTDevice, DNNLayer

class IoTEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    Represents the distributed IoT system for DNN inference.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, num_devices=5, num_layers=10):
        super(IoTEnv, self).__init__()
        
        self.num_devices = num_devices
        self.num_layers = num_layers
        
        # Action Space: Choose one of the D devices to execute the current layer
        self.action_space = spaces.Discrete(num_devices)
        
        # State Space: 
        # [Current Layer Index (1), 
        #  For each device: CPU (1), Memory Remaining (1), Bandwidth (1), Privacy Clearance (1)]
        # Total size = 1 + (4 * num_devices)
        self.state_dim = 1 + (4 * num_devices)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(self.state_dim,), dtype=np.float32)
        
        self.devices = []
        self.model_layers = []
        self.current_layer_idx = 0
        self.previous_device_id = -1 # -1 means source/input
        
        self.reset()

    def reset(self):
        """Resets the environment for a new inference pass."""
        self.devices = generate_iot_network(self.num_devices)
        self.model_layers = generate_dummy_dnn_model(self.num_layers)
        self.current_layer_idx = 0
        self.previous_device_id = -1 # Start at source
        
        # Reset device memory usage
        for d in self.devices:
            d.current_memory_usage = 0
            
        return self._get_observation()

    def _get_observation(self):
        """Constructs the normalized state vector."""
        # Normalize Layer Index (0-1)
        obs = [float(self.current_layer_idx) / self.num_layers]
        
        for d in self.devices:
            # ROI: CPU, Mem Free, Bandwidth, Privacy
            # Normalize CPU (approx max 2.0)
            obs.append(d.cpu_speed / 2.0)
            # Normalize Memory (approx max 500)
            obs.append((d.memory_capacity - d.current_memory_usage) / 500.0)
            # Normalize Bandwidth (approx max 100)
            obs.append(d.bandwidth / 100.0)
            # Privacy is already 0 or 1
            obs.append(float(d.privacy_clearance))
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """
        Executes one step (allocating one layer).
        Args:
            action (int): ID of the selected device.
        """
        selected_device_id = int(action)
        selected_device = self.devices[selected_device_id]
        current_layer = self.model_layers[self.current_layer_idx]
        
        reward = 0
        done = False
        info = {}
        
        # --- 1. Check Constraints ---
        if not selected_device.can_host(current_layer):
            # Penalty for constraint violation
            reward = -100 
            # We treat this as a failure, but usually in RL we might want to continue 
            # or just skip to next episode. Here we'll just penalize heavily and move on 
            # effectively assuming the task failed or ran very slowly.
            # For simplicity, we'll mark episode as done if it fails hard, 
            # or just give huge penalty. Let's give huge penalty.
            
            # To help training, we might add the latency anyway as a baseline?
            # Let's say it enters a fallback mode (cloud) which is slow.
            reward -= 50 
        else:
            # --- 2. Calculate Latency (Cost) ---
            
            # Computation Latency: Demand / (BaseSpeed * DeviceFactor)
            # Simplified: Time = Demand / Speed
            comp_latency = current_layer.computation_demand / selected_device.cpu_speed
            
            # Transmission Latency: Data / LinkSpeed
            # Only if moving from one device to another
            trans_latency = 0
            if self.previous_device_id != -1 and self.previous_device_id != selected_device_id:
                # Transmitting Output of PREVIOUS layer to CURRENT device
                # For layer 0, previous is source (assume 0 cost or fixed cost)
                # Getting input data for this layer
                prev_layer_output = 0
                if self.current_layer_idx > 0:
                     prev_layer_output = self.model_layers[self.current_layer_idx - 1].output_data_size
                else:
                    # Input data size for first layer
                    prev_layer_output = 5.0 # MB
                
                # Min bandwidth of the link (simplified: uses destination bandwidth)
                trans_latency = prev_layer_output / selected_device.bandwidth
            elif self.previous_device_id == -1:
                 # Initial transmission from source to first device
                 input_data = 5.0 # MB
                 trans_latency = input_data / selected_device.bandwidth

            total_latency = comp_latency + trans_latency
            
            # Reward is negative latency
            reward = -total_latency
            
            # Update device state (occupy memory)
            selected_device.current_memory_usage += current_layer.memory_demand

        # --- 3. Advance ---
        self.previous_device_id = selected_device_id
        self.current_layer_idx += 1
        
        if self.current_layer_idx >= self.num_layers:
            done = True
            
        next_obs = self._get_observation()
        
        return next_obs, reward, done, info
