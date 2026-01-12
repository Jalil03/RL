import torch
import time
import sys

class DistributedRunner:
    """
    Executes a PyTorch model in a split fashion based on an allocation map.
    Simulates the network transfer delays between devices.
    """
    def __init__(self, model):
        self.model = model
        self.device_logs = []
        
    def run(self, input_data, allocation_map):
        """
        Args:
            input_data: The input tensor (image).
            allocation_map: List of device IDs, one for each layer in model.layers.
        """
        x = input_data
        current_device = -1 # Source
        
        print("\n--- Starting Distributed Inference ---")
        
        for i, layer in enumerate(self.model.layers):
            target_device = allocation_map[i]
            
            # Simulate Network Transfer if crossing boundaries
            if target_device != current_device:
                data_size_mb = x.element_size() * x.nelement() / (1024 * 1024)
                print(f"[Network] Transferring {data_size_mb:.4f} MB from Dev {current_device} -> Dev {target_device}")
                print(f"          ... (Simulated Delay) ...")
                # time.sleep(0.1) # Optional: Real delay
                
            # Simulate Computation on Device
            print(f"[Device {target_device}] Executing Layer {i}")
            with torch.no_grad():
                x = layer(x)
            
            current_device = target_device
            
        print("--- Inference Complete ---\n")
        return x
