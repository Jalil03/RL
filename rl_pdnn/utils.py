import dataclasses
from typing import List, Dict
import random

@dataclasses.dataclass
class DNNLayer:
    """Represents a single layer of the Neural Network task."""
    layer_id: int
    name: str # e.g. "conv1", "fc1"
    computation_demand: float # Estimated FLOPs or time units
    memory_demand: float # MB required to store weights/output
    output_data_size: float # MB of data to transmit to next layer
    privacy_level: int # Privacy requirement (0=Public, 1=Private)

@dataclasses.dataclass
class IoTDevice:
    """Represents an IoT device in the network."""
    device_id: int
    cpu_speed: float # Relative CPU speed factor (e.g. 1.0 = baseline)
    memory_capacity: float # Total MB
    current_memory_usage: float # Used MB
    bandwidth: float # Mbps link speed
    privacy_clearance: int # Max privacy level this device is allowed to see (0 or 1)
    
    def can_host(self, layer: DNNLayer) -> bool:
        """Check if device has resources and clearance for the layer."""
        if self.current_memory_usage + layer.memory_demand > self.memory_capacity:
            return False # Out of memory
        if self.privacy_clearance < layer.privacy_level:
            return False # Privacy violation
        return True

def generate_dummy_dnn_model(num_layers=10) -> List[DNNLayer]:
    """Generates a sequence of layers representing a simple CNN."""
    layers = []
    for i in range(num_layers):
        # Alternate heavy compute/small data and light compute/large data
        is_conv = i < (num_layers - 2)
        layers.append(DNNLayer(
            layer_id=i,
            name=f"layer_{i}_{'conv' if is_conv else 'fc'}",
            computation_demand=random.uniform(5, 20) if is_conv else random.uniform(1, 5),
            memory_demand=random.uniform(10, 50),
            output_data_size=random.uniform(1, 10),
            privacy_level=1 if i == 0 else 0 # Assume input data (layer 0) is private
        ))
    return layers

def generate_iot_network(num_devices=5) -> List[IoTDevice]:
    """Generates a set of heterogeneous IoT devices."""
    devices = []
    for i in range(num_devices):
        devices.append(IoTDevice(
            device_id=i,
            cpu_speed=random.uniform(0.5, 2.0),
            memory_capacity=random.uniform(100, 500), # MB
            current_memory_usage=0.0,
            bandwidth=random.uniform(10, 100), # Mbps
            privacy_clearance=random.choice([0, 1]) # Some trusted, some not
        ))
    return devices
