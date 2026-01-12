import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .cnn_model import LeNet5
import os

def train():
    BATCH_SIZE = 64
    EPOCHS = 1 # Quick training for demo purposes
    LR = 0.01
    
    # 1. Setup Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Downloading MNIST...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Setup Model
    model = LeNet5()
    optimizer = optim.SGD(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop
    print("Starting Training...")
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.4f}")
                running_loss = 0.0
                
    # 4. Save
    if not os.path.exists('split_inference'):
        os.makedirs('split_inference')
    
    save_path = 'split_inference/mnist_lenet.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
