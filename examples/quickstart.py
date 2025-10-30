#!/usr/bin/env python3
"""
Quick Start Example for Graph Learning on DSMLP

This script demonstrates basic usage of PyTorch Geometric on the Cora dataset.
Use it to verify your environment is properly configured.

Usage:
    python examples/quickstart.py
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv


class SimpleGCN(torch.nn.Module):
    """Simple 2-layer Graph Convolutional Network."""
    
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second layer
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)


def main():
    print("=" * 60)
    print("Graph Learning Quick Start Example")
    print("=" * 60)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load dataset
    print("\n" + "-" * 60)
    print("Loading Cora dataset...")
    dataset = Planetoid(root='/tmp/data', name='Cora')
    data = dataset[0].to(device)
    
    print(f"Dataset: {dataset}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"\nGraph statistics:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Average degree: {data.num_edges / data.num_nodes:.2f}")
    print(f"  Training nodes: {data.train_mask.sum()}")
    print(f"  Validation nodes: {data.val_mask.sum()}")
    print(f"  Test nodes: {data.test_mask.sum()}")
    
    # Initialize model
    print("\n" + "-" * 60)
    print("Initializing model...")
    model = SimpleGCN(
        num_features=dataset.num_features,
        hidden_dim=16,
        num_classes=dataset.num_classes
    ).to(device)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Training loop
    print("\n" + "-" * 60)
    print("Training...")
    
    model.train()
    for epoch in range(1, 51):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            # Evaluation
            model.eval()
            with torch.no_grad():
                logits = model(data)
                
                # Training accuracy
                train_pred = logits[data.train_mask].max(1)[1]
                train_acc = train_pred.eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
                
                # Validation accuracy
                val_pred = logits[data.val_mask].max(1)[1]
                val_acc = val_pred.eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
                
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            model.train()
    
    # Final evaluation
    print("\n" + "-" * 60)
    print("Final Evaluation...")
    model.eval()
    with torch.no_grad():
        logits = model(data)
        
        # Test accuracy
        test_pred = logits[data.test_mask].max(1)[1]
        test_acc = test_pred.eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        
        print(f"Test Accuracy: {test_acc:.4f}")
    
    print("\n" + "=" * 60)
    print("Quick start completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check out the documentation in docs/")
    print("2. Try different models and datasets")
    print("3. Optimize hyperparameters")
    print("4. Submit your benchmark results!")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install required packages:")
        print("  pip install torch torch-geometric")
        print("\nFor detailed setup instructions, see docs/setup/")
    except Exception as e:
        print(f"Error: {e}")
        print("\nFor troubleshooting, see docs/dependencies/troubleshooting.md")
