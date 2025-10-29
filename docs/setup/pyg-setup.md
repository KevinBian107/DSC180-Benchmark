# PyTorch Geometric (PyG) Setup on DSMLP

## Overview

PyTorch Geometric is a library for deep learning on graphs and other irregular structures. This guide covers installation and setup on DSMLP.

## Prerequisites

- DSMLP pod with GPU access
- CUDA-compatible PyTorch installation
- Python 3.8 or higher

## Installation

### Method 1: Using pip (Recommended)

```bash
# First, check your PyTorch and CUDA versions
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Install PyG and dependencies
pip install torch-geometric

# Install optional dependencies for full functionality
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

Replace `${TORCH}` and `${CUDA}` with your versions (e.g., `2.0.0` and `cu118`).

### Method 2: Using conda

```bash
conda install pyg -c pyg
```

## Verification

Test the installation:

```python
import torch
import torch_geometric
from torch_geometric.data import Data

print(f"PyTorch version: {torch.__version__}")
print(f"PyG version: {torch_geometric.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Create a simple graph
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)
print(f"Graph created successfully: {data}")
```

## Common Datasets

Download and cache datasets:

```python
from torch_geometric.datasets import Planetoid, TUDataset

# Citation networks
dataset = Planetoid(root='/private/<username>/data', name='Cora')

# Graph classification datasets
dataset = TUDataset(root='/private/<username>/data', name='MUTAG')
```

## Example: Training a GCN

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Load dataset
dataset = Planetoid(root='/private/<username>/data', name='Cora')
data = dataset[0]

# Define model
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

## Performance Optimization

### Data Loading

Use `torch_geometric.loader.DataLoader` for batch processing:

```python
from torch_geometric.loader import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

### GPU Memory Management

Monitor GPU memory:
```bash
watch -n 1 nvidia-smi
```

Reduce memory usage:
```python
# Use gradient checkpointing
torch.utils.checkpoint.checkpoint(model, data)

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    out = model(data)
    loss = criterion(out, target)
```

## Troubleshooting

### ImportError: cannot import name 'scatter'

Install torch-scatter:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

### CUDA out of memory

- Reduce batch size
- Use gradient accumulation
- Enable gradient checkpointing
- Clear cache: `torch.cuda.empty_cache()`

### Slow data loading

- Store data in `/private/` directory (faster I/O)
- Use `num_workers` in DataLoader
- Pre-process and cache datasets

## Additional Resources

- [PyG Documentation](https://pytorch-geometric.readthedocs.io/)
- [PyG Examples](https://github.com/pyg-team/pytorch_geometric/tree/master/examples)
- [PyG Tutorials](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html)
