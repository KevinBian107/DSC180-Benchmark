# Deep Graph Library (DGL) Setup on DSMLP

## Overview

Deep Graph Library (DGL) is a Python package for deep learning on graphs. This guide covers installation and configuration on DSMLP.

## Prerequisites

- DSMLP pod with GPU access (recommended)
- PyTorch or TensorFlow backend
- Python 3.7 or higher

## Installation

### For CUDA 11.x (Most common on DSMLP)

```bash
# Check CUDA version
nvcc --version

# Install DGL with CUDA support
pip install dgl-cu117 dglgo -f https://data.dgl.ai/wheels/repo.html

# For the latest CUDA 11.8
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
```

### For CPU-only

```bash
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

## Verification

Test the installation:

```python
import dgl
import torch

print(f"DGL version: {dgl.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Create a simple graph
g = dgl.graph(([0, 1, 2], [1, 2, 3]))
print(f"Graph created: {g}")
print(f"Number of nodes: {g.num_nodes()}")
print(f"Number of edges: {g.num_edges()}")
```

## Basic Usage

### Creating Graphs

```python
import dgl
import torch

# Create a graph from edge list
src = torch.tensor([0, 1, 2, 3])
dst = torch.tensor([1, 2, 3, 4])
g = dgl.graph((src, dst))

# Add node features
g.ndata['feat'] = torch.randn(5, 10)

# Add edge features
g.edata['weight'] = torch.randn(4)

# Create a bidirectional graph
g = dgl.to_bidirected(g)
```

### Loading Built-in Datasets

```python
from dgl.data import CoraGraphDataset, CiteseerGraphDataset

# Load Cora dataset
dataset = CoraGraphDataset(raw_dir='/private/<username>/data')
g = dataset[0]

print(f"Number of nodes: {g.num_nodes()}")
print(f"Number of edges: {g.num_edges()}")
print(f"Node feature dimension: {g.ndata['feat'].shape[1]}")
print(f"Number of classes: {dataset.num_classes}")
```

## Example: Training a GCN

```python
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.data import CoraGraphDataset

# Load dataset
dataset = CoraGraphDataset(raw_dir='/private/<username>/data')
g = dataset[0]

# Define GCN model
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)
    
    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g = g.to(device)
features = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']

model = GCN(features.shape[1], 16, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
for epoch in range(100):
    logits = model(g, features)
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f}')
```

## Batch Processing with DGLGraph

```python
from dgl.data import MiniGCDataset
from dgl.dataloading import GraphDataLoader

# Load graph classification dataset
dataset = MiniGCDataset(320, 10, 20, raw_dir='/private/<username>/data')

# Create dataloader
dataloader = GraphDataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=False,
    num_workers=4
)

# Iterate through batches
for batched_graph, labels in dataloader:
    # batched_graph is a single DGLGraph object
    print(f"Batch size: {batched_graph.batch_size}")
    print(f"Total nodes: {batched_graph.num_nodes()}")
    print(f"Total edges: {batched_graph.num_edges()}")
```

## Performance Optimization

### Using GPU

```python
# Move graph to GPU
g = g.to('cuda')

# Use GPU for message passing
with torch.cuda.amp.autocast():
    output = model(g, features)
```

### Sampling for Large Graphs

```python
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader

# Create a neighbor sampler
sampler = MultiLayerFullNeighborSampler(2)

# Create dataloader for node classification
train_nids = torch.where(g.ndata['train_mask'])[0]
dataloader = NodeDataLoader(
    g,
    train_nids,
    sampler,
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=4
)

# Training with sampling
for input_nodes, output_nodes, blocks in dataloader:
    batch_inputs = blocks[0].srcdata['feat']
    batch_labels = blocks[-1].dstdata['label']
    # Train on sampled subgraph
```

## DGL with Heterogeneous Graphs

```python
import dgl

# Create a heterogeneous graph
data_dict = {
    ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
    ('user', 'likes', 'item'): (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 1])),
    ('item', 'liked-by', 'user'): (torch.tensor([0, 1, 1]), torch.tensor([0, 1, 2]))
}
g = dgl.heterograph(data_dict)

print(g)
print(f"Node types: {g.ntypes}")
print(f"Edge types: {g.etypes}")
```

## Troubleshooting

### ImportError: libcudart.so

Ensure CUDA is properly loaded:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### DGL version mismatch

Uninstall and reinstall:
```bash
pip uninstall dgl
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
```

### Out of memory errors

- Use neighbor sampling for large graphs
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

### Slow data loading

- Store datasets in `/private/` directory
- Increase `num_workers` in DataLoader
- Use pinned memory: `pin_memory=True`

## Additional Resources

- [DGL Documentation](https://docs.dgl.ai/)
- [DGL Tutorials](https://docs.dgl.ai/tutorials/blitz/index.html)
- [DGL Examples](https://github.com/dmlc/dgl/tree/master/examples)
- [DGL Discussion Forum](https://discuss.dgl.ai/)
