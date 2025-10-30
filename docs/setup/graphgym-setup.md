# GraphGym Setup on DSMLP

## Overview

GraphGym is a platform for designing and evaluating Graph Neural Networks (GNN). It provides a modular framework for GNN experimentation.

## Prerequisites

- PyTorch Geometric installed
- Python 3.7+
- DSMLP pod with GPU recommended

## Installation

```bash
# Install from PyPI
pip install graphgym

# Or install from source for latest features
git clone https://github.com/snap-stanford/GraphGym.git
cd GraphGym
pip install -e .
```

## Project Structure

GraphGym uses a configuration-based approach:

```
project/
├── configs/
│   └── example.yaml
├── run/
│   └── results/
└── datasets/
```

## Configuration Files

Create a config file (e.g., `configs/gcn_cora.yaml`):

```yaml
out_dir: results
dataset:
  format: PyG
  name: Cora
  task: node
  task_type: classification
  transductive: True
  split: [0.8, 0.1, 0.1]
train:
  mode: custom
  batch_size: 1
  epochs: 200
  optimizer: adam
  lr: 0.01
  weight_decay: 5e-4
model:
  type: gnn
  loss_fun: cross_entropy
gnn:
  layers_pre_mp: 1
  layers_mp: 2
  layers_post_mp: 1
  dim_inner: 16
  layer_type: gcnconv
optim:
  optimizer: adam
  base_lr: 0.01
  weight_decay: 5e-4
```

## Running Experiments

### Single Run

```bash
python main.py --cfg configs/gcn_cora.yaml --repeat 1
```

### Multiple Runs with Different Seeds

```bash
python main.py --cfg configs/gcn_cora.yaml --repeat 5
```

### Grid Search

Create a grid config:

```yaml
# configs/grid.yaml
out_dir: results
dataset:
  format: PyG
  name: Cora
train:
  epochs: 200
gnn:
  layer_type: gcnconv
  dim_inner: [16, 32, 64]  # Grid search over dimensions
optim:
  base_lr: [0.001, 0.01, 0.1]  # Grid search over learning rates
```

Run grid search:
```bash
python main.py --cfg configs/grid.yaml --repeat 3
```

## Custom Datasets

### Adding Custom PyG Dataset

```python
# custom_dataset.py
from torch_geometric.data import InMemoryDataset

class CustomDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return ['data.pt']
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        # Download raw data
        pass
    
    def process(self):
        # Process raw data into PyG format
        pass
```

Register in GraphGym:
```python
from graphgym.register import register_dataset
register_dataset('custom', CustomDataset)
```

## Custom GNN Layers

```python
# custom_layer.py
from torch_geometric.nn import MessagePassing
from graphgym.register import register_layer

@register_layer('custom_conv')
class CustomConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        return self.lin(x_j)
```

Use in config:
```yaml
gnn:
  layer_type: custom_conv
```

## Analyzing Results

GraphGym automatically generates results:

```bash
# View results
cd results/
ls

# Results include:
# - agg/: Aggregated statistics
# - individual/: Individual run results
# - best.yaml: Best configuration
```

## Visualization

```python
from graphgym.utils.agg_runs import agg_runs

# Aggregate results from multiple runs
agg_runs('results/experiment_name', 'val')

# Plot results
import matplotlib.pyplot as plt
import json

with open('results/experiment_name/agg/val/stats.json') as f:
    stats = json.load(f)

plt.plot(stats['epoch'], stats['loss_mean'])
plt.fill_between(stats['epoch'], 
                 stats['loss_mean'] - stats['loss_std'],
                 stats['loss_mean'] + stats['loss_std'],
                 alpha=0.3)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.savefig('training_curve.png')
```

## Best Practices on DSMLP

### 1. Organize Experiments

```bash
mkdir -p /private/<username>/graphgym
cd /private/<username>/graphgym

# Create structured directories
mkdir -p configs results datasets
```

### 2. Use Configuration Templates

Create base config and extend:
```yaml
# configs/base.yaml
out_dir: results
dataset:
  format: PyG
train:
  epochs: 200

# configs/gcn_experiment.yaml
defaults:
  - base

gnn:
  layer_type: gcnconv
```

### 3. Batch Experiments

```bash
# run_experiments.sh
#!/bin/bash

for config in configs/*.yaml; do
    echo "Running $config"
    python main.py --cfg $config --repeat 3
done
```

### 4. Monitor GPU Usage

```bash
watch -n 1 nvidia-smi
```

## Troubleshooting

### Config not found

Ensure config path is relative to run directory:
```bash
python main.py --cfg configs/example.yaml
```

### Out of memory

Reduce batch size or model size in config:
```yaml
train:
  batch_size: 1
gnn:
  dim_inner: 16
```

### Import errors

Ensure GraphGym is properly installed:
```bash
pip install --upgrade graphgym
```

## Advanced Features

### Custom Training Loop

```python
from graphgym.train import train_epoch

def custom_train(loggers, loaders, model, optimizer, scheduler):
    # Custom training logic
    for epoch in range(cfg.optim.max_epoch):
        train_epoch(loggers, loaders, model, optimizer, scheduler)
```

### Distributed Training

```bash
python -m torch.distributed.launch --nproc_per_node=2 main.py --cfg config.yaml
```

## Additional Resources

- [GraphGym Paper](https://arxiv.org/abs/2011.08843)
- [GraphGym GitHub](https://github.com/snap-stanford/GraphGym)
- [Design Space for GNNs](https://arxiv.org/abs/2011.08843)
