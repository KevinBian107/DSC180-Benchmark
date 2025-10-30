# Examples

This directory contains example code for getting started with graph learning on DSMLP.

## Quick Start Example

The `quickstart.py` script demonstrates basic usage of PyTorch Geometric:

```bash
python examples/quickstart.py
```

This example:
- Loads the Cora citation network dataset
- Defines a simple 2-layer GCN model
- Trains the model for 50 epochs
- Reports test accuracy

### Expected Output

```
============================================================
Graph Learning Quick Start Example
============================================================

Device: cuda
GPU: NVIDIA RTX 3090
GPU Memory: 24.00 GB

------------------------------------------------------------
Loading Cora dataset...
Dataset: Cora()
Number of graphs: 1
Number of features: 1433
Number of classes: 7

Graph statistics:
  Nodes: 2708
  Edges: 10556
  Average degree: 3.90
  Training nodes: 140
  Validation nodes: 500
  Test nodes: 1000

------------------------------------------------------------
Initializing model...
Model: SimpleGCN
Parameters: 23624

------------------------------------------------------------
Training...
Epoch 010 | Loss: 1.2345 | Train Acc: 0.8500 | Val Acc: 0.7200
Epoch 020 | Loss: 0.8234 | Train Acc: 0.9500 | Val Acc: 0.7600
Epoch 030 | Loss: 0.5678 | Train Acc: 0.9800 | Val Acc: 0.7800
Epoch 040 | Loss: 0.4123 | Train Acc: 0.9900 | Val Acc: 0.7900
Epoch 050 | Loss: 0.3456 | Train Acc: 1.0000 | Val Acc: 0.8000

------------------------------------------------------------
Final Evaluation...
Test Accuracy: 0.8100

============================================================
Quick start completed successfully!
============================================================

Next steps:
1. Check out the documentation in docs/
2. Try different models and datasets
3. Optimize hyperparameters
4. Submit your benchmark results!
```

## Using the Example

### On DSMLP

1. Launch a pod with GPU:
```bash
launch.sh -i ucsdets/cse152-252-notebook:latest -g 1 -c 4 -m 16
```

2. Clone the repository:
```bash
git clone https://github.com/KevinBian107/DSC180-Benchmark.git
cd DSC180-Benchmark
```

3. Install dependencies:
```bash
pip install torch torch-geometric
```

4. Run the example:
```bash
python examples/quickstart.py
```

## Extending the Example

Try modifying the example to:
- Use different datasets (Citeseer, Pubmed, etc.)
- Change model architecture (more layers, different dimensions)
- Adjust hyperparameters (learning rate, dropout)
- Add validation-based early stopping
- Implement different GNN layers (GAT, GraphSAGE, etc.)

## Additional Examples

More examples coming soon:
- Graph classification with TUDataset
- Link prediction with GAE
- Large graph sampling with NeighborLoader
- Heterogeneous graphs with DGL
- Custom datasets and data loaders

## Contributing Examples

Have a useful example? Please contribute!

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
