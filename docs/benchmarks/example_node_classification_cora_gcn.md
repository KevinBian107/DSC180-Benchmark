# Example: GCN on Cora - Node Classification

## Experiment Information

**Date:** 2024-01-15  
**Contributor:** Example User  
**Task Type:** Node Classification  
**Dataset:** Cora  
**Method:** Graph Convolutional Network (GCN)  

## Hardware & Environment

### DSMLP Configuration
```bash
launch.sh -i ucsdets/cse152-252-notebook:latest -g 1 -c 4 -m 16
```

**Hardware Specifications:**
- GPU: NVIDIA RTX 3090 (24GB)
- CPU: 4 cores
- Memory: 16 GB allocated
- Storage: /private/username/

### Software Versions
- Python: 3.9.7
- PyTorch: 2.0.0+cu118
- CUDA: 11.8
- PyTorch Geometric: 2.3.0
- torch-scatter: 2.1.1
- torch-sparse: 0.6.17

## Dataset Information

**Name:** Cora  
**Source:** [PyG Planetoid Dataset](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid)  

**Statistics:**
- Number of nodes: 2,708
- Number of edges: 10,556
- Number of features: 1,433
- Number of classes: 7

**Splits:**
- Training: 140 nodes (5.2%)
- Validation: 500 nodes (18.5%)
- Testing: 1,000 nodes (36.9%)

**Data Location:**
```
/private/username/data/Cora
```

**Preprocessing:**
```python
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='/private/username/data', 
                    name='Cora',
                    transform=NormalizeFeatures())
```

## Model Configuration

**Architecture:** Two-layer Graph Convolutional Network

**Hyperparameters:**
```yaml
model:
  hidden_dim: 16
  num_layers: 2
  dropout: 0.5
  activation: relu

training:
  learning_rate: 0.01
  weight_decay: 5e-4
  epochs: 200
  optimizer: adam
  early_stopping: false
```

**Model Size:**
- Total parameters: 23,624
- Trainable parameters: 23,624
- Model memory: ~0.1 MB

## Experimental Setup

**Random Seeds:** [1, 2, 3, 4, 5]  
**Number of Runs:** 5  
**Early Stopping:** No  
**Evaluation Metric:** Accuracy on test set  

## Results

### Performance Metrics

#### Main Results

| Metric | Mean | Std | Best | Worst |
|--------|------|-----|------|-------|
| Accuracy | 81.24 | 0.52 | 81.90 | 80.50 |
| F1-Score (Macro) | 80.18 | 0.61 | 81.02 | 79.32 |
| F1-Score (Micro) | 81.24 | 0.52 | 81.90 | 80.50 |

#### Per-Run Results

| Run | Seed | Accuracy | F1-Score | Time (s) | Memory (GB) |
|-----|------|----------|----------|----------|-------------|
| 1 | 1 | 81.20 | 80.15 | 12.4 | 0.8 |
| 2 | 2 | 80.50 | 79.32 | 12.6 | 0.8 |
| 3 | 3 | 81.90 | 81.02 | 12.3 | 0.8 |
| 4 | 4 | 81.10 | 79.98 | 12.5 | 0.8 |
| 5 | 5 | 81.50 | 80.43 | 12.4 | 0.8 |

### Training Metrics

**Training Time:**
- Per epoch: 0.062 seconds
- Total: 12.4 minutes (average)
- Convergence epoch: ~150

**Resource Usage:**
- Peak GPU memory: 0.8 GB
- Average GPU utilization: 35%
- Peak CPU usage: 25%
- Disk space used: 0.15 GB

**Convergence:**
- Best validation performance at epoch: 150 (average)
- Early stopping triggered: No
- Final training loss: 0.0124
- Final validation loss: 0.5832

## Comparison with Baselines

| Method | Accuracy | F1-Score | Time | Memory |
|--------|----------|----------|------|--------|
| GCN (This work) | 81.24 ± 0.52 | 80.18 ± 0.61 | 12.4 s | 0.8 GB |
| GCN (Paper) | 81.50 ± 0.50 | - | - | - |
| GAT | 83.00 ± 0.70 | - | 18.2 s | 1.2 GB |
| GraphSAGE | 78.50 ± 0.80 | - | 15.1 s | 1.0 GB |

**Source of baseline numbers:** 
- Original GCN paper: Kipf & Welling, ICLR 2017
- GAT: Veličković et al., ICLR 2018

## Reproducibility

### Code

**Repository:** https://github.com/example/gcn-experiments  
**Commit:** a1b2c3d4  
**Branch:** main

**Main Script:**
```bash
# Exact command to reproduce results
python train.py --model gcn --dataset cora --hidden_dim 16 --lr 0.01 --weight_decay 5e-4 --epochs 200 --seed 1
```

### Configuration Files

```python
# config.py
config = {
    'model': {
        'type': 'gcn',
        'hidden_dim': 16,
        'num_layers': 2,
        'dropout': 0.5,
    },
    'training': {
        'lr': 0.01,
        'weight_decay': 5e-4,
        'epochs': 200,
        'optimizer': 'adam',
    },
    'dataset': {
        'name': 'cora',
        'normalize_features': True,
    }
}
```

### Data Preparation

```bash
# Data is automatically downloaded by PyG
python train.py  # Will download Cora dataset on first run
```

### Environment Setup

```bash
# Installation commands
pip install torch==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.3.0
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

**Requirements:**
```
torch==2.0.0+cu118
torch-geometric==2.3.0
torch-scatter==2.1.1
torch-sparse==0.6.17
numpy==1.23.5
scikit-learn==1.2.2
```

## Analysis

### Key Observations

1. **Performance**: Achieved 81.24% accuracy, matching the original paper's results (81.5%)
2. **Stability**: Low standard deviation (0.52%) across 5 runs indicates stable training
3. **Efficiency**: Fast training (~12 seconds) with minimal GPU memory usage (0.8 GB)
4. **Convergence**: Model converges around epoch 150, suggesting 200 epochs is sufficient

### Strengths

- Fast training time suitable for rapid experimentation
- Low memory footprint allows running multiple experiments in parallel
- Consistent performance across different random seeds
- Easy to implement and reproduce

### Limitations

- Performance plateaus below state-of-the-art methods (e.g., GAT achieves 83%)
- Fixed 2-layer architecture may not capture complex patterns
- Simple message passing may be insufficient for some graph structures
- Dropout rate of 0.5 may be too aggressive for some datasets

### Potential Improvements

1. **Architecture**: Add residual connections or more layers
2. **Hyperparameters**: Grid search over hidden dimensions and learning rate
3. **Regularization**: Experiment with different dropout rates
4. **Data**: Try data augmentation techniques (edge dropping, node masking)
5. **Optimization**: Use learning rate scheduling for better convergence

## Issues Encountered

### Technical Issues

**Issue 1:** Initial CUDA out of memory error
- **Solution:** Reduced batch size from 64 to 32 (though not applicable for full-batch training)

**Issue 2:** torch-scatter installation failed
- **Solution:** Installed from PyG wheel repository with correct CUDA version:
  ```bash
  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
  ```

### Performance Issues

- **Observation**: GPU utilization relatively low (35%) due to small graph size
- **Note**: For Cora dataset, CPU-only training is also fast (~15 seconds)

## Additional Notes

- This is a standard baseline for node classification on citation networks
- Results are consistent with published benchmarks
- Code is well-tested and suitable for educational purposes
- Can be used as starting point for more complex methods

## References

1. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR 2017.
2. Sen, P., et al. (2008). Collective classification in network data. AI Magazine, 29(3), 93-106. (Cora dataset)
3. PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io/

---

**Template Version:** 1.0  
**Last Updated:** 2024-01-15
