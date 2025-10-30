# Resource Allocation Guide

## Overview

This guide helps you determine the appropriate resource allocation for different types of graph learning tasks on DSMLP.

## Resource Allocation Matrix

### Small Graphs (< 10K nodes)

**Recommended Configuration:**
```bash
launch.sh -i <image> -g 0 -c 2 -m 8
```

- GPU: Not required (CPU sufficient)
- CPU cores: 2
- Memory: 8 GB
- Best for: Small datasets, prototyping, debugging

**Use Cases:**
- Testing code
- Small citation networks (Cora, Citeseer, Pubmed)
- Tutorial examples
- Initial development

### Medium Graphs (10K - 1M nodes)

**Recommended Configuration:**
```bash
launch.sh -i <image> -g 1 -c 4 -m 16
```

- GPU: 1 (recommended)
- CPU cores: 4
- Memory: 16 GB
- Best for: Most research tasks

**Use Cases:**
- Standard benchmarks
- Social networks
- Molecular graphs
- Paper citations

### Large Graphs (1M - 100M nodes)

**Recommended Configuration:**
```bash
launch.sh -i <image> -g 2 -c 8 -m 32
```

- GPU: 2 (required)
- CPU cores: 8
- Memory: 32+ GB
- Best for: Large-scale experiments

**Use Cases:**
- Web graphs
- Knowledge graphs
- Large social networks
- Production models

### Very Large Graphs (> 100M nodes)

**Recommended Configuration:**
- Use sampling methods
- Distributed training
- May require specialized setup

**Strategies:**
- Neighbor sampling
- Mini-batch training
- Graph partitioning
- Consider cloud resources

## Task-Specific Allocations

### Node Classification

#### Transductive (Full graph)
```bash
# Small: Cora, Citeseer
launch.sh -i <image> -g 1 -c 2 -m 8

# Medium: Reddit, PPI
launch.sh -i <image> -g 1 -c 4 -m 16

# Large: Products, Papers
launch.sh -i <image> -g 2 -c 8 -m 32
```

#### Inductive (Sampling)
```bash
# Always use GPU for efficiency
launch.sh -i <image> -g 1 -c 4 -m 16
```

### Graph Classification

#### Small datasets (< 1K graphs)
```bash
launch.sh -i <image> -g 1 -c 2 -m 8
```

#### Large datasets (> 10K graphs)
```bash
launch.sh -i <image> -g 1 -c 8 -m 32
```

**Note:** Need more CPU cores for data loading

### Link Prediction

```bash
# Similar to node classification
launch.sh -i <image> -g 1 -c 4 -m 16

# Increase memory if storing many negative samples
launch.sh -i <image> -g 1 -c 4 -m 32
```

### Graph Generation

```bash
# Memory-intensive
launch.sh -i <image> -g 2 -c 4 -m 32
```

## Model-Specific Requirements

### GCN/GAT (Simple models)
- GPU: 1
- Memory: 8-16 GB
- Lightweight, fast training

### GraphSAINT/Cluster-GCN (Sampling)
- GPU: 1-2
- Memory: 16-32 GB
- Need memory for sampling

### Large Language Models on Graphs
- GPU: 2 (required)
- Memory: 32+ GB
- Very memory-intensive

### Graph Transformers
- GPU: 2 (recommended)
- Memory: 32+ GB
- Attention is expensive

## Hyperparameter Tuning Allocations

### Grid Search
```bash
# Run multiple pods in parallel
for config in configs/*.yaml; do
    launch.sh -i <image> -g 1 -c 2 -m 8 -- python train.py --cfg $config &
done
```

### Bayesian Optimization
```bash
# Single pod with more memory for tracking
launch.sh -i <image> -g 1 -c 4 -m 16
```

## Data Preprocessing Allocations

### CPU-Heavy Preprocessing
```bash
# No GPU needed, more CPU cores
launch.sh -i <image> -g 0 -c 8 -m 16
```

**Tasks:**
- Feature engineering
- Graph construction
- Data cleaning
- Statistics computation

### GPU-Heavy Preprocessing
```bash
# GPU for embeddings
launch.sh -i <image> -g 1 -c 4 -m 16
```

**Tasks:**
- Computing node embeddings
- Pre-training features
- Large matrix operations

## Runtime Estimation

### Training Time Guidelines

**Small models (2-3 layers, 64 hidden dim):**
- Small graphs: 1-5 minutes
- Medium graphs: 5-30 minutes
- Large graphs: 30 minutes - 2 hours

**Large models (4+ layers, 256+ hidden dim):**
- Small graphs: 5-15 minutes
- Medium graphs: 30 minutes - 2 hours
- Large graphs: 2-6 hours

**With hyperparameter search (x10-100):**
- Plan for multiple hours to days
- Use multiple pods in parallel

### Requesting Extended Time

```bash
# Request more than 6 hours
launch.sh -i <image> -g 1 -c 4 -m 16 -t 12
```

**When to use:**
- Large-scale training
- Extensive hyperparameter search
- Multiple experiments in sequence

## Cost-Benefit Analysis

### When to Use GPUs

**Use GPU when:**
- Training neural networks
- Large matrix operations
- Graphs with > 10K nodes
- Batch size > 32

**Skip GPU when:**
- Simple preprocessing
- Small graphs
- Classical ML methods
- Debugging/testing

### When to Request More Memory

**Indicators you need more memory:**
- "Out of memory" errors
- Killed processes
- Using large datasets
- Storing embeddings in memory

**How to reduce memory:**
- Use sampling
- Reduce batch size
- Clear cache regularly
- Use disk-based datasets

### When to Request More CPUs

**Indicators you need more CPUs:**
- Slow data loading
- `num_workers` > current CPUs
- Preprocessing bottleneck
- Parallel experiments

## Monitoring and Adjustment

### Check Current Usage

```bash
# CPU and memory
htop

# GPU
nvidia-smi

# Disk
df -h ~
```

### Adjust Resources Mid-Experiment

If you find you need more resources:

1. Save your work
2. Exit pod
3. Launch new pod with more resources
4. Resume from checkpoint

```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'checkpoint.pth')

# Resume
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

## Resource Request Templates

### Template 1: Development
```bash
# For code development and testing
launch.sh -i ucsdets/cse152-252-notebook:latest -g 0 -c 2 -m 8
```

### Template 2: Standard Training
```bash
# Most common use case
launch.sh -i ucsdets/cse152-252-notebook:latest -g 1 -c 4 -m 16
```

### Template 3: Heavy Training
```bash
# Large models or datasets
launch.sh -i ucsdets/cse152-252-notebook:latest -g 2 -c 8 -m 32
```

### Template 4: Data Processing
```bash
# CPU-intensive preprocessing
launch.sh -i ucsdets/cse152-252-notebook:latest -g 0 -c 8 -m 16
```

### Template 5: Hyperparameter Tuning
```bash
# Lightweight parallel runs
launch.sh -i ucsdets/cse152-252-notebook:latest -g 1 -c 2 -m 8
```

## Fair Use Guidelines

1. **Request only what you need** - Don't hoard resources
2. **Release resources when done** - Exit pods after completion
3. **Use appropriate time limits** - Don't request 12 hours for 1-hour jobs
4. **Share nicely** - Be considerate during peak hours
5. **Clean up** - Delete old files and datasets
6. **Monitor usage** - Check if you're using allocated resources
7. **Report issues** - Help identify resource bottlenecks

## Decision Tree

```
Do you need a GPU?
├─ No (preprocessing/testing)
│  └─ CPU only: -g 0 -c 4 -m 8
│
└─ Yes (training/inference)
   ├─ Small graph (< 10K nodes)
   │  └─ GPU optional: -g 1 -c 2 -m 8
   │
   ├─ Medium graph (10K-1M nodes)
   │  └─ Single GPU: -g 1 -c 4 -m 16
   │
   └─ Large graph (> 1M nodes)
      └─ Multiple GPUs: -g 2 -c 8 -m 32
```

## Quick Reference

| Task Type | GPU | CPU | RAM | Command Flag |
|-----------|-----|-----|-----|--------------|
| Testing | 0 | 2 | 8 | `-g 0 -c 2 -m 8` |
| Small training | 1 | 2 | 8 | `-g 1 -c 2 -m 8` |
| Standard training | 1 | 4 | 16 | `-g 1 -c 4 -m 16` |
| Large training | 2 | 8 | 32 | `-g 2 -c 8 -m 32` |
| Preprocessing | 0 | 8 | 16 | `-g 0 -c 8 -m 16` |
| Hyperparameter search | 1 | 2 | 8 | `-g 1 -c 2 -m 8` (parallel) |
