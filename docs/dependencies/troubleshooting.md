# Dependency Troubleshooting Guide

## Overview

This guide documents common dependency issues encountered when setting up graph learning frameworks on DSMLP and their solutions.

## CUDA-Related Issues

### Issue: CUDA version mismatch

**Symptoms:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution:**
Check CUDA version and install matching packages:
```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Install matching versions
pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### Issue: libcudart.so not found

**Symptoms:**
```
ImportError: libcudart.so.11.0: cannot open shared object file
```

**Solution:**
Add CUDA libraries to path:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Make permanent by adding to ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

## PyTorch Geometric Issues

### Issue: torch-scatter installation fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement torch-scatter
```

**Solution:**
Install from PyG wheel repository:
```bash
# Get PyTorch and CUDA versions
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")

# Install torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
```

### Issue: pyg_lib import error

**Symptoms:**
```
ImportError: cannot import name 'pyg_lib'
```

**Solution:**
Install pyg_lib separately:
```bash
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Issue: Incompatible torch-sparse version

**Symptoms:**
```
RuntimeError: Detected that PyTorch and torch-sparse were compiled with different CUDA versions
```

**Solution:**
Reinstall all PyG dependencies with matching CUDA:
```bash
pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## DGL Issues

### Issue: DGL CUDA version mismatch

**Symptoms:**
```
DGL backend not found or CUDA not available
```

**Solution:**
Install correct DGL version:
```bash
# For CUDA 11.7
pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html

# For CUDA 11.8
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

# Verify installation
python -c "import dgl; print(dgl.__version__)"
```

### Issue: DGL and PyTorch version conflict

**Symptoms:**
```
RuntimeError: version mismatch between DGL and PyTorch
```

**Solution:**
Check compatibility matrix and reinstall:
```bash
# Check versions
python -c "import torch; print(torch.__version__)"
python -c "import dgl; print(dgl.__version__)"

# Install compatible versions
pip install torch==2.0.0
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
```

## Memory Issues

### Issue: CUDA out of memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**

1. Reduce batch size:
```python
# Reduce batch size
batch_size = 16  # Instead of 32 or 64
```

2. Use gradient accumulation:
```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

3. Clear cache:
```python
import torch
torch.cuda.empty_cache()
```

4. Use gradient checkpointing:
```python
from torch.utils.checkpoint import checkpoint

# Wrap forward pass
output = checkpoint(model.layer, input)
```

### Issue: CPU memory exhausted

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. Store data on disk:
```python
# Use memory-mapped arrays
import numpy as np
data = np.load('data.npy', mmap_mode='r')
```

2. Use data streaming:
```python
from torch_geometric.loader import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Package Conflicts

### Issue: Incompatible numpy version

**Symptoms:**
```
ImportError: numpy.core.multiarray failed to import
```

**Solution:**
```bash
pip install --upgrade numpy
pip install numpy==1.23.5  # Or specific compatible version
```

### Issue: Scipy build failures

**Symptoms:**
```
ERROR: Failed building wheel for scipy
```

**Solution:**
Install pre-built binary:
```bash
pip install --upgrade pip
pip install scipy --no-cache-dir
```

### Issue: Protobuf version conflicts

**Symptoms:**
```
TypeError: Descriptors cannot not be created directly
```

**Solution:**
```bash
pip install protobuf==3.20.*
```

## Installation Best Practices

### Create Clean Environments

```bash
# Create new conda environment
conda create -n graph-learning python=3.9
conda activate graph-learning

# Install base packages
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Use Requirements Files

Create `requirements.txt`:
```
torch==2.0.0
torch-geometric==2.3.0
torch-scatter==2.1.1
torch-sparse==0.6.17
dgl==1.1.0
numpy==1.23.5
scipy==1.10.1
```

Install:
```bash
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Pin Versions

Always pin package versions for reproducibility:
```bash
pip freeze > requirements-lock.txt
```

## Debugging Tips

### Check Package Versions

```python
import torch
import torch_geometric
import dgl

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"PyG: {torch_geometric.__version__}")
print(f"DGL: {dgl.__version__}")
```

### Verify CUDA Setup

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

### Test GPU Operations

```python
import torch

x = torch.rand(5, 3)
if torch.cuda.is_available():
    x = x.cuda()
    print(f"Tensor on GPU: {x.device}")
    y = x * 2
    print(f"GPU operation successful: {y.sum()}")
```

## Getting Help

If issues persist:

1. Check error logs carefully
2. Search GitHub issues for the specific package
3. Check DSMLP support documentation
4. Post on relevant discussion forums:
   - PyG: https://github.com/pyg-team/pytorch_geometric/discussions
   - DGL: https://discuss.dgl.ai/
   - DSMLP: https://support.ucsd.edu/

## Useful Commands

```bash
# Check disk space
df -h ~
du -sh ~/.cache/pip

# Clean pip cache
pip cache purge

# Clean conda cache
conda clean --all

# List installed packages
pip list
conda list

# Check Python path
which python

# Check CUDA setup
nvcc --version
nvidia-smi
```
