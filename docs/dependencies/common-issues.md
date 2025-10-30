# Common Dependency Issues

## Quick Reference Guide

This document provides a quick reference for the most common dependency issues and their solutions.

## Critical Dependencies Table

| Package | Recommended Version | CUDA Requirement | Notes |
|---------|-------------------|------------------|-------|
| PyTorch | 2.0.0+ | 11.7 or 11.8 | Base framework |
| PyG | 2.3.0+ | Match PyTorch | Graph library |
| DGL | 1.1.0+ | Match PyTorch | Alternative graph library |
| CUDA | 11.7 or 11.8 | N/A | GPU support |
| Python | 3.8-3.10 | N/A | Avoid 3.11+ for now |

## Top 10 Most Common Issues

### 1. CUDA Version Mismatch
**Quick Fix:**
```bash
python -c "import torch; print(torch.version.cuda)"
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 2. torch-scatter Not Found
**Quick Fix:**
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 3. Out of GPU Memory
**Quick Fix:**
```python
torch.cuda.empty_cache()
# Reduce batch_size in your code
```

### 4. DGL Import Error
**Quick Fix:**
```bash
pip uninstall dgl
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
```

### 5. NumPy Version Incompatibility
**Quick Fix:**
```bash
pip install numpy==1.23.5
```

### 6. Protobuf Version Error
**Quick Fix:**
```bash
pip install protobuf==3.20.*
```

### 7. Missing libcudart.so
**Quick Fix:**
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 8. Scipy Installation Failure
**Quick Fix:**
```bash
pip install scipy --no-cache-dir
```

### 9. PyG Extension Not Found
**Quick Fix:**
```bash
pip install pyg-lib torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 10. Conda Environment Conflicts
**Quick Fix:**
```bash
conda create -n fresh_env python=3.9
conda activate fresh_env
# Reinstall packages
```

## Installation Order

Follow this order to minimize conflicts:

```bash
# 1. Create environment
conda create -n graph-ml python=3.9
conda activate graph-ml

# 2. Install PyTorch with CUDA
pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install PyG
pip install torch-geometric

# 4. Install PyG dependencies
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 5. Install DGL (if needed)
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

# 6. Install other packages
pip install numpy scipy pandas matplotlib scikit-learn
```

## Environment Testing Script

Save as `test_env.py`:

```python
#!/usr/bin/env python3
"""
Test script to verify graph learning environment setup.
"""

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torch_geometric
        print(f"✓ PyTorch Geometric {torch_geometric.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch Geometric import failed: {e}")
    
    try:
        import torch_scatter
        print(f"✓ torch-scatter")
    except ImportError as e:
        print(f"✗ torch-scatter import failed: {e}")
    
    try:
        import dgl
        print(f"✓ DGL {dgl.__version__}")
    except ImportError as e:
        print(f"✗ DGL import failed: {e}")
    
    return True

def test_cuda():
    """Test CUDA availability."""
    import torch
    
    print("\nTesting CUDA...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # Test GPU operation
        x = torch.rand(5, 5).cuda()
        y = x * 2
        print(f"✓ GPU computation successful")
        
        # Check memory
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        print("⚠ CUDA not available - using CPU")

def test_pyg():
    """Test PyG functionality."""
    print("\nTesting PyG...")
    
    try:
        import torch
        from torch_geometric.data import Data
        
        edge_index = torch.tensor([[0, 1, 1, 2],
                                   [1, 0, 2, 1]], dtype=torch.long)
        x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        
        print(f"✓ Created PyG graph: {data}")
        
        if torch.cuda.is_available():
            data = data.cuda()
            print(f"✓ Moved graph to GPU")
        
    except Exception as e:
        print(f"✗ PyG test failed: {e}")

def test_dgl():
    """Test DGL functionality."""
    print("\nTesting DGL...")
    
    try:
        import torch
        import dgl
        
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 3])
        g = dgl.graph((src, dst))
        
        print(f"✓ Created DGL graph: {g}")
        
        if torch.cuda.is_available():
            g = g.to('cuda')
            print(f"✓ Moved graph to GPU")
        
    except Exception as e:
        print(f"✗ DGL test failed: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Graph Learning Environment Test")
    print("=" * 60)
    
    if test_imports():
        test_cuda()
        test_pyg()
        test_dgl()
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
```

Run the test:
```bash
python test_env.py
```

## Platform-Specific Issues

### DSMLP-Specific

**Issue: Pod terminates unexpectedly**
- Cause: 6-hour time limit or resource overuse
- Solution: Save checkpoints frequently, use tmux

**Issue: Storage quota exceeded**
- Cause: Large datasets in home directory
- Solution: Use `/private/<username>` for datasets

**Issue: GPU not accessible**
- Cause: No GPU requested in launch command
- Solution: Use `launch.sh -g 1`

### Docker Image Issues

**Issue: Packages reset after pod restart**
- Cause: Using base image without custom installations
- Solution: Create custom Docker image with dependencies

**Issue: Permission denied errors**
- Cause: Writing to read-only directories
- Solution: Use `/workspace` or `/private/` directories

## Recovery Procedures

### Complete Environment Reset

```bash
# Remove conda environment
conda deactivate
conda env remove -n graph-learning

# Clear caches
pip cache purge
conda clean --all
rm -rf ~/.cache/pip

# Start fresh
conda create -n graph-learning python=3.9
conda activate graph-learning
# Follow installation order above
```

### Partial Reset (Keep Base Packages)

```bash
# Uninstall only graph packages
pip uninstall torch-geometric pyg-lib torch-scatter torch-sparse dgl

# Reinstall
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
```

## Prevention Checklist

- [ ] Always use virtual environments
- [ ] Pin package versions in requirements.txt
- [ ] Check CUDA version before installing
- [ ] Install packages in correct order
- [ ] Test installation after each major package
- [ ] Document working configurations
- [ ] Keep notes on any workarounds needed

## Getting More Help

If you encounter an issue not listed here:

1. Run the test script to collect diagnostic info
2. Check the full [Troubleshooting Guide](troubleshooting.md)
3. Search package-specific GitHub issues
4. Post on discussion forums with error details
5. Contact DSMLP support for platform-specific issues
