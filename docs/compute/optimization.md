# Compute Optimization Guide for DSMLP

## Overview

This guide provides best practices for optimizing compute resource usage on DSMLP when running graph learning experiments.

## Understanding DSMLP Resources

### Available Resources

- **GPUs**: Tesla V100, RTX 2080 Ti, RTX 3090 (varies by availability)
- **CPUs**: Multi-core processors (4-32 cores)
- **Memory**: Up to 128GB RAM per pod
- **Storage**: Home directory (limited), `/private/` (larger quota)

### Resource Limits

- Maximum 2 GPUs per pod
- Maximum 6-hour runtime (default)
- Fair-use policy for shared resources

## GPU Optimization

### 1. Choose the Right GPU

Request specific GPU types when available:
```bash
# For training
launch.sh -i <image> -g 1 -G GeForceRTX3090

# For inference
launch.sh -i <image> -g 1 -G TeslaV100
```

### 2. Monitor GPU Usage

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# In Python
import torch
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
```

### 3. Optimize Batch Size

Find optimal batch size:
```python
def find_optimal_batch_size(model, device, min_size=1, max_size=512):
    """Binary search for optimal batch size."""
    import torch
    from torch_geometric.data import Data
    
    batch_size = max_size
    while batch_size >= min_size:
        try:
            # Create dummy batch
            x = torch.randn(batch_size * 100, 64).to(device)
            edge_index = torch.randint(0, batch_size * 100, (2, batch_size * 1000)).to(device)
            
            # Test forward pass
            with torch.no_grad():
                output = model(x, edge_index)
            
            print(f"✓ Batch size {batch_size} works")
            return batch_size
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                batch_size //= 2
                print(f"Trying smaller batch size: {batch_size}")
            else:
                raise e
    
    return min_size
```

### 4. Use Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast():
            output = model(batch)
            loss = criterion(output, target)
        
        # Backward pass with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 5. Gradient Accumulation

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    # Scale loss by accumulation steps
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    # Update weights every N steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 6. Clear GPU Cache

```python
import torch
import gc

# Clear Python garbage
gc.collect()

# Clear CUDA cache
torch.cuda.empty_cache()

# Reset peak memory stats
torch.cuda.reset_peak_memory_stats()
```

## Memory Optimization

### 1. Use Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class EfficientGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(...)
        self.conv2 = GraphConv(...)
    
    def forward(self, x, edge_index):
        # Checkpoint expensive operations
        x = checkpoint(self.conv1, x, edge_index)
        x = checkpoint(self.conv2, x, edge_index)
        return x
```

### 2. Neighbor Sampling for Large Graphs

```python
from torch_geometric.loader import NeighborLoader

# Sample neighbors instead of using full graph
loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],  # 2-hop neighbors
    batch_size=1024,
    shuffle=True,
    num_workers=4
)

for batch in loader:
    output = model(batch.x, batch.edge_index)
```

### 3. Data Loading Optimization

```python
from torch_geometric.loader import DataLoader

# Optimize data loading
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,           # Parallel loading
    pin_memory=True,         # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

### 4. Use In-Memory Datasets

```python
from torch_geometric.data import InMemoryDataset

class MyDataset(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root)
        # Load all data into memory at once
        self.data, self.slices = torch.load(self.processed_paths[0])
```

## CPU Optimization

### 1. Request Appropriate CPU Cores

```bash
# Request more cores for data preprocessing
launch.sh -i <image> -g 1 -c 8

# Use cores for parallel data loading
loader = DataLoader(..., num_workers=8)
```

### 2. Optimize Data Preprocessing

```python
import multiprocessing as mp

def preprocess_graph(graph):
    # Your preprocessing logic
    return processed_graph

# Parallel preprocessing
with mp.Pool(processes=8) as pool:
    processed_graphs = pool.map(preprocess_graph, raw_graphs)
```

### 3. Use Efficient Data Structures

```python
# Use sparse matrices for adjacency
from scipy.sparse import csr_matrix
import torch

# Convert to COO format for PyG
sparse_adj = csr_matrix(adj_matrix)
edge_index = torch.tensor(np.vstack(sparse_adj.nonzero()), dtype=torch.long)
```

## Storage Optimization

### 1. Organize Data Efficiently

```bash
# Use /private/ for datasets
mkdir -p /private/<username>/datasets
mkdir -p /private/<username>/checkpoints
mkdir -p /private/<username>/results

# Symlink to home directory
ln -s /private/<username>/datasets ~/datasets
```

### 2. Compress Datasets

```python
import pickle
import gzip

# Save compressed
with gzip.open('dataset.pkl.gz', 'wb') as f:
    pickle.dump(dataset, f)

# Load compressed
with gzip.open('dataset.pkl.gz', 'rb') as f:
    dataset = pickle.load(f)
```

### 3. Clean Up Regularly

```bash
# Check disk usage
du -sh ~/* /private/<username>/*

# Clean pip cache
pip cache purge

# Clean conda cache
conda clean --all

# Remove old checkpoints
find ~/checkpoints -mtime +30 -delete
```

## Training Optimization

### 1. Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

# Reduce on plateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)
    scheduler.step(val_loss)
```

### 2. Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

# Usage
early_stopping = EarlyStopping(patience=20)
for epoch in range(num_epochs):
    val_loss = validate(...)
    if early_stopping(val_loss):
        print("Early stopping triggered")
        break
```

### 3. Checkpoint Saving

```python
import os

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    
    # Save best model separately
    if loss < best_loss:
        best_path = filepath.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)

# Save every N epochs
if epoch % 10 == 0:
    save_checkpoint(model, optimizer, epoch, loss, 
                   f'/private/{username}/checkpoints/model_epoch_{epoch}.pth')
```

## Experiment Management

### 1. Use Configuration Files

```python
import yaml

config = {
    'model': {
        'hidden_dim': 64,
        'num_layers': 3,
        'dropout': 0.5
    },
    'training': {
        'lr': 0.01,
        'batch_size': 32,
        'epochs': 200
    }
}

# Save config
with open('config.yaml', 'w') as f:
    yaml.dump(config, f)

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

### 2. Log Everything

```python
import logging
from datetime import datetime

# Setup logging
log_file = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Log metrics
logging.info(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}")
```

### 3. Use Weights & Biases (Optional)

```python
import wandb

# Initialize
wandb.init(project="graph-learning", config=config)

# Log metrics
wandb.log({
    'epoch': epoch,
    'loss': loss,
    'accuracy': accuracy,
    'learning_rate': optimizer.param_groups[0]['lr']
})
```

## Resource Monitoring Script

```python
#!/usr/bin/env python3
import torch
import psutil
import time

def monitor_resources(interval=5):
    """Monitor and log resource usage."""
    while True:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # GPU usage
        if torch.cuda.is_available():
            gpu_mem_alloc = torch.cuda.memory_allocated(0) / 1024**3
            gpu_mem_cached = torch.cuda.memory_reserved(0) / 1024**3
            
            print(f"CPU: {cpu_percent}% | "
                  f"RAM: {memory.percent}% | "
                  f"GPU Mem: {gpu_mem_alloc:.2f}/{gpu_mem_cached:.2f} GB")
        else:
            print(f"CPU: {cpu_percent}% | RAM: {memory.percent}%")
        
        time.sleep(interval)

if __name__ == "__main__":
    monitor_resources()
```

## Best Practices Summary

1. **Request only what you need** - Don't over-allocate resources
2. **Monitor usage** - Use nvidia-smi and monitoring tools
3. **Save frequently** - Checkpoints every N epochs
4. **Clean up** - Delete old files and clear caches
5. **Use batch processing** - Efficient data loading
6. **Optimize memory** - Gradient checkpointing, mixed precision
7. **Log everything** - Track experiments and metrics
8. **Test on small data first** - Verify before full runs
9. **Use tmux/screen** - Persist sessions
10. **Document configurations** - Reproducibility is key
