# DSMLP Platform Setup Guide

## Overview

DSMLP (Data Science & Machine Learning Platform) is UCSD's dedicated platform for data science and machine learning workloads. This guide covers the setup process for running graph learning experiments.

## Prerequisites

- UCSD SSO credentials
- Access to DSMLP (request through ITS if needed)
- Basic knowledge of Docker and Kubernetes

## Initial Setup

### 1. Access DSMLP

Connect via SSH:
```bash
ssh <username>@dsmlp-login.ucsd.edu
```

### 2. Launch a Pod

Basic launch command:
```bash
launch.sh -i ucsdets/cse152-252-notebook:latest
```

For GPU access:
```bash
launch.sh -i ucsdets/cse152-252-notebook:latest -g 1
```

For more resources:
```bash
launch.sh -i ucsdets/cse152-252-notebook:latest -g 1 -m 16 -c 4
```

Parameters:
- `-i`: Docker image
- `-g`: Number of GPUs (0-2)
- `-m`: Memory in GB
- `-c`: Number of CPU cores

### 3. Storage Configuration

DSMLP provides several storage options:

- **Home directory**: `~/` (limited space, backed up)
- **Private directory**: `/private/<username>` (more space, not backed up)
- **Team directory**: `/teams/<teamname>` (shared space)

Best practices:
- Store code in home directory
- Store datasets in private or team directories
- Clean up temporary files regularly

### 4. Environment Setup

Create a conda environment:
```bash
conda create -n graph-learning python=3.9
conda activate graph-learning
```

## Custom Docker Images

For reproducibility, create custom Docker images:

1. Create a Dockerfile:
```dockerfile
FROM ucsdets/cse152-252-notebook:latest

# Install additional packages
RUN pip install torch-geometric pyg-lib torch-scatter torch-sparse

# Set working directory
WORKDIR /workspace
```

2. Build and push:
```bash
docker build -t <dockerhub-username>/graph-learning:latest .
docker push <dockerhub-username>/graph-learning:latest
```

3. Launch with custom image:
```bash
launch.sh -i <dockerhub-username>/graph-learning:latest -g 1
```

## Resource Limits

Be aware of resource limitations:
- Maximum 2 GPUs per pod
- Maximum runtime: 6 hours (can be extended)
- Fair-use policy applies

## Tips for Efficient Usage

1. **Save checkpoints frequently** - pods can be terminated
2. **Use tmux or screen** - maintain sessions during disconnections
3. **Monitor resource usage** - use `nvidia-smi` for GPU monitoring
4. **Clean up** - delete pods when finished to free resources

## Troubleshooting

### Pod Won't Start
- Check if you've exceeded quota
- Try reducing resource requirements
- Verify image exists and is accessible

### Connection Issues
- Ensure VPN is connected if off-campus
- Check UCSD SSO is active
- Try reconnecting

### Storage Full
- Check disk usage: `du -sh ~/*`
- Clean conda cache: `conda clean --all`
- Remove old datasets

## Additional Resources

- [DSMLP Support Page](https://support.ucsd.edu/services?id=kb_article_view&sys_kb_id=2770a2dadb579414947a0fa8139619f5)
- [Launch Script Documentation](https://support.ucsd.edu/services?id=kb_article_view&sysparm_article=KB0032273)
