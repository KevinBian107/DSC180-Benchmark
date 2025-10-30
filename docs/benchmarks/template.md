# Benchmark Report Template

## Experiment Information

**Date:** YYYY-MM-DD  
**Contributor:** Your Name  
**Task Type:** [Node Classification / Graph Classification / Link Prediction / Other]  
**Dataset:** Dataset Name  
**Method:** Method/Model Name  

## Hardware & Environment

### DSMLP Configuration
```bash
launch.sh command used:
```

**Hardware Specifications:**
- GPU: [Model, e.g., RTX 3090]
- CPU: [Cores allocated]
- Memory: [GB allocated]
- Storage: [Location used, e.g., /private/]

### Software Versions
- Python: X.X.X
- PyTorch: X.X.X
- CUDA: XX.X
- PyTorch Geometric: X.X.X (if applicable)
- DGL: X.X.X (if applicable)
- Other relevant packages:

## Dataset Information

**Name:** Dataset Name  
**Source:** [Link or citation]  
**Statistics:**
- Number of nodes: X
- Number of edges: X
- Number of features: X
- Number of classes: X
- Number of graphs: X (for graph classification)

**Splits:**
- Training: XX%
- Validation: XX%
- Testing: XX%

**Data Location:**
```
/path/to/dataset
```

**Preprocessing:**
```python
# Describe any preprocessing steps
```

## Model Configuration

**Architecture:** [GCN, GAT, GIN, etc.]

**Hyperparameters:**
```yaml
# Copy your configuration here
model:
  hidden_dim: 64
  num_layers: 2
  dropout: 0.5

training:
  learning_rate: 0.01
  weight_decay: 5e-4
  epochs: 200
  batch_size: 32
  optimizer: adam

other:
  # Any other relevant hyperparameters
```

**Model Size:**
- Total parameters: X
- Trainable parameters: X
- Model memory: X MB/GB

## Experimental Setup

**Random Seeds:** [1, 2, 3, 4, 5] (list all seeds used)  
**Number of Runs:** X  
**Early Stopping:** Yes/No (patience: X)  
**Evaluation Metric:** Primary metric used for model selection  

## Results

### Performance Metrics

#### Main Results

| Metric | Mean | Std | Best | Worst |
|--------|------|-----|------|-------|
| Accuracy | XX.XX | X.XX | XX.XX | XX.XX |
| F1-Score (Macro) | XX.XX | X.XX | XX.XX | XX.XX |
| F1-Score (Micro) | XX.XX | X.XX | XX.XX | XX.XX |
| [Other metrics] | XX.XX | X.XX | XX.XX | XX.XX |

#### Per-Run Results

| Run | Seed | Accuracy | F1-Score | Time (s) | Memory (GB) |
|-----|------|----------|----------|----------|-------------|
| 1 | 1 | XX.XX | XX.XX | XXX | X.X |
| 2 | 2 | XX.XX | XX.XX | XXX | X.X |
| 3 | 3 | XX.XX | XX.XX | XXX | X.X |
| ... | ... | ... | ... | ... | ... |

### Training Metrics

**Training Time:**
- Per epoch: X.XX seconds
- Total: X.XX minutes/hours
- Convergence epoch: X

**Resource Usage:**
- Peak GPU memory: X.XX GB
- Average GPU utilization: XX%
- Peak CPU usage: XX%
- Disk space used: X.XX GB

**Convergence:**
- Best validation performance at epoch: X
- Early stopping triggered: Yes/No
- Final training loss: X.XXXX
- Final validation loss: X.XXXX

### Learning Curves

```
Include plots or describe training/validation curves:
- Training loss curve
- Validation loss curve
- Validation accuracy curve
```

## Comparison with Baselines

| Method | Accuracy | F1-Score | Time | Memory |
|--------|----------|----------|------|--------|
| This work | XX.XX ± X.XX | XX.XX ± X.XX | XXX s | X.X GB |
| Baseline 1 | XX.XX ± X.XX | XX.XX ± X.XX | XXX s | X.X GB |
| Baseline 2 | XX.XX ± X.XX | XX.XX ± X.XX | XXX s | X.X GB |

**Source of baseline numbers:** [Citation or link]

## Reproducibility

### Code

**Repository:** [Link to code repository]  
**Commit:** [Specific commit hash]  
**Branch:** [Branch name if not main]

**Main Script:**
```bash
# Exact command to reproduce results
python train.py --config configs/experiment.yaml --seed 1
```

### Configuration Files

```yaml
# Include full configuration file contents
# or link to config file in repository
```

### Data Preparation

```bash
# Commands to download/prepare data
python prepare_data.py --dataset cora --output /private/username/data
```

### Environment Setup

```bash
# Installation commands
pip install -r requirements.txt

# Or
conda env create -f environment.yml
```

**Requirements:**
```
torch==2.0.0
torch-geometric==2.3.0
# ... list all dependencies with versions
```

## Analysis

### Key Observations

1. [Observation 1]
2. [Observation 2]
3. [Observation 3]

### Strengths

- [Strength 1]
- [Strength 2]

### Limitations

- [Limitation 1]
- [Limitation 2]

### Potential Improvements

- [Idea 1]
- [Idea 2]

## Issues Encountered

### Technical Issues

**Issue 1:** Description
- **Solution:** How it was resolved

**Issue 2:** Description
- **Solution:** How it was resolved

### Performance Issues

- Any unexpected behaviors
- Performance bottlenecks
- Memory issues

## Additional Notes

- Any other relevant information
- Special considerations
- Future work suggestions

## References

1. [Paper/Dataset citation]
2. [Method citation]
3. [Other relevant references]

## Appendix

### Detailed Logs

```
Include relevant log snippets if needed
```

### Additional Plots

[Include or link to additional visualization]

### Hyperparameter Search

If hyperparameter search was performed:

| Config | Accuracy | F1-Score | Notes |
|--------|----------|----------|-------|
| Config 1 | XX.XX | XX.XX | - |
| Config 2 | XX.XX | XX.XX | - |

---

**Template Version:** 1.0  
**Last Updated:** YYYY-MM-DD
