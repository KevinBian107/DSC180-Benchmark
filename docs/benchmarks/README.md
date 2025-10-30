# Benchmark Results

## Overview

This directory contains benchmark results for various graph learning methods on different datasets. Results are organized by task type and dataset.

## Benchmark Categories

### 1. Node Classification
- Citation networks (Cora, Citeseer, Pubmed)
- Social networks (Reddit, Twitter)
- Product networks (Amazon, OGB)

### 2. Graph Classification
- Molecular graphs (MUTAG, PROTEINS)
- Social networks (IMDB, COLLAB)
- Bioinformatics (ENZYMES, DD)

### 3. Link Prediction
- Knowledge graphs
- Social networks
- Citation networks

### 4. Graph Generation
- Molecular generation
- Social network generation

## How to Use This Directory

### Adding New Benchmark Results

1. Use the [benchmark template](template.md)
2. Create a new file: `<task>_<dataset>_<method>.md`
3. Fill in all required sections
4. Include reproducibility information

### Example Naming
- `node_classification_cora_gcn.md`
- `graph_classification_mutag_gin.md`
- `link_prediction_cora_gae.md`

## Benchmark Standards

### Required Information

Each benchmark result should include:

1. **Experiment Setup**
   - Hardware specifications
   - Software versions
   - Random seeds used
   - Hyperparameters

2. **Results**
   - Mean and standard deviation
   - Multiple runs (at least 3)
   - Training time
   - Memory usage

3. **Reproducibility**
   - Code repository
   - Configuration files
   - Data preprocessing steps
   - Exact commands used

### Metrics to Report

#### Node Classification
- Accuracy
- F1-score (macro/micro)
- Training time per epoch
- Total training time
- GPU memory usage

#### Graph Classification
- Accuracy
- F1-score
- ROC-AUC
- Training time
- Inference time

#### Link Prediction
- ROC-AUC
- Average Precision
- Hits@K
- MRR (Mean Reciprocal Rank)

## Existing Benchmarks

### Node Classification

| Dataset | Method | Accuracy | F1-Score | Time | Notes |
|---------|--------|----------|----------|------|-------|
| Cora | GCN | - | - | - | [Details](node_classification_cora_gcn.md) |
| Cora | GAT | - | - | - | [Details](node_classification_cora_gat.md) |
| Citeseer | GCN | - | - | - | [Details](node_classification_citeseer_gcn.md) |

### Graph Classification

| Dataset | Method | Accuracy | F1-Score | Time | Notes |
|---------|--------|----------|----------|------|-------|
| MUTAG | GIN | - | - | - | [Details](graph_classification_mutag_gin.md) |
| PROTEINS | GCN | - | - | - | [Details](graph_classification_proteins_gcn.md) |

### Link Prediction

| Dataset | Method | ROC-AUC | AP | Time | Notes |
|---------|--------|---------|-----|------|-------|
| Cora | GAE | - | - | - | [Details](link_prediction_cora_gae.md) |

## Contributing

### Before Running Benchmarks

1. Verify environment setup
2. Check GPU availability
3. Ensure datasets are properly loaded
4. Set random seeds for reproducibility

### During Benchmarking

1. Run multiple times (≥3 runs)
2. Monitor resource usage
3. Save all outputs and logs
4. Take notes on any issues

### After Benchmarking

1. Fill out the template completely
2. Include all code and configs
3. Verify results are reproducible
4. Submit via pull request

## Benchmark Guidelines

### Reproducibility Checklist

- [ ] Random seeds documented
- [ ] All hyperparameters listed
- [ ] Software versions specified
- [ ] Hardware specs provided
- [ ] Code/config files included
- [ ] Data preprocessing described
- [ ] Multiple runs completed
- [ ] Statistics computed (mean, std)

### Fair Comparison

When comparing methods:
- Use same dataset splits
- Use same evaluation metrics
- Report same statistics
- Use same hardware (if possible)
- Document any differences

### Reporting Negative Results

We encourage reporting:
- Failed experiments
- Unexpected results
- Performance degradation
- Implementation issues

This helps others avoid similar problems.

## Common Benchmarking Mistakes

1. **Single run** - Always run multiple times
2. **No random seeds** - Makes results irreproducible
3. **Missing hyperparameters** - Can't reproduce results
4. **Inconsistent metrics** - Use standard metrics
5. **No error bars** - Report mean ± std
6. **Cherry-picking** - Report all results honestly

## Resources

### Standard Benchmarks
- [Open Graph Benchmark (OGB)](https://ogb.stanford.edu/)
- [PyG Datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)
- [DGL Datasets](https://docs.dgl.ai/api/python/dgl.data.html)

### Evaluation Tools
- [PyG transforms](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html)
- [Scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

### Best Practices
- [ML Reproducibility Checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)
- [Papers with Code](https://paperswithcode.com/)

## Questions?

If you have questions about:
- How to run benchmarks
- Which metrics to use
- How to report results

Please open an issue or contact the maintainers.

## Update Log

Track when benchmarks are added or updated:

| Date | Dataset | Method | Contributor | Notes |
|------|---------|--------|-------------|-------|
| - | - | - | - | - |
