# DSC180-Benchmark

Comprehensive documentation for setting up and benchmarking graph learning repositories on UCSD's DSMLP platform.

## Overview

This repository provides:
- **Setup guides** for various graph learning frameworks on DSMLP
- **Dependency troubleshooting** documentation
- **Compute optimization** guidelines for efficient resource usage
- **Benchmark results** from conducted experiments

## Quick Start

### For New Users

1. [DSMLP Platform Setup](docs/setup/dsmlp-setup.md) - Get started on DSMLP
2. [Run Quick Start Example](examples/quickstart.py) - Test your environment
3. [Choose a Framework](docs/setup/) - PyG, DGL, or GraphGym

### Documentation

1. [Setup Guides](docs/setup/) - Installation and configuration
2. [Troubleshooting](docs/dependencies/troubleshooting.md) - Common issues
3. [Optimization](docs/compute/optimization.md) - Performance tuning
4. [Benchmarks](docs/benchmarks/) - Experiment results

## Documentation Structure

```
docs/
├── setup/              # Setup guides for different frameworks
│   ├── dsmlp-setup.md
│   ├── pyg-setup.md
│   ├── dgl-setup.md
│   └── graphgym-setup.md
├── dependencies/       # Dependency management and troubleshooting
│   ├── troubleshooting.md
│   └── common-issues.md
├── compute/           # Compute resource optimization
│   ├── optimization.md
│   └── resource-allocation.md
└── benchmarks/        # Benchmark results and templates
    ├── README.md
    ├── template.md
    └── example_node_classification_cora_gcn.md
examples/              # Example code and tutorials
├── README.md
└── quickstart.py
```

## Contributing

To add new documentation or benchmark results:
1. Follow the existing structure
2. Use the provided templates
3. Document any issues encountered and their solutions
4. Submit a pull request with clear descriptions

## Supported Frameworks

- PyTorch Geometric (PyG)
- Deep Graph Library (DGL)
- GraphGym
- Other graph learning tools

## Resources

- [DSMLP Documentation](https://support.ucsd.edu/services?id=kb_article_view&sys_kb_id=2770a2dadb579414947a0fa8139619f5)
- [PyTorch Geometric Docs](https://pytorch-geometric.readthedocs.io/)
- [DGL Documentation](https://docs.dgl.ai/)
