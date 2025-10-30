# Contributing to DSC180-Benchmark

Thank you for your interest in contributing to DSC180-Benchmark! This guide will help you add documentation and benchmark results.

## Ways to Contribute

1. **Setup Guides** - Document new frameworks or improved setup procedures
2. **Troubleshooting** - Add solutions to dependency issues you've encountered
3. **Optimization Tips** - Share performance optimization techniques
4. **Benchmark Results** - Submit new benchmark results
5. **Bug Fixes** - Fix errors in existing documentation
6. **Improvements** - Enhance existing documentation

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone
git clone https://github.com/<your-username>/DSC180-Benchmark.git
cd DSC180-Benchmark
```

### 2. Create a Branch

```bash
git checkout -b add-<feature-name>
```

Example branch names:
- `add-pyg-troubleshooting`
- `add-cora-benchmark`
- `fix-dgl-setup`

## Contributing Documentation

### Setup Guides

Location: `docs/setup/`

1. Create a new file: `<framework>-setup.md`
2. Follow the structure of existing setup guides
3. Include:
   - Overview
   - Prerequisites
   - Installation steps
   - Verification
   - Common usage examples
   - Troubleshooting

### Dependency Documentation

Location: `docs/dependencies/`

1. Add to `troubleshooting.md` or `common-issues.md`
2. Use this format:

```markdown
### Issue: Brief description

**Symptoms:**
```
Error message or behavior
```

**Solution:**
```bash
Commands or code to fix
```

**Explanation:** Why this works
```

### Compute Optimization

Location: `docs/compute/`

1. Add tips to `optimization.md` or `resource-allocation.md`
2. Include:
   - Problem description
   - Solution/approach
   - Code examples
   - Performance impact

## Contributing Benchmarks

### 1. Run Your Experiments

- Use multiple random seeds (≥3)
- Monitor resource usage
- Save all configurations
- Document everything

### 2. Use the Template

Copy `docs/benchmarks/template.md` and fill it out completely:

```bash
cp docs/benchmarks/template.md docs/benchmarks/<task>_<dataset>_<method>.md
```

### 3. Required Information

- [ ] Complete hardware specifications
- [ ] All software versions
- [ ] Dataset information and source
- [ ] Model architecture and hyperparameters
- [ ] Results from multiple runs (mean ± std)
- [ ] Training time and resource usage
- [ ] Reproducibility instructions (code, commands)
- [ ] Configuration files

### 4. File Naming Convention

Format: `<task>_<dataset>_<method>.md`

Examples:
- `node_classification_cora_gcn.md`
- `graph_classification_mutag_gin.md`
- `link_prediction_citeseer_gae.md`

### 5. Update the Index

Add your benchmark to `docs/benchmarks/README.md` in the appropriate table.

## Documentation Standards

### Writing Style

- Use clear, concise language
- Include code examples
- Provide context and explanation
- Use proper markdown formatting
- Check spelling and grammar

### Code Examples

```python
# Include comments
# Show complete, runnable examples
# Use proper syntax highlighting

import torch
from torch_geometric.data import Data

# Example code here
```

### Command Examples

```bash
# Include comments for complex commands
# Show expected output when helpful

launch.sh -i <image> -g 1 -c 4 -m 16
```

### Formatting Guidelines

- Use headers appropriately (##, ###, ####)
- Use code blocks with language specification
- Use tables for structured data
- Use bullet points for lists
- Include links to external resources

## Pull Request Process

### 1. Commit Your Changes

```bash
git add .
git commit -m "Add: Brief description of changes"
```

Commit message examples:
- `Add: DGL setup guide for DSMLP`
- `Add: Benchmark results for GCN on Cora`
- `Fix: CUDA version mismatch solution`
- `Update: Memory optimization strategies`

### 2. Push to Your Fork

```bash
git push origin add-<feature-name>
```

### 3. Create Pull Request

1. Go to the original repository
2. Click "New Pull Request"
3. Select your branch
4. Fill out the PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] New setup guide
- [ ] New benchmark result
- [ ] Troubleshooting documentation
- [ ] Bug fix
- [ ] Documentation improvement

## Checklist
- [ ] Followed existing documentation structure
- [ ] Included code examples where appropriate
- [ ] Tested all commands/code
- [ ] Updated relevant index/README files
- [ ] Checked for spelling/grammar errors

## Additional Context
Any other relevant information
```

### 4. Review Process

- Maintainers will review your PR
- Address any requested changes
- Once approved, your changes will be merged

## Best Practices

### For Setup Guides

1. Test all commands on DSMLP
2. Include version numbers
3. Provide troubleshooting section
4. Link to official documentation
5. Keep it up-to-date

### For Benchmarks

1. Run multiple times (≥3 runs)
2. Use consistent random seeds
3. Report mean ± standard deviation
4. Include all hyperparameters
5. Make results reproducible
6. Be honest about limitations

### For Troubleshooting

1. Describe the problem clearly
2. Include error messages
3. Provide tested solutions
4. Explain why the solution works
5. Link to relevant issues/discussions

## Code of Conduct

- Be respectful and constructive
- Help others learn
- Give credit where due
- Report issues honestly
- Collaborate openly

## Questions?

If you have questions:

1. Check existing documentation
2. Search closed issues/PRs
3. Open a new issue with the `question` label
4. Reach out to maintainers

## Recognition

Contributors will be recognized in:
- Repository contributors list
- Benchmark update log
- Acknowledgments section

Thank you for contributing to DSC180-Benchmark!

## Appendix: Quick Reference

### File Structure

```
DSC180-Benchmark/
├── README.md
├── CONTRIBUTING.md
└── docs/
    ├── setup/
    │   ├── dsmlp-setup.md
    │   ├── pyg-setup.md
    │   ├── dgl-setup.md
    │   └── graphgym-setup.md
    ├── dependencies/
    │   ├── troubleshooting.md
    │   └── common-issues.md
    ├── compute/
    │   ├── optimization.md
    │   └── resource-allocation.md
    └── benchmarks/
        ├── README.md
        ├── template.md
        └── <benchmark-files>.md
```

### Useful Commands

```bash
# Check links in markdown
markdown-link-check docs/**/*.md

# Format markdown
prettier --write "docs/**/*.md"

# Preview markdown locally
grip -b README.md
```
