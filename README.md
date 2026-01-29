# WINASPARSE1BIT: Triple-Sparse Transformer Architecture

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](docs/WINASPARSE1BIT_paper.pdf)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official implementation of **"WINASPARSE1BIT: A Triple-Sparse Transformer Architecture Achieving 15× Efficiency via Weight-Informed Neuron Activation"**

**Author**: J B Swann  
**Affiliation**: Independent Research  
**Project**: Kingstun

## Overview

WINASPARSE1BIT achieves **15.2× efficiency improvement** over dense transformers while maintaining **superior accuracy** (+6.71%) through triple sparsity:

1. **1.58-bit ternary weights** {-1, 0, +1}
2. **WINA gating**: Weight-Informed Neuron Activation
3. **Learnable connectivity masks**: Explicit Boolean circuit topology

### Key Results

| Model | Dim | Accuracy | FLOPs | Efficiency | Sparsity |
|-------|-----|----------|-------|------------|----------|
| TinyTransformer (Dense) | 64 | 72.2% | 0.73 | 1.0× | 0% |
| **WINASPARSE1BIT** | 256 | **78.91%** | **0.066** | **15.2×** | **88.4%** |

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NLTK (for dataset generation)

### Setup

```bash
# Clone repository
git clone https://github.com/jbswann/winasparse1bit
cd winasparse1bit

# Install dependencies
pip install -r requirements.txt

# Download NLTK WordNet data
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Quick Start

### 1. Generate Dataset

```bash
python tools/generate_lexical_deep.py
```

This creates `data/lexical_real_deep_v1.jsonl` with 99,186 lexical reasoning examples.

### 2. Train Model

**Phase 8 (Optimal Configuration - dim=256)**:
```bash
python main.py preset=model_zero_scaled_v1 training.mode=full
```

**Phase 7 (Baseline - dim=64)**:
```bash
python main.py preset=model_zero_baseline training.mode=full
```

**Phase 9a (Large Scale - dim=512)**:
```bash
python main.py preset=model_zero_scaled_v2_dim512 training.mode=full
```

### 3. Evaluate

Results are logged to `logs/` directory. Key metrics:
- **Accuracy**: Test set performance
- **FLOPs**: Computational efficiency
- **Sparsity**: Connection/activation/precision sparsity

## Architecture

### Triple Sparsity Components

**1. Low-Bit Linear Layer**:
```python
y = α · (W_ternary ⊙ M_connectivity) x + b
```
where W_ternary ∈ {-1, 0, +1}

**2. WINA Gating**:
```python
g = σ(||W_ternary||₁ / fan_in - δ)
h = g ⊙ ReLU(y)
```

**3. Sparse Attention**:
```python
A = softmax(QK^T / √d) ⊙ M_attention
```

### Model Configurations

| Config | Dim | Layers | Heads | Parameters |
|--------|-----|--------|-------|------------|
| Phase 7 | 64 | 4 | 4 | ~6K |
| Phase 8 | 256 | 4 | 4 | ~65K |
| Phase 9a | 512 | 4 | 4 | ~260K |

## Reproducing Paper Results

### Phase 8 (Production Configuration)

```bash
# Train with 4 random seeds
for seed in 42 101 777 999; do
    python main.py preset=model_zero_scaled_v1 training.seed=$seed
done

# Expected results:
# Mean Accuracy: 78.91% (±0.51%)
# Mean FLOPs: 0.066 (±0.002)
# Mean Sparsity: 88.4% (±0.5%)
```

### Baseline Comparison

```bash
# Train dense TinyTransformer baseline
python main.py preset=tiny_transformer training.mode=full

# Expected: 72.2% accuracy, 0.73 FLOPs
```

## Dataset

The lexical reasoning dataset contains 99,186 examples across 4 categories:

1. **Hypernym/Hyponym chains** (logical entailment)
2. **Meronym/Holonym relations** (part-whole)
3. **Synset traversal** (synonym/antonym)
4. **Coordinate terms** (classification)

**Format** (JSONL):
```json
{"premise": "dog", "hypothesis": "animal", "label": 1, "category": "hypernym"}
{"premise": "hot", "hypothesis": "cold", "label": 0, "category": "antonym"}
```

**Statistics**:
- Train: 79,349 examples (80%)
- Validation: 9,919 examples (10%)
- Test: 9,918 examples (10%)

## Project Structure

```
winasparse1bit/
├── sparse_llm/          # Core architecture
│   ├── layers.py        # LowBitLinear (1.58-bit)
│   ├── transformer.py   # SparseStack
│   └── tiny_transformer.py  # Dense baseline
├── rl/                  # RL training loop
│   ├── env.py          # Environment
│   └── trainer.py      # REINFORCE
├── config/
│   └── global_params.yaml  # All presets
├── tools/
│   └── generate_lexical_deep.py  # Dataset generator
├── data/               # Generated datasets
├── main.py            # Training entry point
└── docs/
    └── WINASPARSE1BIT_paper.tex  # LaTeX paper
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{swann2026winasparse1bit,
  title={WINASPARSE1BIT: A Triple-Sparse Transformer Architecture Achieving 15× Efficiency via Weight-Informed Neuron Activation},
  author={Swann, J B},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contact

**J B Swann**  
Independent Research  
Charleston, SC, USA

For questions or collaboration inquiries, please open an issue on GitHub.

## Acknowledgments

This work was conducted as independent research without institutional affiliation or funding. Thanks to the open-source community for PyTorch and related tools.

## Reproducibility

All experiments were conducted on:
- **Hardware**: NVIDIA RTX 4090 (24GB VRAM)
- **Software**: PyTorch 2.0, Python 3.10
- **Seeds**: 42, 101, 777, 999 (for multi-seed validation)

Training takes approximately:
- Phase 7 (dim=64): ~30 minutes
- Phase 8 (dim=256): ~2 hours
- Phase 9a (dim=512): ~4 hours

## Future Work

- **Hardware Optimization**: CUDA/Triton kernels for 10× wall-clock speedup
- **Hybrid Architecture**: Sparse + dense layer combinations
- **Larger Datasets**: Scaling to dim=1024+ with expanded data
- **Transfer Learning**: Evaluation on GLUE/SuperGLUE benchmarks
