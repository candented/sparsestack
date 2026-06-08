# WINASPARSE1BIT

**Weight-Informed Neuron Activation with Stacked 1.58-Bit Sparsity for Efficient Sparse Transformers**

[![Paper](https://img.shields.io/badge/Preprint-v2_PDF-red)](docs/WINASPARSE1BIT_arxiv_v2.pdf)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.pending-lightgrey)](https://zenodo.org/)
[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

> Reference implementation accompanying [arXiv preprint v2 (June 2026)](docs/WINASPARSE1BIT_arxiv_v2.pdf).
> v1 paper draft preserved under `docs/archive/`. See [CHANGELOG.md](CHANGELOG.md) for the v1 → v2 diff.

**Author:** J B Swann — Independent Research, Project Kingstun

---

## Overview

WINASPARSE1BIT is a sparse transformer architecture that jointly applies **three orthogonal sparsity dimensions**:

1. **1.58-bit ternary weight quantization** — weights mapped to `{-1, 0, +1}` with a per-layer learned scale `α` and a straight-through estimator for gradients
2. **WINA gating** — Weight-Informed Neuron Activation, gating MLP neurons on the joint magnitude of activations *and* outgoing weight vectors, producing a self-reinforcing sparsity feedback loop
3. **Learnable Boolean connectivity masks** — explicit circuit topology, controlled online during training by a REINFORCE policy over a discrete 800-action grid `(τ, δ, k_act, mode)`

The three axes interact **multiplicatively**, yielding superlinear FLOPs reduction.

---

## Headline results

### Locked classification baselines (multi-seed)

| Phase | Task | Dim | Params | Accuracy | FLOPs | Conn. sparsity | Efficiency |
|---|---|---:|---:|---:|---:|---:|---:|
| 7 | Lexical (WordNet) | 64 | 6K | 77.50% | 0.110 | 80.0% | 9.1× |
| **8 (Model Zero)** | **Lexical (WordNet)** | **256** | **65K** | **78.91%** | **0.066** | **88.4%** | **15.2×** |
| 9a | Lexical (WordNet) | 512 | 260K | 78.32% | 0.087 | 84.2% | 11.5× |
| 13 | FLD modus ponens | 256 | 65K | 100.00% | 0.085 | ~80% | — |
| 13 | FLD syllogism | 256 | 65K | 98.44% | 0.087 | ~80% | — |
| 13 | FLD mixed | 256 | 65K | 96.88% | 0.090 | ~80% | — |

Phase 8 multi-seed validation: **78.91% ± 0.51%** across seeds {42, 101, 777, 999}.

### New in v2 — tuned baseline via auto-phase-transfer

The `auto_phase_transfer.py` sweep produced **iter7_hifi** (single-seed, 10K steps):

| Metric | Value |
|---|---:|
| Math-CoT verification accuracy | **0.956** |
| FLOPs ratio | 0.073 |
| Connectivity sparsity (final) | 0.90 |
| Trace utilisation delta | +0.484 |

The same converged circuit transferred to FLD (frozen connectivity mask) at ~0.87 accuracy with ~33% lower FLOPs than training from scratch.

### Causal language model track (Phase 15) — work in progress

`CausalSparseTransformer` extends the architecture to autoregressive language modelling with RoPE and causal attention. v2 reports the **forward-pass and 200-step end-to-end smoke results** confirming sparse-mode LM training is functional (mode-comparison eval shows a +0.48 nat divergence between `dense` and `circuit_masked` forwards, proving the sparse path is active). No scale LM result yet.

---

## Installation

### Requirements

- Python 3.13+ (3.10+ should work but is not the development target)
- PyTorch 2.0+ with CUDA support recommended
- See `requirements.txt` for the full dependency list

### Setup

```bash
git clone https://github.com/candented/sparsestack
cd winasparse1bit

pip install -r requirements.txt

# WordNet data for the lexical task family
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## Quick start

### Reproduce the locked Phase 8 (Model Zero) baseline

```bash
# Generate the WordNet-derived lexical dataset
python tools/generate_lexical_deep.py
# → data/lexical_real_deep_v1.jsonl (99,186 examples)

# Single-seed full training
python main.py preset=model_zero_scaled_v1 training.mode=full

# Multi-seed validation
for seed in 42 101 777 999; do
  python main.py preset=model_zero_scaled_v1 training.seed=$seed
done
# Expected: mean acc 78.91% ± 0.51%, mean FLOPs 0.066 ± 0.002
```

### Reproduce the iter7_hifi tuned baseline

```bash
python main.py preset=model_zero_math_v1 \
    training.entropy_bonus=0.007 \
    annealing.lambda_final.conn=0.4 \
    annealing.lambda_final.act=0.3 \
    annealing.lambda_final.prec=0.3 \
    training.learning_rate=3e-4 \
    reward.acc_penalty_scale=2.0 \
    training.seed=42 annealing.total_steps=10000
```

### Cross-task transfer (math-CoT → FLD)

```bash
# Three-arm ablation
python cross_task_transfer.py --mode baseline --seed 42 --steps 5000
python cross_task_transfer.py --mode frozen   --seed 42 --steps 5000
python cross_task_transfer.py --mode free     --seed 42 --steps 5000
```

### Mechanistic interpretability

```bash
# Requires Phase 8 checkpoints at two seeds
python mech_interp.py
# → logs/mech_interp/results.json + per-layer kill-frequency .npy files
```

### Causal LM smoke (RTX 4060-class GPU)

```bash
python train_lm.py preset=phase15a_tinystories_poc_sparse \
    training.max_steps=200 training.log_interval=20 \
    training.eval_interval=999999 training.save_interval=999999 \
    model.dim=256 model.num_layers=2 model.mlp_hidden_dim=1024 \
    model.num_heads=4 training.batch_size=4 \
    training.gradient_accumulation_steps=1 \
    lm.seq_len=256 lm.streaming=true
# Expected: loss 10.77 → 6.84 over 200 steps, ~12 s wall-clock
```

### Run regression tests

```bash
pytest tests/ -v
```

---

## Architecture at a glance

A WINASPARSE1BIT model is a stack of transformer blocks where the MLP sublayer is replaced by a sparse gated feed-forward (`WINA_MLP`), and every linear layer is a `LowBitLinear` — ternary weights with a learned per-layer scale `α`, modulated by both a structural Boolean mask `M` and a runtime mask `M̃`.

### LowBitLinear forward

```
W_masked   = W ⊙ M ⊙ M̃                                  # mask
α          = mean|W_masked| over active entries           # per-layer scale
W_ternary  = sign(W_masked) · 𝟏[|W_masked| > δ]          # ternarize
W_eff      = (α · W_ternary).detach() + W_masked          # STE
                          − W_masked.detach()
y          = x W_effᵀ + b
```

### WINA gate

```
s_i        = mean|h_i|_b + mean|W₂,·i|_j                  # heuristic_v0
g_i        = 𝟏[i ∈ top-k(s)]   or   𝟏[s_i > ν · max(s)]
h_gated    = g ⊙ h
```

The self-reinforcing loop:

```
small ‖W₂,·i‖ → low s_i → g_i = 0 → ∂ℒ/∂W₂,·i = 0 → W₂,·i stays small
```

### Mode interface

| Mode | Mask M | Runtime mask M̃ | Ternary? | WINA gate? |
|---|:-:|:-:|:-:|:-:|
| `dense` | identity | identity | yes | off |
| `circuit_masked` | structural | identity | yes | follows preset |
| `full_sparse` | structural | τ-derived | yes | on |

All three modes ternarize — this is the fixed 1.58-bit floor. `dense` differs from the sparse modes only in mask application and gating.

For complete mathematical detail, derivations, and the Lexical Ceiling theorem, see the [v2 preprint](docs/WINASPARSE1BIT_arxiv_v2.pdf).

---

## Repository layout

```
winasparse1bit_public_repo/
├── sparse_llm/                  # Architecture
│   ├── layers.py                # LowBitLinear (1.58-bit ternary + STE + masks)
│   ├── mlp.py                   # WINA_MLP with neuron gating
│   ├── transformer.py           # SparseTransformer (classification)
│   ├── causal_transformer.py    # CausalSparseTransformer (LM, Phase 15) — NEW
│   ├── hybrid.py                # HybridStack (Sparse-Dense-Sparse)
│   ├── tiny_transformer.py      # Dense baseline
│   ├── masks.py                 # Mask utilities
│   ├── oct_layers.py            # RotationalLinear (OCT geometry tasks)
│   └── datasets/                # Dataset loaders
│       ├── lexical_real.py
│       ├── fld_loader.py        # Formal logic deduction
│       ├── math_cot_loader.py   # Math chain-of-thought verification
│       ├── lm_dataset.py        # Causal LM (TinyStories / OpenWebText)
│       └── logical.py
├── rl/                          # REINFORCE training framework
│   ├── env.py                   # ConfigEnv, action grid, FLOPs proxy
│   ├── trainer.py               # REINFORCE update loop
│   ├── policy_net.py            # Policy network
│   ├── reward_fn.py             # Shaped reward (acc + FLOPs + sparsity)
│   ├── lambda_scheduler.py      # Lambda annealing
│   ├── config_space.py / config_applier.py
│   └── summary_logger.py
├── verifiers/                   # Task verifiers
│   ├── logical_checker.py
│   ├── symbolic_checker.py
│   └── oct_manifold.py
├── config/
│   ├── global_params.yaml       # All presets (Phase 7-15)
│   ├── presets/                 # Extended preset library
│   └── loader.py
├── tools/                       # Dataset generation
│   ├── generate_lexical_deep.py
│   ├── generate_lexical_json.py
│   ├── preprocess_atomic.py
│   ├── preprocess_conceptnet.py
│   ├── merge_datasets.py
│   ├── download_atomic.sh
│   └── download_conceptnet.sh
├── tests/                       # Regression shield suite
│   ├── test_baseline_behavior.py
│   ├── test_causal_transformer.py
│   ├── test_hybrid_architecture.py
│   ├── test_parity_cot.py
│   ├── test_phase13_datasets.py
│   ├── test_lexical_real.py
│   ├── test_masks_and_flops.py
│   ├── test_reward_fn.py
│   ├── test_smoke_easy.py
│   └── test_symbolic_checker.py
├── data/
│   └── sample_data.jsonl        # Format example (full data via tools/)
├── docs/
│   ├── WINASPARSE1BIT_arxiv_v2.tex   # Preprint source
│   ├── WINASPARSE1BIT_arxiv_v2.pdf   # Preprint PDF (37 pages)
│   ├── compile_preprint.sh           # Rebuild PDF from .tex
│   └── archive/
│       └── WINASPARSE1BIT_paper_v1.tex   # v1 paper (archived)
├── main.py                      # Classification training entry point
├── train_lm.py                  # Causal LM training (single-GPU + DDP)
├── eval_lm.py                   # LM perplexity / mode-comparison eval
├── generate.py                  # LM sampling utility
├── load_checkpoint.py
├── auto_phase.py                # Tier-A auto-hyperparameter sweep
├── auto_phase_transfer.py       # Transfer-aware variant (produces iter7_hifi)
├── cross_task_transfer.py       # Three-arm frozen-circuit ablation
├── mech_interp.py               # Cross-seed circuit-overlap analysis
├── utils.py
├── CHANGELOG.md
├── CITATION.cff                 # GitHub / Zenodo citation widget
├── .zenodo.json                 # Zenodo deposit metadata
├── LICENSE                      # MIT
├── README.md                    # This file
├── FILE_MANIFEST.md             # Complete file inventory
├── REPOSITORY_INFO.md           # Scope and Zenodo workflow
└── requirements.txt
```

---

## Reproducibility

The accompanying preprint includes a standard reproducibility section listing exact commands, preset YAMLs, seeds, and expected wall-clocks for every headline result. See [docs/WINASPARSE1BIT_arxiv_v2.pdf](docs/WINASPARSE1BIT_arxiv_v2.pdf) §10.

### Hardware

| Phase | Reference hardware | Approximate runtime |
|---|---|---|
| 7 (dim=64) | RTX 4090 (24 GB) | ~30 min |
| **8 (dim=256, locked)** | **RTX 4090** | **~2 hours** |
| 9a (dim=512) | RTX 4090 | ~4 hours |
| 13 (FLD) | RTX 4000 Ada (20 GB, RunPod) | ~28 min total (~$0.12) |
| iter7_hifi | RTX 4060 Laptop (8 GB) | ~17 min |
| 14B sweep | 3× RTX 5090 (32 GB each) | ~2.5 hours |
| 15A-C (LM, planned) | RTX 4090 → 3× RTX 5090 → 3× B200 | hours to days |

### Multi-seed protocol

Phase 8 results are validated across seeds **{42, 101, 777, 999}**. Other phases use seed 42 unless specifically noted. iter7_hifi is currently single-seed (42); multi-seed validation is the prerequisite for promoting it to a frozen preset.

---

## Citing this work

If you use this code or refer to the preprint, please cite both:

**Software (via Zenodo DOI, once assigned on first GitHub release):**

```bibtex
@software{swann2026winasparse1bit_software,
  author       = {Swann, J B},
  title        = {WINASPARSE1BIT: Reference Implementation (v2-preprint-2026-06)},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v2-preprint-2026-06},
  doi          = {10.5281/zenodo.PENDING},
  url          = {https://github.com/jbswann/winasparse1bit}
}
```

**Preprint:**

```bibtex
@article{swann2026winasparse1bit,
  author  = {Swann, J B},
  title   = {WINASPARSE1BIT: Weight-Informed Neuron Activation with Stacked
             1.58-Bit Sparsity for Efficient Sparse Transformers},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026},
  month   = {6}
}
```

A `CITATION.cff` file is included for automatic citation rendering on GitHub and ingestion by Zenodo.

---

## Building the preprint locally

```bash
bash docs/compile_preprint.sh
# → docs/WINASPARSE1BIT_arxiv_v2.pdf
```

Requires a LaTeX distribution (MiKTeX on Windows, TeX Live on Linux / macOS). The compile script auto-discovers `pdflatex` in standard locations and configures MiKTeX for unattended package downloads on first run.

---

## License

[MIT](LICENSE)

---

## Contact

Issues and questions: open a [GitHub issue](https://github.com/jbswann/winasparse1bit/issues).

This is independent research conducted without institutional affiliation or external funding.
Charleston, SC, USA.
