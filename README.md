# Differentiable Multi-Layer VG-RAM WNN

A PyTorch implementation of **multi-layer VG-RAM Weightless Neural Networks** trainable end-to-end via backpropagation. The forward pass preserves the classic discrete VG-RAM semantics while a smooth surrogate backward pass enables gradient-based optimisation.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
   - [Traditional VG-RAM](#traditional-vg-ram)
   - [Multi-Layer Extension](#multi-layer-extension)
   - [Data Flow](#data-flow)
3. [Mathematical Formulation](#mathematical-formulation)
   - [Minchinton Cells](#minchinton-cells)
   - [Hamming Distance](#hamming-distance)
   - [Memory Selection](#memory-selection)
   - [Neuron Output](#neuron-output)
   - [Voting Aggregation](#voting-aggregation)
   - [Straight-Through Estimator](#straight-through-estimator)
   - [Binary Regularisation](#binary-regularisation)
   - [Temperature Annealing](#temperature-annealing)
4. [Project Structure](#project-structure)
5. [Module Reference](#module-reference)
6. [Installation](#installation)
7. [Usage](#usage)
8. [YAML Configuration Reference](#yaml-configuration-reference)
   - [data](#data-section)
   - [network](#network-section)
   - [training](#training-section)
   - [wandb](#wandb-section)
9. [GPU Memory Management](#gpu-memory-management)
10. [Weights & Biases Integration](#weights--biases-integration)
11. [License](#license)

---

## Overview

A **VG-RAM WNN** (Virtual Generalising Random Access Memory Weightless Neural Network) is a neural architecture where each neuron stores input–output pairs in a local memory and retrieves the output associated with the nearest stored pattern (measured by Hamming distance). Traditional VG-RAM networks are trained by one-shot memorisation and are inherently non-differentiable.

This project extends the VG-RAM WNN to **multiple layers** and makes it **trainable by backpropagation** using a **hard-forward / soft-backward** strategy:

- **Forward pass**: fully discrete — Minchinton cell comparisons produce bits, Hamming distance selects the closest memory entry, and majority voting yields a class prediction. This preserves classic VG-RAM behaviour.
- **Backward pass**: smooth surrogates replace every discrete operation so that gradients can flow through the network. Sigmoid relaxations, expected Hamming distance, and softmin attention are used as differentiable proxies.

The implementation targets **MNIST** digit classification and ships with three ready-to-use YAML configurations (tiny, small, full).

---

## Architecture

### Traditional VG-RAM

In a traditional single-layer VG-RAM WNN:

1. **Minchinton cells** compare pairs of input values to produce a binary vector.
2. Each neuron stores *M* pairs *(pattern, value)* in its memory.
3. Given an input binary vector, the neuron finds the stored pattern with the smallest **Hamming distance** and returns the associated value.
4. Training is one-shot: examples are simply stored in memory.

### Multi-Layer Extension

This implementation generalises VG-RAM to an arbitrary number of layers:

| Layer | Input | Output |
|-------|-------|--------|
| **First layer** | Raw image pixels (784 for MNIST) | 8-bit byte per neuron, converted to scalar in \[0, 1\] |
| **Intermediate layers** (0 or more, identical architecture) | Scalar outputs from the previous layer | 8-bit byte per neuron, converted to scalar in \[0, 1\] |
| **Output layer** | Scalar outputs from the previous layer | 10-class logits per neuron; aggregated by voting |

Each layer is a standard VG-RAM layer (Minchinton cells + memory lookup). The number of intermediate layers is configurable (including zero, which recovers the original two-layer architecture).

### Data Flow

```
Image (batch, 1, 28, 28)
        │
        ▼  flatten
  (batch, 784)
        │
        ▼  First Layer: Minchinton cells → bits → memory lookup → 8-bit output → scalar
  (batch, N1)
        │
        ▼  Intermediate Layer(s): same structure (repeated 0..n times)
  (batch, Ni)
        │
        ▼  Output Layer: Minchinton cells → bits → memory lookup → 10-class logits
  (batch, N_out, 10)
        │
        ▼  Aggregation: sum of neuron logits → softmax → cross-entropy loss
  (batch, 10)
```

---

## Mathematical Formulation

Every discrete operation uses the **Straight-Through Estimator** (STE) pattern: the forward pass computes the hard/discrete value while the backward pass routes gradients through a smooth surrogate.

### Minchinton Cells

Each synapse compares two input positions and produces a single bit.

**Forward (hard):**

$$b_k = \mathbf{1}[x_{p_k} > x_{q_k}]$$

**Backward (soft):**

$$\tilde{b}_k = \sigma\!\left(\frac{x_{p_k} - x_{q_k}}{\tau_b}\right)$$

where $\sigma(z) = 1/(1+e^{-z})$ and $\tau_b > 0$ is the Minchinton temperature.

### Hamming Distance

Measures the mismatch between input bits and a stored memory pattern.

**Forward (hard):**

$$d_m = \sum_{k=1}^{P} \mathbf{1}[b_k \neq a_{mk}]$$

**Backward (soft) — expected Hamming distance:**

$$\tilde{d}_m = \sum_{k=1}^{P} \left(\tilde{b}_k + \tilde{a}_{mk} - 2\,\tilde{b}_k\,\tilde{a}_{mk}\right)$$

where $\tilde{a}_{mk} = \sigma(q_{mk} / \tau_a)$ is the soft version of the stored pattern bit, parameterised by learnable logit $q_{mk}$.

**Derivatives:**

$$\frac{\partial \tilde{d}_m}{\partial \tilde{b}_k} = 1 - 2\,\tilde{a}_{mk}, \qquad \frac{\partial \tilde{d}_m}{\partial \tilde{a}_{mk}} = 1 - 2\,\tilde{b}_k$$

### Memory Selection

Selects the best-matching memory entry.

**Forward (hard):**

$$m^\star = \arg\min_m\; d_m, \qquad y = V_{m^\star}$$

**Backward (soft) — softmin attention:**

$$\alpha_m = \frac{\exp(-\beta\,\tilde{d}_m)}{\sum_r \exp(-\beta\,\tilde{d}_r)}, \qquad \tilde{y} = \sum_m \alpha_m\,\tilde{V}_m$$

where $\beta > 0$ is the inverse temperature controlling the sharpness of the selection.

### Neuron Output

**Intermediate layers:** each neuron outputs an 8-bit vector. The bits are converted to a scalar in $[0, 1]$ via positional binary weighting:

$$y_j^{\text{scalar}} = \frac{\sum_{r=0}^{7} 2^r\, \text{bit}_r}{2^8 - 1}$$

**Output layer:** each neuron outputs a 10-dimensional vector (one per class). In the hard forward this is binarised; in the soft backward it is a sigmoid-relaxed probability vector.

### Voting Aggregation

**Forward (hard):**

$$c^\star = \arg\max_c \sum_{n=1}^{N} \mathbf{1}[o_n = c]$$

**Backward (soft):**

$$s = \sum_{n=1}^{N} \tilde{o}_n, \qquad \hat{p} = \mathrm{softmax}(s), \qquad \mathcal{L} = -\log \hat{p}_{y^\star}$$

### Straight-Through Estimator

All hard/soft pairs are connected via the STE pattern:

$$z = z_{\text{soft}} + \operatorname{stop\_gradient}(z_{\text{hard}} - z_{\text{soft}})$$

In PyTorch: `soft + (hard - soft).detach()`. The forward value equals `hard`; the backward gradient flows through `soft`.

### Binary Regularisation

To encourage memory patterns and intermediate-layer outputs to remain close to binary values:

$$R = \lambda \cdot \mathrm{mean}\!\left(\tilde{v} \cdot (1 - \tilde{v})\right)$$

This penalty is zero when $\tilde{v} \in \{0, 1\}$ and maximal at $\tilde{v} = 0.5$.

### Temperature Annealing

During training, temperatures are annealed exponentially so the soft backward starts smooth and gradually approaches hard behaviour:

$$\tau(e) = \tau_{\text{start}} \cdot \left(\frac{\tau_{\text{end}}}{\tau_{\text{start}}}\right)^{e/(E-1)}$$

- $\tau_b$ and $\tau_a$ **decrease** (sharper sigmoids).
- $\beta$ **increases** (sharper softmin selection).

---

## Project Structure

```
vg-ram-backpropagation/
├── README.md                     This file
├── requirements.txt              Python dependencies
├── .gitignore
├── train.py                      Main training and evaluation script
├── configs/
│   ├── mnist_tiny.yaml           100 samples, batch 10
│   ├── mnist_small.yaml          900 train / 100 test, batch 10
│   └── mnist_full.yaml           50k train / 10k val / 10k test
└── vgram/
    ├── __init__.py               Package exports
    ├── functional.py             STE ops, Hamming, softmin, regularisation
    ├── minchinton.py             MinchintonLayer module
    ├── memory.py                 VGRAMMemory module (with chunking support)
    ├── layer.py                  VGRAMLayer (Minchinton + memory)
    ├── network.py                VGRAMNetwork (multi-layer + aggregation)
    ├── data.py                   MNIST loading with balanced sampling
    └── annealing.py              TemperatureScheduler
```

---

## Module Reference

### `vgram/functional.py`

Pure functions implementing the hard-forward / soft-backward building blocks:

| Function | Purpose |
|----------|---------|
| `ste(hard, soft)` | Straight-through estimator |
| `minchinton_compare(u, v, tau)` | Bit comparison with sigmoid STE |
| `hard_hamming(b, a)` | Discrete Hamming distance |
| `expected_hamming(b_soft, a_soft)` | Differentiable expected Hamming |
| `soft_memory_select(d, v, beta)` | Softmin-weighted memory readout |
| `hard_memory_select(d, v)` | Argmin memory readout |
| `bits_to_scalar(bits)` | Binary vector to \[0, 1\] scalar |
| `binary_regularization(soft)` | Penalty pushing values toward 0 or 1 |

### `vgram/minchinton.py` — `MinchintonLayer`

Vectorised Minchinton-cell layer. For *N* neurons with *P* synapses each, randomly pairs input positions at construction time (stored as fixed buffers) and produces binary vectors via STE.

### `vgram/memory.py` — `VGRAMMemory`

Stores learnable memory patterns (`pattern_logits`) and values (`value_logits`) for *N* neurons with *M* entries each. Supports **neuron chunking** and **gradient checkpointing** to control peak GPU memory.

### `vgram/layer.py` — `VGRAMLayer`

Combines a `MinchintonLayer` and a `VGRAMMemory`. Intermediate layers convert the multi-bit output to a scalar in \[0, 1\]; the output layer keeps raw class logits.

### `vgram/network.py` — `VGRAMNetwork`

Stacks an arbitrary number of `VGRAMLayer` modules. The last layer's outputs are aggregated by majority voting (hard) or summed logits (soft). Provides `regularization_loss()` for binary regularisation.

### `vgram/data.py`

Downloads MNIST via torchvision and builds DataLoaders with balanced per-class sampling (for small configs) or sequential splitting (for large configs).

### `vgram/annealing.py` — `TemperatureScheduler`

Exponentially anneals `tau_b`, `tau_a`, and `beta` across all layers at each epoch.

---

## Installation

**Requirements:** Python 3.8+ and a working PyTorch installation.

```bash
git clone https://github.com/albertodesouza/vg-ram-backpropagation.git
cd vg-ram-backpropagation
pip install -r requirements.txt
```

The dependencies are:

```
torch>=2.0.0
torchvision>=0.15.0
pyyaml>=6.0
wandb>=0.15.0        # optional, for experiment tracking
```

MNIST is downloaded automatically on the first run.

---

## Usage

```bash
python3 train.py --config configs/mnist_tiny.yaml
```

### Provided Configurations

| Config | Train | Val | Test | Batch | Epochs | Description |
|--------|-------|-----|------|-------|--------|-------------|
| `mnist_tiny.yaml` | 100 | — | 100 (same) | 10 | 200 | Quick sanity check; 10 samples per class |
| `mnist_small.yaml` | 50,000 | 10,000 | 10,000 | 20 | 200 | Medium-scale experiment |
| `mnist_full.yaml` | 50,000 | 10,000 | 10,000 | 20 | 200 | Full-scale with larger network |

Each config is self-contained — it specifies data, network architecture, training hyperparameters, and optional W&B settings.

---

## YAML Configuration Reference

### `data` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str | `mnist` | Dataset name |
| `train_samples` | int | 60000 | Number of training samples |
| `val_samples` | int | 0 | Number of validation samples (carved from train set) |
| `test_samples` | int | 10000 | Number of test samples |
| `batch_size` | int | — | Mini-batch size |
| `same_train_test` | bool | false | If true, test set equals training set |

When `train_samples + val_samples >= 60000`, the code uses sequential splitting instead of balanced per-class sampling.

### `network` Section

```yaml
network:
  input_size: 784
  num_classes: 10

  first_layer:
    num_neurons: 64
    num_synapses: 32
    num_entries: 100
    output_dim: 8

  intermediate_layers:
    count: 0                    # number of identical intermediate layers
    num_neurons: 48
    num_synapses: 32
    num_entries: 50
    output_dim: 8

  output_layer:
    num_neurons: 32
    num_synapses: 32
    num_entries: 100
    output_dim: 10
```

**Layer parameters:**

| Parameter | Description |
|-----------|-------------|
| `num_neurons` | Number of neurons *N* in the layer |
| `num_synapses` | Number of Minchinton synapses *P* per neuron |
| `num_entries` | Number of memory entries *M* per neuron |
| `output_dim` | Dimension of stored values (8 for bytes, 10 for classes) |
| `neuron_chunk_size` | Process neurons in chunks of this size (0 = no chunking) |
| `use_grad_checkpoint` | Enable gradient checkpointing (true/false) |

Setting `intermediate_layers.count` to 0 gives a two-layer network (first + output). Setting it to *n* inserts *n* identical intermediate layers, each with independent weights.

### `training` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 50 | Number of training epochs |
| `lr` | float | 0.01 | Learning rate (Adam optimiser) |
| `tau_b_start` / `tau_b_end` | float | 1.0 / 0.1 | Minchinton sigmoid temperature range |
| `tau_a_start` / `tau_a_end` | float | 1.0 / 0.1 | Memory-pattern sigmoid temperature range |
| `beta_start` / `beta_end` | float | 1.0 / 10.0 | Softmin inverse temperature range |
| `lambda_bin_mem` | float | 0.001 | Binary regularisation weight for memory patterns |
| `lambda_bin_out` | float | 0.001 | Binary regularisation weight for intermediate outputs |
| `device` | str | `auto` | `cpu`, `cuda`, or `auto` (auto-detect) |
| `seed` | int | 42 | Random seed |

### `wandb` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Enable Weights & Biases logging |
| `project` | str | `vgram-wnn` | W&B project name |
| `run_name` | str | — | Display name for the run |
| `tags` | list | \[\] | Tags for filtering runs |
| `notes` | str | — | Free-text notes |

---

## GPU Memory Management

The dominant memory consumer is the intermediate tensor of shape `(batch, N, M, P)` created during the Hamming distance computation. Two mechanisms are available to reduce peak memory:

### Neuron Chunking

Instead of processing all *N* neurons simultaneously, they are processed in chunks of `neuron_chunk_size`. This reduces peak memory by a factor of *N / chunk_size*.

```yaml
first_layer:
  neuron_chunk_size: 32       # process 32 neurons at a time
```

Set to `0` (or omit) to disable chunking.

### Gradient Checkpointing

When enabled, forward activations are freed after each chunk and recomputed during the backward pass, trading ~30% more compute for significantly less memory.

```yaml
first_layer:
  use_grad_checkpoint: true
```

### Device Selection

Choose CPU or GPU per experiment:

```yaml
training:
  device: cpu      # cpu | cuda | auto
```

For networks with large memory banks (*M* > 500), CPU may avoid GPU memory limitations at the cost of slower computation.

---

## Weights & Biases Integration

To enable W&B tracking, set `enabled: true` in the `wandb` section of your YAML config:

```yaml
wandb:
  enabled: true
  project: vgram-wnn
  run_name: my-experiment
```

On first use, authenticate with:

```bash
wandb login
```

**Logged metrics (per epoch):** `train/loss`, `train/accuracy`, `train/reg_loss`, `val/loss`, `val/accuracy`, `schedule/tau_b`, `schedule/tau_a`, `schedule/beta`, `epoch_time_s`.

**Logged at end of training:** `test/loss`, `test/accuracy`, `final_train/loss`, `final_train/accuracy`.

---

## License

This project is provided for research purposes. See the repository for licence details.
