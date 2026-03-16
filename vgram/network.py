"""
Multi-layer VG-RAM Weightless Neural Network.

Forward is fully discrete (classic VG-RAM semantics).
Backward uses smooth surrogates so the network can be trained end-to-end
with standard gradient-based optimisers.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import binary_regularization, ste
from .layer import VGRAMLayer


class VGRAMNetwork(nn.Module):
    """Stack of VG-RAM layers ending with a classification head.

    The last layer must have ``output_dim == num_classes`` and acts as the
    voting layer.  All preceding layers convert their multi-bit output to
    a scalar in [0, 1] that becomes the input of the next layer.

    Args:
        layer_configs: list of dicts, each containing the kwargs for
                       :class:`VGRAMLayer` **except** ``input_size`` and
                       ``is_output``, which are inferred automatically.
        input_size:    dimensionality of the raw (flat) input (e.g. 784).
        num_classes:   number of output classes (default 10 for MNIST).
    """

    def __init__(
        self,
        layer_configs: list[dict[str, Any]],
        input_size: int = 784,
        num_classes: int = 10,
    ):
        super().__init__()
        self.num_classes = num_classes

        layers: list[VGRAMLayer] = []
        in_size = input_size

        for i, cfg in enumerate(layer_configs):
            is_last = i == len(layer_configs) - 1
            layer = VGRAMLayer(
                input_size=in_size,
                num_neurons=cfg["num_neurons"],
                num_synapses=cfg["num_synapses"],
                num_entries=cfg["num_entries"],
                output_dim=cfg["output_dim"],
                tau_b=cfg.get("tau_b", 1.0),
                tau_a=cfg.get("tau_a", 1.0),
                beta=cfg.get("beta", 1.0),
                is_output=is_last,
                neuron_chunk_size=cfg.get("neuron_chunk_size", 0),
                use_grad_checkpoint=cfg.get("use_grad_checkpoint", False),
            )
            layers.append(layer)
            in_size = cfg["num_neurons"] if not is_last else in_size

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, C, H, W) or (batch, input_size).

        Returns:
            logits: (batch, num_classes) — soft logits for the loss,
                    but the hard prediction equals argmax of discrete votes.
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        for layer in self.layers[:-1]:
            x = layer(x)                 # (batch, N_i)

        last_out = self.layers[-1](x)    # (batch, N_last, num_classes)

        # --- Aggregation ---
        # Hard: majority vote
        hard_votes = last_out.detach().argmax(dim=-1)            # (batch, N_last)
        hard_counts = torch.zeros(
            x.size(0), self.num_classes,
            device=last_out.device, dtype=last_out.dtype,
        )
        for c in range(self.num_classes):
            hard_counts[:, c] = (hard_votes == c).float().sum(dim=-1)

        # Soft: sum of soft neuron outputs
        soft_logits = last_out.sum(dim=1)                        # (batch, num_classes)

        return ste(hard_counts, soft_logits)

    def regularization_loss(
        self,
        lambda_bin_mem: float = 0.001,
        lambda_bin_out: float = 0.001,
    ) -> torch.Tensor:
        """Binary-encouraging regularisation on memory patterns and values."""
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            reg = reg + lambda_bin_mem * binary_regularization(
                layer.memory.pattern_soft()
            )
            if not layer.is_output:
                reg = reg + lambda_bin_out * binary_regularization(
                    layer.memory.value_soft()
                )
        return reg

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Hard prediction (no gradient)."""
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=-1)
