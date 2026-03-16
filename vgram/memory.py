"""
VG-RAM memory bank.

Each neuron stores M entries of (pattern, value).  During the forward pass the
memory is queried with hard Hamming distance and argmin selection.  During the
backward pass, gradients flow through the expected-Hamming / softmin surrogate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .functional import (
    expected_hamming,
    hard_hamming,
    hard_memory_select,
    soft_memory_select,
    ste,
)


class VGRAMMemory(nn.Module):
    """Differentiable VG-RAM memory for one layer.

    Args:
        num_neurons:       N — number of neurons sharing this memory bank.
        num_entries:       M — memory slots per neuron.
        num_synapses:      P — bits per stored pattern.
        output_dim:        D — dimensionality of stored values
                           (8 for byte outputs, 10 for class logits, etc.).
        tau_a:             temperature for the stored-pattern sigmoid relaxation.
        beta:              inverse temperature for the softmin memory selection.
        neuron_chunk_size: process neurons in chunks of this size to limit peak
                           GPU memory.  0 or None means no chunking.
        use_grad_checkpoint: if True, use gradient checkpointing to trade
                           compute for memory (recomputes forward during backward).
    """

    def __init__(
        self,
        num_neurons: int,
        num_entries: int,
        num_synapses: int,
        output_dim: int,
        tau_a: float = 1.0,
        beta: float = 1.0,
        neuron_chunk_size: int = 0,
        use_grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.num_entries = num_entries
        self.num_synapses = num_synapses
        self.output_dim = output_dim
        self.tau_a = tau_a
        self.beta = beta
        self.neuron_chunk_size = neuron_chunk_size or num_neurons
        self.use_grad_checkpoint = use_grad_checkpoint

        self.pattern_logits = nn.Parameter(
            torch.randn(num_neurons, num_entries, num_synapses) * 0.1
        )
        self.value_logits = nn.Parameter(
            torch.randn(num_neurons, num_entries, output_dim) * 0.1
        )

    def _forward_chunk(
        self,
        bits_chunk: torch.Tensor,
        pattern_logits_chunk: torch.Tensor,
        value_logits_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Process a chunk of neurons.

        Args:
            bits_chunk:           (batch, C, P)
            pattern_logits_chunk: (C, M, P)
            value_logits_chunk:   (C, M, D)

        Returns:
            (batch, C, D)
        """
        a_hard = (pattern_logits_chunk > 0).float()
        a_soft = torch.sigmoid(pattern_logits_chunk / self.tau_a)
        a = ste(a_hard, a_soft)                                # (C, M, P)

        v_hard = (value_logits_chunk > 0).float()
        v_soft = torch.sigmoid(value_logits_chunk)
        v = ste(v_hard, v_soft)                                # (C, M, D)

        b_exp = bits_chunk.unsqueeze(2)                        # (B, C, 1, P)
        a_exp = a.unsqueeze(0)                                 # (1, C, M, P)

        d_hard = hard_hamming(b_exp, a_exp.detach())           # (B, C, M)
        d_soft = expected_hamming(b_exp, a_exp)                # (B, C, M)

        B = bits_chunk.shape[0]
        v_exp = v.unsqueeze(0)                                 # (1, C, M, D)
        out_hard = hard_memory_select(d_hard, v_exp.detach().expand(B, -1, -1, -1))
        out_soft = soft_memory_select(d_soft, v_exp.expand(B, -1, -1, -1), self.beta)
        return ste(out_hard, out_soft)                         # (B, C, D)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """Query the memory with input bit vectors.

        Args:
            bits: (batch, N, P) — binary input from Minchinton cells (via STE,
                  so the tensor carries soft gradients even though its forward
                  values are in {0, 1}).

        Returns:
            output: (batch, N, D) — selected memory values (hard on forward,
                    soft gradients on backward).
        """
        N = self.num_neurons
        cs = self.neuron_chunk_size

        if cs >= N:
            return self._forward_chunk(
                bits, self.pattern_logits, self.value_logits,
            )

        chunks = []
        for start in range(0, N, cs):
            end = min(start + cs, N)
            b_c = bits[:, start:end, :]
            p_c = self.pattern_logits[start:end]
            v_c = self.value_logits[start:end]

            if self.use_grad_checkpoint and self.training:
                out_c = grad_checkpoint(
                    self._forward_chunk, b_c, p_c, v_c,
                    use_reentrant=False,
                )
            else:
                out_c = self._forward_chunk(b_c, p_c, v_c)
            chunks.append(out_c)

        return torch.cat(chunks, dim=1)                        # (B, N, D)

    def pattern_soft(self) -> torch.Tensor:
        """Return the soft (sigmoid) version of stored patterns for regularisation."""
        return torch.sigmoid(self.pattern_logits / self.tau_a)

    def value_soft(self) -> torch.Tensor:
        """Return the soft (sigmoid) version of stored values for regularisation."""
        return torch.sigmoid(self.value_logits)
