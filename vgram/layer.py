"""
VG-RAM layer: Minchinton cells followed by memory lookup.

A layer contains N neurons.  Each neuron applies P Minchinton-cell comparisons
to the input, then looks up the closest stored pattern in its memory and
returns the associated value.
"""

import torch
import torch.nn as nn

from .functional import bits_to_scalar, ste
from .memory import VGRAMMemory
from .minchinton import MinchintonLayer


class VGRAMLayer(nn.Module):
    """Single VG-RAM layer (Minchinton cells + memory bank).

    Args:
        input_size:   flat input dimensionality.
        num_neurons:  N neurons in this layer.
        num_synapses: P synapses (Minchinton cells) per neuron.
        num_entries:  M memory entries per neuron.
        output_dim:   D — dimension of each neuron's output
                      (8 for byte, 10 for class logits).
        tau_b:        Minchinton sigmoid temperature.
        tau_a:        memory-pattern sigmoid temperature.
        beta:         softmin inverse temperature.
        is_output:    if True, output is kept as (batch, N, D) logits;
                      if False, the D-bit output is converted to a scalar
                      in [0, 1] so it can feed the next layer.
    """

    def __init__(
        self,
        input_size: int,
        num_neurons: int,
        num_synapses: int,
        num_entries: int,
        output_dim: int,
        tau_b: float = 1.0,
        tau_a: float = 1.0,
        beta: float = 1.0,
        is_output: bool = False,
        neuron_chunk_size: int = 0,
        use_grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.is_output = is_output
        self.output_dim = output_dim

        self.minchinton = MinchintonLayer(
            input_size=input_size,
            num_neurons=num_neurons,
            num_synapses=num_synapses,
            tau=tau_b,
        )
        self.memory = VGRAMMemory(
            num_neurons=num_neurons,
            num_entries=num_entries,
            num_synapses=num_synapses,
            output_dim=output_dim,
            tau_a=tau_a,
            beta=beta,
            neuron_chunk_size=neuron_chunk_size,
            use_grad_checkpoint=use_grad_checkpoint,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_size)

        Returns:
            If is_output: (batch, N, D) — raw neuron outputs (e.g. class logits).
            Else:         (batch, N)    — scalar per neuron in [0, 1].
        """
        bits = self.minchinton(x)        # (batch, N, P)
        out = self.memory(bits)          # (batch, N, D)

        if self.is_output:
            return out

        hard_scalar = bits_to_scalar(out.detach())
        soft_scalar = bits_to_scalar(out)
        return ste(hard_scalar, soft_scalar)  # (batch, N)
