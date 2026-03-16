"""
Minchinton cell layer for VG-RAM networks.

Each synapse compares two positions of the input and outputs a single bit.
Forward is hard (indicator function); backward uses a sigmoid surrogate.
"""

import torch
import torch.nn as nn

from .functional import minchinton_compare


class MinchintonLayer(nn.Module):
    """Vectorised Minchinton-cell layer.

    For *num_neurons* neurons, each with *num_synapses* synapses, the layer
    picks two input positions per synapse and produces a binary vector.

    The synapse index pairs are drawn uniformly at random during construction
    and kept fixed (registered as buffers).

    Args:
        input_size:   dimensionality of the flat input vector.
        num_neurons:  number of neurons (N).
        num_synapses: number of synapses per neuron (P).
        tau:          initial temperature for the sigmoid relaxation.
    """

    def __init__(
        self,
        input_size: int,
        num_neurons: int,
        num_synapses: int,
        tau: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.num_synapses = num_synapses
        self.tau = tau

        idx_p = torch.randint(0, input_size, (num_neurons, num_synapses))
        idx_q = torch.randint(0, input_size, (num_neurons, num_synapses))
        self.register_buffer("idx_p", idx_p)
        self.register_buffer("idx_q", idx_q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_size) — continuous input values in [0, 1].

        Returns:
            bits: (batch, num_neurons, num_synapses) — binary via STE.
        """
        u = x[:, self.idx_p]   # (batch, N, P)
        v = x[:, self.idx_q]   # (batch, N, P)
        return minchinton_compare(u, v, self.tau)
