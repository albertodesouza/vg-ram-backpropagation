"""
Temperature annealing scheduler for VG-RAM networks.

During training the temperatures are annealed so that the soft backward
starts smooth and gradually approaches the hard discrete behaviour:
  - tau_b (Minchinton sigmoid)  : decrease over epochs
  - tau_a (memory-pattern sigmoid): decrease over epochs
  - beta  (softmin inverse temp)  : increase over epochs
"""

from __future__ import annotations

import math

from .network import VGRAMNetwork


class TemperatureScheduler:
    """Exponential annealing of tau_b, tau_a, and beta across all layers.

    At epoch *e* (0-based) out of *total_epochs*:

        tau(e) = tau_start * (tau_end / tau_start) ^ (e / (total_epochs - 1))
        beta(e) = beta_start * (beta_end / beta_start) ^ (e / (total_epochs - 1))

    Args:
        network:      the :class:`VGRAMNetwork` whose layers will be updated.
        total_epochs: total number of training epochs.
        tau_b_start, tau_b_end: range for Minchinton temperature.
        tau_a_start, tau_a_end: range for memory-pattern temperature.
        beta_start, beta_end:   range for softmin inverse temperature.
    """

    def __init__(
        self,
        network: VGRAMNetwork,
        total_epochs: int,
        tau_b_start: float = 1.0,
        tau_b_end: float = 0.1,
        tau_a_start: float = 1.0,
        tau_a_end: float = 0.1,
        beta_start: float = 1.0,
        beta_end: float = 10.0,
    ):
        self.network = network
        self.total_epochs = max(total_epochs, 1)
        self.tau_b_start = tau_b_start
        self.tau_b_end = tau_b_end
        self.tau_a_start = tau_a_start
        self.tau_a_end = tau_a_end
        self.beta_start = beta_start
        self.beta_end = beta_end

    def _interp(self, start: float, end: float, epoch: int) -> float:
        if self.total_epochs <= 1:
            return end
        ratio = epoch / (self.total_epochs - 1)
        return start * (end / start) ** ratio

    def step(self, epoch: int) -> dict[str, float]:
        """Update all layer temperatures for the given epoch.

        Returns:
            dict with current values of tau_b, tau_a, beta (for logging).
        """
        tau_b = self._interp(self.tau_b_start, self.tau_b_end, epoch)
        tau_a = self._interp(self.tau_a_start, self.tau_a_end, epoch)
        beta = self._interp(self.beta_start, self.beta_end, epoch)

        for layer in self.network.layers:
            layer.minchinton.tau = tau_b
            layer.memory.tau_a = tau_a
            layer.memory.beta = beta

        return {"tau_b": tau_b, "tau_a": tau_a, "beta": beta}
