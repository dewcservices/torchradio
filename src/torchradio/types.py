"""Defines the core types used throughout Torchradio."""

from dataclasses import dataclass
from typing import Protocol

import torch

COMPLEX_DTYPE = torch.complex64
""" The common datatype for all complex tensors in Torchradio."""


@dataclass
class Transmission:
    """The core output type for any transmitter. A `Transmission` consists of two parts: a signal and a metadata dictionary.

    Args:
        signal: A complex-valued tensor that is propagated into the environment
        metadata: Used to record arbitrary information that can be used by downstream loss functions.
            The most commonly used metadata key is 'bits' to capture the bits before the `TransmissionAlgortihm`
            produces an output signal.

    """

    signal: torch.Tensor
    metadata: dict[str, torch.Tensor]


Reception = dict[str, torch.Tensor]
""" The `ReceptionAlgorithm` author is free to capture whatever fields are required for downstream loss functions. A `Reception` is
the receiver analogue of `Transmission.metadata`."""


@dataclass
class DeviceLogs:
    """Captures the `Transmission` and `Reception` logs from a simulation run.

    Each `Transmission` and `Reception` is associated with a device name.

    Args:
        tx: Maps device names to `Transmission`s.
        rx: Maps device names to `Reception`s.

    """

    tx: dict[str, Transmission]
    rx: dict[str, Reception]


class TransmissionAlgorithm(Protocol):
    """All transmitters in Torchradio are required to implement a `TransmissionAlgorithm`.

    A `TransmissionAlgorithm` is any `Callable` that produces a `Transmission` with a complex-valued
    `signal` with shape `[batch_size, n_timesteps]`.
    """

    def __call__(self, n_timesteps: int, batch_size: int) -> Transmission:
        """Return a `Transmission` given `n_timesteps` and `batch_size`.

        Args:
            n_timesteps: Number of timesteps to simulate.
            batch_size: How many batches to yield.

        Returns:
            A `Transmission` with a complex-valued signal with shape `[batch_size, n_timesteps]`.

        """
        ...


class ReceptionAlgorithm(Protocol):
    """All receivers in Torchradio are required to implement a `ReceptionAlgorithm`.

    A `ReceptionAlgorithm` is any `Callable` that produces a `Reception` from an input `signal`.

    """

    def __call__(self, signal: torch.Tensor) -> Reception:
        """Return a `Reception` from a complex-valued input `signal`.

        Args:
            signal: A complex-valued input tensor from the environment.

        Returns:
            A `Reception` with that maps each key to a `torch.Tensor` for downstream loss functions.

        """
        ...
