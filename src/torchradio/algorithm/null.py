"""Simple (null) transmitters for testing and debugging."""

import torch

from torchradio import (
    Reception,
    ReceptionAlgorithm,
    Transmission,
    TransmissionAlgorithm,
)
from torchradio.types import COMPLEX_DTYPE


def get_constant_transmission_algorithm(
    val: complex | torch.nn.Parameter,
) -> TransmissionAlgorithm:
    """Get a basic transmission algorithm that output a constant complex value.

    Args:
        val: constant complex value to transmit

    Returns:
        A `TransmissionAlgorithm` that outputs a tensor where every element is `val`.

    Example:
        ```
        >>> tx = get_constant_transmitter(1 + 1j)
        >>> tx(5, 2)
        Transmission(signal=tensor([[1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j],
                [1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j]]), metadata={})
        ```

    """

    def transmission_algorithm(n_timesteps: int, batch_size: int = 1) -> Transmission:
        vector = val * torch.ones(batch_size, n_timesteps, dtype=COMPLEX_DTYPE)
        return Transmission(signal=vector, metadata={})

    return transmission_algorithm


def get_null_transmission_algortihm() -> TransmissionAlgorithm:
    """Get a null transmission algorithm that output nothing.

    Returns:
        A `TransmissionAlgorithm` that outputs a complex-valued tensors of zeros.

    Example:
        ```
        >>> tx = get_null_transmission_algortihm()
        >>> tx(5, 2)
        Transmission(signal=tensor([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]]), metadata={})
        ```

    """
    return get_constant_transmission_algorithm(0 + 0j)


def get_null_reception_algorithm() -> ReceptionAlgorithm:
    """Get a null reception algorithm that does not do any processing,.

    Returns:
        A null `ReceptionAlgorithm` that outputs an empty `Reception` dictionary.

    Example:
        ```
        >>> rx = get_null_reception_algorithm()
        >>> signal = torch.zeros([2, 5], dtype=torch.complex64)
        >>> rx(signal)
        {}
        ```

    """

    def reception_algorithm(signal: torch.Tensor) -> Reception:  # noqa: ARG001
        return {}

    return reception_algorithm
