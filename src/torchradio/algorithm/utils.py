"""Common helper functions for working with algorithms."""

from collections.abc import Generator, Iterable

import torch

from torchradio import Receiver, Transmitter


def get_random_bits(n_bits: int, batch_size: int = 1) -> torch.Tensor:
    """Output a random packet of bits.

    Args:
        n_bits: How many bits to output per batch.
        batch_size: How many batches to create.

    Returns:
        An 0-1 integer valued tensor with shape `[batch_size, n_bits]`.

    """
    return torch.randint(0, 2, size=[batch_size, n_bits])


def get_all_parameters(
    transmitters: Iterable[Transmitter],
    receivers: Iterable[Receiver],
) -> Generator[torch.nn.Parameter, None, None]:
    """Get the parameters for multiple devices to enable joint optimization.

    The parameters must be defined as a single iterator to be used with
    `torch.optim`.


    Args:
        transmitters: Transmitters to get parameters from.
        receivers: Receivers to get parameters from.

    Returns:
        Parameters for all devices as a single iterator.

    """
    for transmitter in transmitters:
        for p in transmitter.parameters():
            yield p

    for receiver in receivers:
        for p in receiver.parameters():
            yield p
