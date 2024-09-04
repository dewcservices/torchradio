"""A device pairs a `TransmissionAlgorithm` or a `ReceptionAlgorithm` with other physical characteristics."""

from collections.abc import Iterator

import torch

from torchradio.position import Position, SpatialDistribution, get_null_distribution
from torchradio.types import (
    Reception,
    ReceptionAlgorithm,
    Transmission,
    TransmissionAlgorithm,
)


class Device:
    """The base `Device` class consists of a `SpatialDistribution` that is sampled whenever the device is placed in an environment."""

    def __init__(
        self,
        spatial_distribution: SpatialDistribution | None = None,
    ) -> None:
        """Create a new device with a `spatial_distribution`.

        Args:
            spatial_distribution: A `Callable` that randomly samples `Position`s.
                Defaults to None. If None, a null `SpatialDistribution` is used,
                which always returns `Position(x=0, y=0, z=0)`.

        """
        if spatial_distribution is None:
            spatial_distribution = get_null_distribution()

        self._spatial_distribution = spatial_distribution

    @property
    def spatial_distribution(self) -> SpatialDistribution:
        """Get the device's spatial distribution.

        Returns
            The device's spatial distribution.

        """
        return self._spatial_distribution

    def place(self) -> Position:
        """Sample a `Position` from the device's spatial distribution.

        Returns
            A `Position` randomly sampled according to the device's spatial distribution.

        """
        return self._spatial_distribution()

    @staticmethod
    def _get_parameters(
        algorithm: TransmissionAlgorithm | ReceptionAlgorithm,
    ) -> Iterator[torch.nn.Parameter]:
        """Get the trainable parameters for an algortihm.

        Returns
            The algorithms's trainable parameters. An empty iterator is returned if the
                algortihm does not have trainable parameters.

        """
        if isinstance(algorithm, torch.nn.Module):
            return algorithm.parameters()
        return iter(())


class Transmitter(Device):
    """A `Device` that transmits signals into the environment."""

    def __init__(
        self,
        algorithm: TransmissionAlgorithm,
        spatial_distribution: SpatialDistribution | None = None,
        max_gain: float | None = None,
    ) -> None:
        """Create a new `Transmitter` from an algorithm and a spatial distribution.

        Args:
            algorithm: A `Callable` that produces `Transmission`s.
            spatial_distribution: A `Callable` that randomly samples `Position`s.
                Defaults to None. If None, a null `SpatialDistribution` is used,
                which always returns `Position(x=0, y=0, z=0)`.
            max_gain: A saturation limit on transmissions. The real and imaginary
                components of transmitted signals are capped to `max_gain`.

        """
        super().__init__(spatial_distribution)
        self._algorithm = algorithm
        self._max_gain = max_gain

    def __call__(self, n_timesteps: int, batch_size: int = 1) -> Transmission:
        """Invoke the underlying `TransmissionAlgorithm` to produce a `Tranmission`.

        Args:
            n_timesteps: Number of timesteps to transmit for.
            batch_size: How many simulations to conduct in parallel.

        Returns:
            A `Transmission` with a complex-valued signal with shape `[batch_size, n_timesteps]`.

        """
        if n_timesteps < 1:
            err = f"{n_timesteps=} < 1"
            raise ValueError(err)

        tx = self._algorithm(n_timesteps, batch_size)

        if "raw" in tx.metadata:
            err = f'{self._algorithm} uses the reserved keyword "raw" in the output metadata dictionary.'
            raise RuntimeError(err)

        tx.metadata["_raw"] = tx.signal  # track desired output signal under "raw"

        # cap any real or imaginary values to self._max_gain
        if (
            self._max_gain is not None
            and torch.max(torch.abs(tx.signal)) > self._max_gain
        ):
            real = tx.signal.real
            imag = tx.signal.imag
            real[real > self._max_gain] = self._max_gain
            imag[imag > self._max_gain] = self._max_gain
            tx.signal = torch.complex(real, imag)

        return tx

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Get the transmitter's trainable parameters.

        Returns
            The transmitter's trainable parameters. An empty iterator is returned if the
                transmitter does not have trainable parameters.

        """
        return self._get_parameters(self._algorithm)


class Receiver(Device):
    """A `Device` that receives signals from the environment."""

    def __init__(
        self,
        algorithm: ReceptionAlgorithm,
        spatial_distribution: SpatialDistribution | None = None,
    ) -> None:
        """Create a new `Receiver` from an algorithm and a spatial distribution.

        Args:
            algorithm: A `Callable` that produces `Receptions`s.
            spatial_distribution: A `Callable` that randomly samples `Position`s.
                Defaults to None. If None, a null `SpatialDistribution` is used,
                which always returns `Position(x=0, y=0, z=0)`.

        """
        super().__init__(spatial_distribution)
        self._algorithm = algorithm

    def __call__(self, signal: torch.Tensor) -> Reception:
        """Invoke the underlying `ReceptionAlgorithm` to produce a `Reception` dictionary.

        Args:
            signal: Input signal from the environment.

        Returns:
            A dictionary the summarizes various aspects of the received signal. Common
                keys for reconstructive receivers include 'bit_probabilities' and 'bits'.

        """
        return self._algorithm(signal)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Get the receiver's trainable parameters.

        Returns
            The receiver's trainable parameters. An empty iterator is returned if the
                receiever does not have trainable parameters.

        """
        return self._get_parameters(self._algorithm)
