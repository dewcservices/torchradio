"""Example trainable algorithms to get started with.

New users are recommended to use `DenseRadio` to ensure that input parameters are compatible.
A `DenseRadio` provides compatible `tx` and `rx` methods. When defining transmitters and receivers,
ensure the corresponding method is passed.

"""

import warnings

import torch
from torch import nn

from torchradio import Reception, Transmission
from torchradio.algorithm.utils import get_random_bits
from torchradio.types import COMPLEX_DTYPE


class DenseTransmissionAlgorithm(nn.Module):
    """An opinionated `TransmissionAlgorithm` that uses trainable dense layers.

    Do not expect this algorithm to be especially performant. It is an example to help
    users get started with training transmitters using `Torchradio`.
    """

    def __init__(self, n_input_bits: int, tx_length_per_bit: float) -> None:
        """Create a new `DenseTranmissionAlgorithm`.

        Args:
            n_input_bits: An input parameter used to shape the number of parameters in the
                dense layers.
            tx_length_per_bit: An input parameter used to shape the number of parameters in the
                dense layers.

        """
        super().__init__()

        self._input_length = n_input_bits
        self._output_length = int(tx_length_per_bit * n_input_bits)

        # don't worry new users about Lazy API warnings
        warnings.filterwarnings("ignore", module="torch")

        self._net = nn.Sequential(
            nn.LazyLinear(4 * self._output_length),
            nn.ReLU(),
            nn.LazyLinear(2 * self._output_length),
        )

    def __call__(self, n_timesteps: int, batch_size: int = 1) -> Transmission:
        """Apply a series of dense layers to randomly generated bits.

        Args:
            n_timesteps: How many timesteps to transmit for.
            batch_size: How many transmissions to create in parallel.

        Returns:
            A `Transmission` with a complex-valued signal tensor with shape `[n_timesteps, batch_size]`
                and a `metadata` dictionary that records the transmitted bits using the key "bits".

        """
        transmission = torch.zeros([batch_size, n_timesteps], dtype=COMPLEX_DTYPE)
        bits = get_random_bits(
            self._input_length * (n_timesteps // self._output_length),
            batch_size,
        )
        signal = self._net(bits.reshape(-1, self._input_length).float()).reshape(
            batch_size,
            -1,
            2,
        )
        transmission[:, : len(signal[0])] += signal[..., 0] + 1j * signal[..., 1]

        return Transmission(
            signal=transmission,
            metadata={"bits": bits},
        )


class DenseReceptionAlgorithm(nn.Module):
    """An opinionated `ReceptionAlgorithm` that uses trainable dense layers.

    Do not expect this algorithm to be especially performant. It is an example to help
    users get started with training receivers using `Torchradio`.
    """

    def __init__(self, n_bits: int, window_length: int) -> None:
        """Create a new `DenseReceptionAlgorithm`.

        Args:
            n_bits: An input parameter used to shape the number of parameters in the
                dense layers. Should be compatible with the `n_input_bits` specified in
                `DenseTransmissionAlgorithm`.
            window_length: An input parameter used cutoff unused bits from
                `DenseTransmissionAlgorithm`. Usually this is set to
                `n_input_bits * tx_length_per_bit` from the instantiation of
                `DenseTransmissionAlgorithm`.


        """
        super().__init__()

        self._n_bits = n_bits
        self._window_length = window_length

        # don't worry new users about Lazy API warnings
        warnings.filterwarnings("ignore", module="torch")

        self._net = nn.Sequential(
            nn.LazyLinear(4 * self._n_bits),
            nn.ReLU(),
            nn.LazyLinear(2 * self._n_bits),
            nn.ReLU(),
            nn.LazyLinear(self._n_bits),
            nn.Sigmoid(),
        )

    def __call__(self, signal: torch.Tensor) -> Reception:
        """Apply a series of dense layers an input signal.

        Args:
            signal: A 2D complex valued input tensor.


        Returns:
            A `Reception` dictionary that records the decoded bits using the key "bits",
                and the probability of each bit under "bit_probabilities".

        """
        batch_size = signal.shape[0]

        # trim unused timesteps
        to_trim = signal.shape[1] % self._window_length
        if to_trim != 0:
            signal = signal[:, :-to_trim]

        reshaped = (
            torch.stack([signal.real, signal.imag])
            .transpose(0, 1)
            .transpose(2, 1)
            .reshape(-1, 2 * self._window_length)
        )
        bit_probabilities = self._net(reshaped).reshape(batch_size, -1)

        decoded = torch.round(bit_probabilities).int()

        return {
            "bit_probabilities": bit_probabilities,
            "bits": decoded,
        }


class DenseRadio(nn.Module):
    """A convenient wrapper for `DenseTransmissionAlgorithm` and `DenseReceptionAlgorithm`.

    `DenseRadio` guarantees that the underlying `TransmissionAlgorithm` and `ReceptionAlgorithm` are
    compatible. A `DenseRadio` provides compatible `tx` and `rx` methods. When defining transmitters
    and receivers, ensure the corresponding method is passed.

    Example:
        ```
        >>> radio = DenseRadio(8, 2)
        >>> env = NullEnvironment()
        >>> env.place({"tx": radio.tx}, {"rx": radio.rx})
        ```

    """

    def __init__(self, n_input_bits: int, tx_length_per_bit: float) -> None:
        """Create a new `DenseRadio`. See DenseTransmissionAlgorithm.__init__."""
        super().__init__()

        self._tx = DenseTransmissionAlgorithm(n_input_bits, tx_length_per_bit)
        self._rx = DenseReceptionAlgorithm(
            n_input_bits,
            int(n_input_bits * tx_length_per_bit),
        )

    def tx(self, n_timesteps: int, batch_size: int = 1) -> Transmission:
        """See `DenseTransmissionAlgorithm.__call__`."""
        return self._tx(n_timesteps, batch_size)

    def rx(self, signal: torch.Tensor) -> Reception:
        """See `DenseReceptionAlgorithm.__call__`."""
        return self._rx(signal)
