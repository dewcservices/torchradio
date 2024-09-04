"""Common digital modulation algorithms."""

import numpy as np
import torch
from commpy import PSKModem, QAMModem  # type: ignore

from torchradio import Reception, Transmission
from torchradio.algorithm.utils import get_random_bits
from torchradio.types import COMPLEX_DTYPE


class Modem:
    """Torchradio compatible wrapper for commpy.PSKModem and commpy.QAMModem.

    Currently implemented using the third-party `commpy` library. Note that `Modem` does not contain
    any trainable parameters. A `Modem` provides compatible `tx` and `rx` methods. When defining
    transmitters and receivers, ensure the corresponding method is passed.

    Example:
        ```
        >>> modem = Modem("psk", 8)
        >>> env = NullEnvironment()
        >>> env.place({"tx": modem.tx}, {"rx": modem.rx})
        ```

    """

    def __init__(
        self,
        mode: str,
        n_symbols: int,
        demod_type: str = "hard",
    ) -> None:
        """Create a new `Modem` using either phase-shift keying (PSK) or quadrature amplitude modulation (QAM).

        Args:
            mode: Modulation mode. Must be either "psk" or "qam".
            n_symbols: Number of symbols to use for modulation. If `mode == "psk", `n_symbols`
                must be a power of 2. If `mode == "qam"`, `n_symbols` must be a power of 2 and
                square.
            demod_type: "hard" or "soft" decision boundary for bit demodulation.


        Raises:
            ValueError: If `mode not in ["psk", "qam"]`.
            ValueError: If `n_symbols` is not a power of 2 or square (square only required for `mode == "qam"`).
            ValueError: If `demod_type not in ["hard", "soft"]`.
            NotImplementedError: If `demod_type == "soft"`.

        """
        if mode not in ["psk", "qam"]:
            err = f'{mode=} not in ["psk", "qam"]'
            raise ValueError(err)

        if demod_type not in ["hard", "soft"]:
            err = f'{demod_type=} not in ["hard", "soft"]'
            raise ValueError(err)

        if demod_type == "soft":
            raise NotImplementedError

        self._modem = PSKModem(m=n_symbols) if mode == "psk" else QAMModem(m=n_symbols)
        self._demod_type = demod_type

    @property
    def n_input_bits(self) -> int:
        """Get number of bits per symbol.

        Returns
            Number of bits per symbol.

        """
        return int(np.log2(len(self._modem.constellation)))

    def tx(self, n_timesteps: int, batch_size: int = 1) -> Transmission:
        """Apply digital modulation to randomly generated bits.

        Args:
            n_timesteps: How many timesteps to transmit for.
            batch_size: How many transmissions to create in parallel.

        Returns:
            A `Transmission` with a complex-valued signal tensor with shape `[n_timesteps, batch_size]`
                and a `metadata` dictionary that records the transmitted bits using the key "bits".

        """
        bits = get_random_bits(n_timesteps * self.n_input_bits, batch_size)
        signal = self._modulate(bits)

        return Transmission(
            signal=signal,
            metadata={"bits": bits},
        )

    def _modulate(self, bits: torch.Tensor) -> torch.Tensor:
        signal = [self._modem.modulate(b) for b in bits]
        return torch.tensor(np.array(signal), dtype=COMPLEX_DTYPE)

    def rx(self, signal: torch.Tensor) -> Reception:
        """Demodulates an input signal using a prespecified digital demodulation technique.

        Args:
            signal: A 2D complex valued input tensor.

        Returns:
            A `Reception` dictionary that records the decoded bits using the key "bits".

        """
        return {"bits": self._demodulate(signal)}

    def _demodulate(self, signal: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            np.array(
                [
                    self._modem.demodulate(s, demod_type=self._demod_type)
                    for s in signal
                ],
            ),
        )


class DSSS(Modem):
    """Direct-sequence spread spectrum modulation and demodulation.

    Currently implemented using the third-party `commpy` library. Note that `DSSS` does not contain
    any trainable parameters. `DSSS` provides compatible `tx` and `rx` methods. When defining
    transmitters and receivers, ensure the corresponding method is passed.

    Example:
        ```
        >>> chip_sequence = torch.randint(0, 2, (4,))  # binary sequence of length 4
        >>> modem = DSSS(chip_sequence)
        >>> env = NullEnvironment()
        >>> env.place({"tx": modem.tx}, {"rx": modem.rx})
        ```

    """

    _threshold = 0.5

    def __init__(
        self,
        chip_sequence: torch.Tensor,
        mode: str = "psk",
        n_symbols: int = 2,
        demod_type: str = "hard",
    ) -> None:
        """Create a new `DSSS` radio using either phase-shift keying (PSK) or quadrature amplitude modulation (QAM).

        Args:
            chip_sequence: A binary sequence to spread the input bits with.
            mode: Modulation mode. Must be either "psk" or "qam".
            n_symbols: Number of symbols to use for modulation. If `mode == "psk", `n_symbols`
                must be a power of 2. If `mode == "qam"`, `n_symbols` must be a power of 2 and
                square.
            demod_type: "hard" or "soft" decision boundary for bit demodulation.


        Raises:
            ValueError: If `mode not in ["psk", "qam"]`.
            ValueError: If `n_symbols` is not a power of 2 or square (square only required for `mode == "qam"`).
            ValueError: If `demod_type not in ["hard", "soft"]`.
            NotImplementedError: If `demod_type == "soft"`.

        """
        super().__init__(mode, n_symbols, demod_type)
        self._chip_sequence = chip_sequence

    def tx(self, n_timesteps: int, batch_size: int = 1) -> Transmission:
        """Encode randomly generated bits with a chip sequence before modulating using commpy.Modem.

        Args:
            n_timesteps: How many timesteps to transmit for.
            batch_size: How many transmissions to create in parallel.

        Returns:
            A `Transmission` with a complex-valued signal tensor with shape `[n_timesteps, batch_size]`
                and a `metadata` dictionary that records the transmitted bits using the key "bits".

        """
        n_bits = n_timesteps * self.n_input_bits / len(self._chip_sequence)

        if not n_bits.is_integer():
            err = f"A chip sequence of length {len(self._chip_sequence)} for {n_timesteps} timesteps requires {n_bits} bits. Please reselect n_timesteps such that n_timesteps * self.n_input_bits / len(self._chip_sequence) is an integer."
            raise ValueError(err)

        bits = get_random_bits(int(n_bits), batch_size)

        repeated_bits = torch.repeat_interleave(bits, len(self._chip_sequence), dim=-1)
        tiled_chip_sequence = torch.tile(self._chip_sequence, [batch_size, int(n_bits)])
        dsss = torch.logical_xor(repeated_bits, tiled_chip_sequence)

        return Transmission(
            signal=self._modulate(dsss),
            metadata={"bits": bits},
        )

    def rx(self, signal: torch.Tensor) -> Reception:
        """Demodulates and despreads an input signal.

        Args:
            signal: A 2D complex valued input tensor.

        Returns:
            A `Reception` dictionary that records the decoded bits using the key "bits".

        """
        batch_size = signal.shape[0]
        dsss = self._demodulate(signal)

        return {
            "bits": (
                torch.logical_xor(
                    dsss.reshape(batch_size, -1, len(self._chip_sequence)),
                    self._chip_sequence,
                )
                .type(torch.float)
                .mean(axis=-1)
                > self._threshold
            )
            .type(torch.int)
            .reshape(batch_size, -1),
        }
