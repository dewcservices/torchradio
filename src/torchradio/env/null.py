"""Abstract environments that do not require physical device placements."""

import torch

from torchradio.device import Position
from torchradio.env.base import BaseEnvironment


class NullEnvironment(BaseEnvironment):
    """The simplest possible environment with ideal conditions."""

    def _compute_propagation_parameters(self) -> None:
        """No propagation parameters to compute."""

    def _in_bounds(self, position: Position) -> bool:  # noqa: ARG002
        """Everything is in bounds."""
        return True

    def _propagate(self, tx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """No attenuation or reflection. Simply sum all transmissions together."""
        summed = torch.sum(torch.stack(list(tx.values()), axis=1), axis=1)  # type: ignore
        return {receiver_name: summed for receiver_name in self._receivers}

    def _get_background_noise(
        self,
        n_timesteps: int,
        batch_size: int = 1,
    ) -> dict[str, torch.Tensor]:
        """No background noise."""
        return {
            receiver_name: torch.zeros([batch_size, n_timesteps], dtype=torch.complex64)
            for receiver_name in self._receivers
        }


class ControlledSNREnvironment(NullEnvironment):
    """A single-channel environment that applies additive white Gaussian noise (AWGN) to a specific signal-to-noise ratio (SNR).

    A null environment with controllable SNR that supports a single transmitter and receiver pair.
    During propagation, the transmitted signal is reduced to unit power. The correct amount of AWGN is
    then generated to achieve the specified SNR.

    Raises
        ValueError: If more than one transmitter or more than one receiver.

    """

    def __init__(self, snr: float) -> None:
        """Create a `ControlledSNREnvironment` for single channel simulations with Gaussian noise.

        Args:
            snr: Desired signal-to-noise ratio in dB.

        """
        self._snr = torch.tensor([snr])

    @property
    def snr(self) -> float:
        """Get the currently set signal-to-noise ratio in dB.

        Returns
            Current signal-to-noise ratio in dB.

        """
        return self._snr.numpy()[0]

    def set_snr(self, snr: float) -> None:
        """Set signal-to-noise ratio (SNR) for the receiver in dB.

        Args:
            snr: Desired signal-to-noise ratio in dB.

        """
        self._snr = torch.tensor([snr])

    def _compute_propagation_parameters(self) -> None:
        """No propagation parameters to compute - check that only one transmitter and one receiver have been placed."""
        if len(self.transmitters) != 1:
            err = f"{self.__class__.__name__} only supports the placement of a single transmitter. {len(self.transmitters)=} != 1"
            raise ValueError(err)

        if len(self.receivers) != 1:
            err = f"{self.__class__.__name__} only supports the placement of a single receiver. {len(self.receivers)=} != 1"
            raise ValueError(err)

    def _propagate(self, tx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Attenuate signal to low power such that, when it is summed with unit power background noise, the desired SNR is achieved."""
        summed = torch.sum(torch.stack(list(tx.values()), axis=1), axis=1)  # type: ignore
        norm_tx = summed / torch.sqrt(torch.var(summed))
        tx_ = torch.sqrt(10 ** (self._snr / 10)) * norm_tx
        return {receiver_name: tx_ for receiver_name in self._receivers}

    def _get_background_noise(
        self,
        n_timesteps: int,
        batch_size: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Get background noise with unit power."""
        noise = torch.randn(batch_size, n_timesteps) + 1j * torch.randn(
            batch_size,
            n_timesteps,
        )
        norm_noise = noise / torch.sqrt(torch.var(noise))
        return {receiver_name: norm_noise for receiver_name in self._receivers}


class RandomBoundedSNREnvironment(ControlledSNREnvironment):
    """A single-channel environment that applies a random amount of additive white Gaussian noise (AWGN) according to preset SNR bounds.

    A null environment with bounded SNR that supports a single transmitter and receiver pair.
    During propagation, the transmitted signal is reduced to unit power. The correct amount of AWGN is
    then generated to achieve an SNR within the provided bounds.

    Raises
        ValueError: If more than one transmitter or more than one receiver.

    """

    def __init__(self, snr_min: float, snr_max: float) -> None:
        """Create a `RandomBoundedSNREnvironment` for single channel simulations with Gaussian noise.

        Args:
            snr_min: Minimum desired signal-to-noise ratio in dB.
            snr_max: Maximum desired signal-to-noise ratio in dB.

        Raise:
            ValueError: if `snr_max < snr_min`.

        """
        if snr_max < snr_min:
            err = f"{snr_max=}<{snr_min=}."
            raise ValueError(err)

        self._snr_min = snr_min
        self._snr_diff = snr_max - snr_min

    def _propagate(self, tx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Attenuate signal to low power such that, when it is summed with unit power background noise, the desired SNR is achieved."""
        snr = self._snr_min + torch.rand(1) * self._snr_diff
        summed = torch.sum(torch.stack(list(tx.values()), axis=1), axis=1)  # type: ignore
        norm_tx = summed / torch.sqrt(torch.var(summed))
        tx_ = torch.sqrt(10 ** (snr / 10)) * norm_tx
        return {receiver_name: tx_ for receiver_name in self._receivers}


class RandomAWGNEnvironment(NullEnvironment):
    """A null environment that produces a random level of background noise for each receiver.

    This random level is sampled per receiver according to the power bounds supplied at instantiation.
    """

    def __init__(
        self,
        p_min: float,
        p_max: float,
        *,
        normalize_before_noise: bool = False,
    ) -> None:
        """Initialize a new `RandomAWGNEnviroment` with bounded background noise levels.

        Args:
            p_min: Minimum background noise power.
            p_max: Maximum background noise power.
            normalize_before_noise: Set to True if the aggregated signals should be normalized
                to unit power before adding background noise. Defaults to False.

        """
        self.set_bounds(p_min, p_max)
        self._normalize = normalize_before_noise

    @property
    def bounds(self) -> tuple[float, float]:
        """Get current noise bounds.

        Returns
            Background noise bounds as a tuple.

        """
        return self._p_min, self._p_max

    def set_bounds(self, p_min: float, p_max: float) -> None:
        """Set noise bounds for all receivers.

        Args:
            p_min: Minimum background noise power.
            p_max: Maximum background noise power.

        """
        if p_max < p_min:
            err = f"{p_max=} < {p_min=}"
            raise ValueError(err)

        if p_min < 0:
            err = f"{p_min=} < 0"

        self._p_min = p_min
        self._p_max = p_max

    def _propagate(self, tx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Randomly attenuate each transmission and sum.

        The amount of random attenutation is different for each transmitter/receiver pair.
        If self.normalize == True, the received signals are normalized to unit power.
        """
        # copy each tx for each rx
        concatenated = torch.concat([tx_[None] for tx_ in tx.values()])
        duplicated = torch.tile(
            concatenated,
            [self.n_receivers, 1, 1, 1],
        )  # [rx, tx, batch, t]

        # take weighted sum
        weights = torch.rand(self.n_receivers, len(tx), duplicated.shape[2])
        weights /= weights.sum(axis=1, keepdim=True)  # average over transmitters
        weighted = weights[..., None] * duplicated
        result = weighted.sum(axis=1)  # [rx, batch, t]

        if self._normalize:
            powers = torch.var(result, axis=-1)
            result = result / torch.sqrt(powers)

        return dict(zip(self._receivers, result, strict=True))

    def _get_background_noise(
        self,
        n_timesteps: int,
        batch_size: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Get background noise with random power level for each batch/receiver sampled in self.bounds."""
        # Get background noise with unit power
        noise = torch.randn(
            self.n_receivers,
            batch_size,
            n_timesteps,
        ) + 1j * torch.randn(
            self.n_receivers,
            batch_size,
            n_timesteps,
        )
        norm_noise = noise / torch.sqrt(torch.var(noise, axis=-1))[..., None]

        # Apply random power level for each batch/receiver
        noise_levels = self._p_min + (self._p_max - self._p_min) * torch.rand(
            self.n_receivers,
            batch_size,
        )
        adjusted_noise = torch.sqrt(noise_levels[..., None]) * norm_noise

        return dict(zip(self._receivers, adjusted_noise, strict=True))
