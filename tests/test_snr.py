# ruff: noqa: D103

import numpy as np
import pytest
import torch

from torchradio import Receiver, Transmitter
from torchradio.algorithm import Modem
from torchradio.env.null import ControlledSNREnvironment
from torchradio.position import get_null_distribution


@pytest.fixture
def n_timesteps() -> int:
    return 100


@pytest.fixture
def modem() -> Modem:
    return Modem("psk", 2)


@pytest.fixture
def transmitter(modem: Modem) -> Transmitter:
    return Transmitter(modem.tx, get_null_distribution())


@pytest.fixture
def receiver(modem: Modem) -> Receiver:
    return Receiver(modem.rx, get_null_distribution())


@pytest.fixture
def env() -> ControlledSNREnvironment:
    return ControlledSNREnvironment(0)


@pytest.mark.parametrize("snr", [-10.0, -5.0, 0.0, 5.0, 10.0])
def test_snr(
    snr: float,
    env: ControlledSNREnvironment,
    n_timesteps: int,
    transmitter: Transmitter,
    receiver: Receiver,
) -> None:
    """Check that ControlledSNREnvironment produces signals at the receiver with the correct SNR."""
    transmitter_name = "test_tx"
    receiver_name = "test_rx"
    transmitters = {transmitter_name: transmitter}
    receivers = {receiver_name: receiver}

    env.reset()
    env.set_snr(snr)
    env.place(transmitters, receivers)
    device_logs = env.simulate(n_timesteps)

    # check that transmission is a raw multiple of (received signal minus background noise)
    transmission = device_logs.tx[transmitter_name].signal
    background_noise = device_logs.rx[receiver_name]["noise"]
    propagated = device_logs.rx[receiver_name]["raw"] - background_noise
    ratios = torch.abs(transmission) / torch.abs(propagated)
    assert torch.allclose(ratios[0], ratios)

    # check SNR is correct
    signal_power = torch.var(propagated).numpy()
    noise_power = torch.var(background_noise).numpy()
    computed_snr = 10 * np.log10(signal_power / noise_power)
    assert np.isclose(snr, computed_snr, atol=1e-3)


def test_single_pair(
    env: ControlledSNREnvironment,
    transmitter: Transmitter,
    receiver: Receiver,
) -> None:
    """Check that ControlledSNREnvironment raises a ValueError if more than one transmitter or receiver is place."""
    transmitters = {"a": transmitter, "b": transmitter}
    receivers = {"a": receiver, "b": receiver}

    env.reset()
    with pytest.raises(
        ValueError,
        match="only supports the placement of a single transmitter",
    ):
        env.place(transmitters, receivers)
