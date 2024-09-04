# ruff: noqa: D103

import time

import pytest
import torch

from torchradio import (
    Receiver,
    Reception,
    ReceptionAlgorithm,
    Transmission,
    TransmissionAlgorithm,
    Transmitter,
)
from torchradio.env import NullEnvironment
from torchradio.position import get_null_distribution


@pytest.fixture
def n_timesteps() -> int:
    return 100


@pytest.fixture
def transmission_algorithms() -> tuple[TransmissionAlgorithm, TransmissionAlgorithm]:
    def transmit1(n_timesteps: int, batch_size: int = 1) -> Transmission:
        return Transmission(
            signal=(1 + 1j)
            * torch.ones(batch_size, n_timesteps, dtype=torch.complex64),
            metadata={"time": torch.tensor(time.time())},
        )

    def transmit2(n_timesteps: int, batch_size: int = 1) -> Transmission:
        vector = (-1 - 1j) * torch.ones(batch_size, n_timesteps, dtype=torch.complex64)
        vector[
            :,
            torch.arange(n_timesteps) % 2 == 0,
        ] = 0  # map every second element to zero
        return Transmission(signal=vector, metadata={"time": torch.tensor(time.time())})

    return transmit1, transmit2


@pytest.fixture
def reception_algorithms() -> tuple[ReceptionAlgorithm, ReceptionAlgorithm]:
    def receive1(signal: torch.Tensor) -> Reception:
        # returns the sum of the inputs
        return {"sum": torch.sum(signal)}

    def receive2(signal: torch.Tensor) -> Reception:
        # returns the max and the argmax of the inputs
        mag = torch.abs(signal)
        return {"max": torch.max(mag), "argmax": torch.argmax(mag)}

    return receive1, receive2


def test_simple(
    n_timesteps: int,
    transmission_algorithms: tuple[TransmissionAlgorithm, TransmissionAlgorithm],
    reception_algorithms: tuple[ReceptionAlgorithm, ReceptionAlgorithm],
) -> None:
    transmit1, transmit2 = transmission_algorithms
    receive1, receive2 = reception_algorithms

    env = NullEnvironment()

    transmitters = {
        "tx1": Transmitter(transmit1, get_null_distribution()),
        "tx2": Transmitter(transmit2, get_null_distribution(x=10, y=10, z=10), 0.5),
    }

    receivers = {
        "rx1": Receiver(receive1, get_null_distribution(x=100)),
        "rx2": Receiver(receive2, get_null_distribution(x=50, y=80, z=10)),
    }

    env.reset()
    env.place(transmitters, receivers)
    env.simulate(n_timesteps)
