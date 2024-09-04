import pytest
import torch

from torchradio import Receiver, Transmitter
from torchradio.algorithm.null import (
    get_constant_transmission_algorithm,
    get_null_reception_algorithm,
)
from torchradio.env.base import BaseEnvironment
from torchradio.env.box import (
    BoxEnvironment,
    PlanarEnvironment,
)
from torchradio.env.null import (
    ControlledSNREnvironment,
    NullEnvironment,
    RandomAWGNEnvironment,
)


class SimpleNonDifferentiableEnvironment(NullEnvironment):
    """An example non-differentiable environment to test that `.is_differentiable()` returns False."""

    def _propagate(self, tx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # The same `_propagate` method as NullEnvironment, but the received tensors
        # are detached to break differentiability.
        summed = torch.sum(torch.concat(list(tx.values())), axis=0)  # type: ignore
        return {receiver_name: summed.detach() for receiver_name in self._receivers}


class NewParamNonDifferentiableEnvironment(NullEnvironment):
    """An example non-differentiable environment to test that `.is_differentiable()` returns False."""

    def _propagate(self, tx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # The same `_propagate` method as NullEnvironment, but the received tensors
        # are detached and added to a new parameter to break differentiability.
        summed = torch.sum(torch.concat(list(tx.values())), axis=0)  # type: ignore
        new_param = torch.nn.Parameter(torch.tensor(0 + 0j))
        return {
            receiver_name: new_param + summed.detach()
            for receiver_name in self._receivers
        }


@pytest.fixture
def null_environment() -> NullEnvironment:  # noqa: D103
    return NullEnvironment()


@pytest.mark.parametrize(
    "env",
    [
        NullEnvironment(),
        RandomAWGNEnvironment(0.1, 0.1),
        ControlledSNREnvironment(10),
        BoxEnvironment(10, 10, 10),
        PlanarEnvironment(10, 10),
    ],
)
def test_differentiability(env: BaseEnvironment) -> None:
    """Check that all Torchradio environments are differentiable."""
    assert env.is_differentiable()


@pytest.mark.parametrize(
    "env_cls",
    [
        SimpleNonDifferentiableEnvironment,
        NewParamNonDifferentiableEnvironment,
    ],
)
def test_non_differentiability(env_cls: type[BaseEnvironment]) -> None:
    """Check test non-differentiable environments are actually non-differentiable.

    Assert that a non-differentiability warning is raised upon initialization.

    """
    with pytest.warns(UserWarning):
        env = env_cls()
    assert not env.is_differentiable()


@pytest.mark.parametrize(
    ("transmitters", "expected_value"),
    [
        ({"tx": Transmitter(get_constant_transmission_algorithm(1 + 1j))}, 1 + 1j),
        (
            {
                "tx1": Transmitter(get_constant_transmission_algorithm(1 + 1j)),
                "tx2": Transmitter(get_constant_transmission_algorithm(2 + 2j)),
            },
            3 + 3j,
        ),
    ],
)
def test_null_correct(
    null_environment: NullEnvironment,
    transmitters: dict[str, Transmitter],
    expected_value: complex,
) -> None:
    """Check that NullEnvironment has been implemented correctly using constant transmissions."""
    receiver = Receiver(get_null_reception_algorithm())

    # place and simulate
    null_environment.place(transmitters, {"rx1": receiver, "rx2": receiver})
    device_logs = null_environment.simulate(10, 2)

    assert torch.all(device_logs.rx["rx1"]["raw"] == expected_value).item()
    assert torch.all(device_logs.rx["rx2"]["raw"] == expected_value).item()
