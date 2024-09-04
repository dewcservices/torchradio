"""Defines the `BaseEnvironment` that all `Torchradio` environments are expected to inherit from."""

import warnings
from abc import ABC, abstractmethod
from os import PathLike

import torch

from torchradio.algorithm.null import (
    get_constant_transmission_algorithm,
    get_null_reception_algorithm,
)
from torchradio.device import Position, Receiver, Transmitter
from torchradio.types import DeviceLogs


class BaseEnvironment(ABC):
    """Abstract base class for `Torchradio` environments.

    The core concept behind `Torchradio` is the use of a common simulation environment
    that enables backpropagation from the loss function back to the transmitters. It is
    assumed that all environments used with `Torchradio` are child classes of
    `BaseEnvironment`. The `BaseEnvironment` automates many common operations, such as
    device placement, noise simulation and signal aggregation.

    All child classes of `BaseEnvironment` are expected to implement the abstract methods:
    `_compute_propagation_parameters`, `_in_bounds`, `_propagate` and `_get_background_noise`.
    """

    def __init__(self, *, disable_differentiability_check: bool = False) -> None:
        """Create a clean environment that can be populated with devices."""
        if not disable_differentiability_check and not self.is_differentiable():
            msg = f"{self} does not appear to be differentiable. Check your environment logic. Raise an issue at the GitHub repository if you have received this warning in error."
            warnings.warn(msg, stacklevel=2)
        self.reset()

    def reset(self) -> None:
        """Remove all devices from the environment.

        Example:
            ```
            >>> env.place(...)
            >>> env.n_devices
            8
            >>> env.reset()
            >>> env.n_devices
            0
            ```

        """
        self._transmitters: dict[str, Transmitter] = {}
        self._receivers: dict[str, Receiver] = {}

        self._transmitter_positions: dict[str, Position] = {}
        self._receiver_positions: dict[str, Position] = {}

    @property
    def n_transmitters(self) -> int:
        """Get current number of transmitters.

        Returns
            The number of transmitters placed in the environment.

        """
        return len(self._transmitters)

    @property
    def n_receivers(self) -> int:
        """Get current number of receivers.

        Returns
            The number of receivers placed in the environment.

        """
        return len(self._receivers)

    @property
    def n_devices(self) -> int:
        """Get current number of devices.

        Returns
            The number of devices placed in the environment.

        """
        return self.n_transmitters + self.n_receivers

    @property
    def transmitters(self) -> dict[str, Position]:
        """Get current transmitters and their positions.

        Returns
            A map from transmitter names to positions

        """
        return self._transmitter_positions

    @property
    def receivers(self) -> dict[str, Position]:
        """Get current receivers and their positions.

        Returns
            A map from receiver names to positions

        """
        return self._receiver_positions

    @property
    def devices(self) -> dict[str, dict[str, Position]]:
        """Get a dictionary that summarizes the environment's devices with their current positions.

        Returns
            Two maps. One that maps transmitter names to positions and one that maps receiver names to positions.

        """
        return {"transmitters": self.transmitters, "receivers": self.receivers}

    def place(
        self,
        transmitters: dict[str, Transmitter],
        receivers: dict[str, Receiver],
    ) -> None:
        """Place devices in the environment.

        Child classes must specify whether a device is in or out-of-bounds via the `_in_bounds` method.
        After the devices have been placed, `self._compute_propagation_parameters()` is called to determine
        simulation parameters. These simulation parameters do not need to be recomputed until the devices have
        been re-placed.

        Args:
            transmitters: Maps device names to `Transmitter`s.
            receivers: Maps device names to `Receivers`s.

        Example:
            ```
            >>> transmitter_1 = torchradio.algorithm.null.get_constant_transmitter(1 + 1j)
            >>> transmitter_2 = torchradio.algorithm.null.get_constant_transmitter(1 + 0j)
            >>> transmitter_3 = torchradio.algorithm.null.get_null_transmitter()
            >>> receiver = torchradio.algorithm.null.get_null_receiver()
            >>> some_transmitters = {"tx1": transmitter_1, "tx2": transmitter_2}
            >>> other_transmitters = {"tx3": transmitter_3}
            >>> all_transmitters = {**some_transmitters, **other_transmitters}
            >>> receivers = {"rx": receiver}
            >>> env = torchradio.env.null.NullEnvironment()
            >>> env.place(some_transmitters, receivers)
            >>> env.n_devices
            3
            >>> env.place(all_transmitters, receivers)
            >>> env.n_devices
            4
            ```

        Raises:
            ValueError: No receivers provided.
            TypeError: A non-`Transmitter` was found in `transmitters` or a non-`Receiver` was found in `receivers`.

        """
        if len(receivers) == 0:
            err = f"{self.__class__.__name__} requires at least one receiver: {len(receivers)=}"
            raise ValueError(err)

        for transmitter_name, transmitter in transmitters.items():
            if not isinstance(transmitter, Transmitter):
                err = f"transmitters contains a non-transmitter: {transmitter_name}: {transmitter.__class__.__name__}"
                raise TypeError(err)

        for receiver_name, receiver in receivers.items():
            if not isinstance(receiver, Receiver):
                err = f"receivers contains a non-receiver: {receiver_name}: {receiver.__class__.__name__}"
                raise TypeError(err)

        self.reset()

        self._transmitters = transmitters
        self._receivers = receivers

        self._transmitter_positions = self._place_devices(self._transmitters)
        self._receiver_positions = self._place_devices(self._receivers)

        self._compute_propagation_parameters()

    def simulate(self, n_timesteps: int, batch_size: int = 1) -> DeviceLogs:
        """Run the simulation for `n_timesteps` with `batch_size`.

        Args:
            n_timesteps: How many timesteps to simulate. Must be positive.
            batch_size: How many batches to simulate. Must be positive.

        Returns:
            Device logs for benchmarking performance and computing gradients.

        Example:
            ```
            >>> transmitter = torchradio.algorithm.null.get_null_transmitter()
            >>> receiver = torchradio.algorithm.null.get_null_receiver()
            >>> env = torchradio.env.null.NullEnvironment()
            >>> env.place({"tx": transmitter}, {"rx": receiver})
            >>> env.simulate(20, 3)
            ```

        Raises:
            ValueError: If a provided `SpatialDistribution` is incompatible with the environment
                according to the child class `_in_bounds` methods.
            RuntimeError: If child classes don't correctly implement `_propagate` or
                `_get_background_noise` to account for all devices.

        """
        # get transmissions from every transmitter in the environment for n_timesteps
        tx_logs = {}
        for transmitter_name, transmitter in self._transmitters.items():
            transmission = transmitter(n_timesteps, batch_size)

            if tuple(transmission.signal.shape) != (batch_size, n_timesteps):
                err = f"Received signal with shape {transmission.signal.shape} from transmitter: {transmitter_name}. Expected {(batch_size, n_timesteps)}."
                raise RuntimeError(err)

            tx_logs[transmitter_name] = transmission

        # propagate transmitted signals to every receiver in the environment
        if self.n_transmitters > 0:
            propagations = self._propagate(
                {
                    transmitter_name: tx.signal
                    for transmitter_name, tx in tx_logs.items()
                },
            )
            if propagations.keys() != self._receivers.keys():
                err = f"self._propagate did not account for all receivers: {propagations.keys()=} vs {self._receivers.keys()=}."
                raise RuntimeError(err)
        else:
            propagations = {
                receiver_name: torch.zeros(
                    [batch_size, n_timesteps],
                    dtype=torch.complex64,
                )
                for receiver_name in self.receivers
            }

        # get random background noise for every receiver in the environment
        background_noise = self._get_background_noise(n_timesteps, batch_size)
        if background_noise.keys() != self._receivers.keys():
            err = f"self._get_background_noise did not account for all receivers: {background_noise.keys()=} vs {self._receivers.keys()=}."
            raise RuntimeError(err)

        # sum the aggregated transmissions with the background noise for every receiver in the environment
        rx = {
            receiver_name: background_noise[receiver_name] + propagations[receiver_name]
            for receiver_name in self._receivers
        }

        # process the received signals
        rx_logs = {}
        for receiver_name, receiver in self._receivers.items():
            raw = rx[receiver_name]
            rx_logs[receiver_name] = receiver(raw)

            # track received signal before receiver processing
            if "raw" in rx_logs[receiver_name]:
                err = f'{receiver_name} uses the reserved keyword "raw" in the Reception dictionary.'
                raise RuntimeError(err)
            rx_logs[receiver_name]["raw"] = raw

            # record isolated background noise for training and analysis
            if "noise" in rx_logs[receiver_name]:
                err = f'{receiver_name} uses the reserved keyword "noise" in the Reception dictionary.'
                raise RuntimeError(err)
            rx_logs[receiver_name]["noise"] = background_noise[receiver_name]

        return DeviceLogs(tx=tx_logs, rx=rx_logs)

    def _place_devices(
        self,
        devices: dict[str, Transmitter] | dict[str, Receiver],
        max_n_attempts: int = 100,
    ) -> dict[str, Position]:
        """Place an iterable of devices in the environment according to their spatial distributions.

        Args:
            devices: Maps devices names to `Device`s.
            max_n_attempts: How many attempts to try placing each device before raising a `ValueError`.

        Returns:
            A mapping from device names to `Position`s.

        Raises:
            ValueError: If a device cannot be suitably placed within `max_n_attempts`. This indicates
                that the provided `SpatialDistribution` is unsuitable for the current environment.

        """
        positions = {}
        for device_name, device in devices.items():
            n_attempts = 0
            while True:
                n_attempts += 1
                if n_attempts >= max_n_attempts:
                    err = f"Device {device_name} has a spatial distribution that is incompatible with the current environment."
                    raise ValueError(err)

                position = device.place()

                if self._in_bounds(position):
                    break

            positions[device_name] = position

        return positions

    def visualize(
        self,
        *,
        show: bool = True,
        save_path: PathLike | None = None,
    ) -> None:
        """Visualize the environment with the currently placed devices.

        It is not mandatory for child classes to override this method. However, it may make it
        easier for users to interact with the environment if they are provided with a convenient
        visualization method to see where devices are currently placed.

        Args:
            show: Show an interactive plot within the Python process.
            save_path: Path to save the visualization to.

        """
        raise NotImplementedError

    @abstractmethod
    def _compute_propagation_parameters(self) -> None:
        """Compute and store any necessary propagation parameters as object attributes before the simulation begins."""
        ...

    @abstractmethod
    def _in_bounds(self, position: Position) -> bool:
        """Determine whether the device is in bounds according to the environment's specifications.

        Args:
            position: A position in the environment

        Returns:
            True if the position is valid according to the environment's specifications.

        """
        ...

    @abstractmethod
    def _propagate(self, tx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Propagate transmitted signals to every receiver in the environment.

        Args:
            tx: A map from transmitter names to transmitted signals.

        Returns:
            A dictionary mapping each receiver to a complex-valued input signal (not including background noise).

        """
        ...

    @abstractmethod
    def _get_background_noise(
        self,
        n_timesteps: int,
        batch_size: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Compute random background noise for every receiver in the environment.

        Args:
            n_timesteps: Number of timesteps of background noise.
            batch_size: How many batches of background noise.

        Returns:
            A dictionary mapping each receiver to a complex-valued background noise
                tensor with shape `[n_timesteps, batch_size]`.

        """
        ...

    def is_differentiable(self) -> bool:
        """Check the environment is differentiable.

        Gotcha: The differentiability check assumes that devices can be placed at
        Position(x=0, y=0, z=0). If this is impossible, you must either write your own
        differentiability check, or alter your environment to allow placements at
        the origin.

        Gotcha: As part of the differentiability check, the environment will be reset.
        If you have devices placed in the environment before calling this function, they
        will need to be re-placed afterwards.

        Returns:
            True if the environment passes a basic differentiability test, where the output
                magnitude of a simple transmitter is driven to zero. If your environment is
                incorrectly classified as non-differentiable, please raise an issue at
                the repository issue tracker.

        Example:
                ```
                >>> env = torchradio.env.null.NullEnvironment()
                >>> env.is_differentiable()
                True
                ```

        """
        self.reset()

        # get basic trainable transmitter and a null receiver
        init_val = 1 + 1j
        weight = torch.nn.Parameter(torch.tensor(init_val))
        transmitter = Transmitter(get_constant_transmission_algorithm(weight))
        receiver = Receiver(get_null_reception_algorithm())

        # place and simulate
        self.place({"tx": transmitter}, {"rx": receiver})
        device_logs = self.simulate(10, 2)

        # create an arbitrary loss function that drives the raw received signal to zero
        raw = device_logs.rx["rx"]["raw"]
        sum_squares = torch.sum(raw.real**2 + raw.imag**2)
        opt = torch.optim.SGD([weight], lr=0.1)

        try:
            torch.nn.MSELoss()(
                sum_squares,
                torch.tensor(0, dtype=torch.float32),
            ).backward()
        except RuntimeError:
            return False  # tensors does not require grad and does not have a grad_fn

        opt.step()

        self.reset()

        return (
            weight.item() != init_val
        )  # if differentiable, the weight parameter should have changed
