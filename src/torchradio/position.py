"""Groups together common classes and functions for randomly sampling physical device positions in environments."""

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class Position:
    """A concrete instantiation of a device's position in the environment.

    Args:
        x: x coordinate
        y: y coordinate
        z: z coordinate

    """

    x: float
    y: float
    z: float


class SpatialDistribution(Protocol):
    """All devices are required to include a `SpatialDistribution`.

    A `SpatialDistribution` is any `Callable` returns a randomly sampled `Position` each time it is invoked.

    """

    def __call__(self) -> Position:
        """Randomly sample a `Position` from an underlying distribution.

        Returns
            A randomly sampled `Position`.

        """
        ...


def get_uniform_distribution(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
) -> SpatialDistribution:
    """Get a `SpatialDistribution` that samples uniformly from some 3D bounds.

    Args:
        x_bounds: Desired interval for the x coordinate
        y_bounds: Desired interval for the y coordinate
        z_bounds: Desired interval for the z coordinate

    Returns:
        A `SpatialDistribution` that returns a uniformly-sampled `Position` according to the provided bounds.

    Example:
        ```
        >>> distribution = get_uniform_distribution((0, 10), (0, 10), (0, 5))
        >>> distribution()
        Position(x=9.82526, y=1.6853619, z=1.1326883)
        ```

    Raises:
        ValueError: If `bounds[1] < bounds[0]`.

    """
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    z_min, z_max = z_bounds

    if x_max < x_min:
        err = f"{x_max=}<{x_min=}"
        raise ValueError(err)

    if y_max < y_min:
        err = f"{y_max=}<{y_min=}"
        raise ValueError(err)

    if z_max < z_min:
        err = f"{z_max=}<{z_min=}"
        raise ValueError(err)

    min_tensor = torch.tensor([x_min, y_min, z_min])
    max_tensor = torch.tensor([x_max, y_max, z_max])
    diff_tensor = max_tensor - min_tensor

    def distribution() -> Position:
        as_tensor = min_tensor + (torch.rand(3) * diff_tensor)
        return Position(*as_tensor.numpy())

    return distribution


def get_null_distribution(
    x: float = 0,
    y: float = 0,
    z: float = 0,
) -> SpatialDistribution:
    """Get a trivial or (null) `SpatialDistribution` that always samples the same `Position`.

    Args:
        x: x coordinate
        y: y coordinate
        z: z coordinate

    Returns:
        A `SpatialDistribution` that always returns the same `Position` as described by `x`, `y` and `z`.

    Example:
        ```
        >>> distribution = get_null_distribution(1, 2)
        >>> distribution()
        Position(x=1, y=2, z=0)
        ```

    """
    return lambda: Position(x, y, z)
