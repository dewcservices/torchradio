"""Simple box-like environments."""

from dataclasses import dataclass
from os import PathLike

from torchradio.device import Position
from torchradio.env.null import NullEnvironment


@dataclass
class Bounds3D:
    """A 3D region bounded by `[0, 0, 0]` and `[x_max, y_max, z_max]`.

    Args:
        x_max: Bound x coordinates between 0 and `x_max`.
        y_max: Bound y coordinates between 0 and `y_max`.
        z_max: Bound z coordinates between 0 and `z_max`.

    Example:
        ```
        >>> bounds = Bounds3D(10, 10, 5)
        >>> Position(9, 6, 4) in bounds
        True
        >>> Position(9, 6, 6) in bounds
        False
        >>> Position(9, -1, 4) in bounds
        False
        ```

    """

    x_max: float
    y_max: float
    z_max: float

    def __contains__(self, position: Position) -> bool:
        """Determine if position is inside `Bounds3D`.

        Args:
            position: Position to test.

        Returns:
            True if the position is in bounds.

        """
        if position.x < 0 or position.y < 0 or position.z < 0:
            return False

        return (
            position.x <= self.x_max
            and position.y <= self.y_max
            and position.z <= self.z_max
        )


class BoxEnvironment(NullEnvironment):
    """A simple environment defined by a 3D region."""

    def __init__(self, x_max: float, y_max: float, z_max: float) -> None:
        """Create a new bounded `BoxEnvironment`.

        A `Position` is in bounds if `0 <= x <= x_max`, `0 <= y <= y_max` and `0 <= z <= z_max`.

        Args:
            x_max: Maximum x coordinate
            y_max: Maximum y coordinate
            z_max: Maximum z coordinate


        Raises:
            ValueError: if `x_max < 0` or `y_max < 0` or `z_max < 0`.

        """
        if x_max < 0:
            err = f"{x_max=} < 0"
            raise ValueError(err)

        if y_max < 0:
            err = f"{y_max=} < 0"
            raise ValueError(err)

        if z_max < 0:
            err = f"{z_max=} < 0"
            raise ValueError(err)

        self._bounds = Bounds3D(x_max, y_max, z_max)
        super().__init__()

    @property
    def bounds(self) -> Bounds3D:
        """Public API for internal bounds.

        Returns
            Three-dimensional bounds for the environment.

        """
        return self._bounds

    def visualize(
        self,
        *,
        show: bool = True,
        save_path: PathLike | None = None,
    ) -> None:
        """Visualize the environment with the currently placed devices."""
        raise NotImplementedError

    def _in_bounds(self, position: Position) -> bool:
        return position in self._bounds


class PlanarEnvironment(BoxEnvironment):
    """A simple environment defined by a flat 2D region.

    Devices must have spatial distributions that place them at a height of z=0.
    """

    def __init__(self, x_max: float, y_max: float) -> None:
        """Create a new bounded `PlanarEnvironment`.

        A `Position` is in bounds if `0 <= x <= x_max`, `0 <= y <= y_max` and `z == 0`.

        Args:
            x_max: Maximum x coordinate
            y_max: Maximum y coordinate


        Raises:
            ValueError: if `x_max < 0` or `y_max < 0`.

        """
        super().__init__(x_max, y_max, 0)

    def visualize(
        self,
        *,
        show: bool = True,
        save_path: PathLike | None = None,
    ) -> None:
        """Visualize the environment with the currently placed devices."""
        raise NotImplementedError
