import pytest
import torch

from torchradio.algorithm import DSSS


@pytest.mark.parametrize(
    ("chip_sequence", "mode", "n_symbols"),
    [
        (torch.tensor([1, 0]), "psk", 2),
        (torch.tensor([1, 0]), "qam", 4),
        (torch.tensor([1, 0, 1, 1]), "psk", 4),
        (torch.tensor([1, 0, 1, 1]), "qam", 16),
        (torch.tensor([1, 1, 1, 1, 0, 0, 1, 1]), "psk", 2),
        (torch.tensor([1, 0, 1, 1, 0, 1, 1, 1]), "psk", 16),
        (torch.tensor([1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1]), "qam", 16),
        (torch.tensor([1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]), "psk", 2),
    ],
)
def test_dsss(chip_sequence: torch.Tensor, mode: str, n_symbols: int) -> None:
    """Check DSSS modulation and demodulation are correct."""
    dsss = DSSS(chip_sequence, mode, n_symbols)
    transmission = dsss.tx(80, 4)
    reception = dsss.rx(transmission.signal)
    assert torch.all(transmission.metadata["bits"] == reception["bits"]).item()
