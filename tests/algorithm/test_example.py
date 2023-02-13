# ruff: noqa: D103

from torchradio.algorithm.example import DenseRadio


def test_example() -> None:
    dense_radio = DenseRadio(8, 2)
    tx = dense_radio.tx(64, 2)
    rx = dense_radio.rx(tx.signal)

    assert tx.metadata["bits"].shape == rx["bits"].shape
