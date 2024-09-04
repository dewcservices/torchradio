[![](https://github.com/dewcservices/torchradio/actions/workflows/python.yml/badge.svg)](https://github.com/dewcservices/torchradio/actions/workflows/python.yml)

# Torchradio

Torchradio is a Python library for building differentiable RF simulations.

> [!NOTE]
> Torchradio is *not* affiliated with the official PyTorch project.

## Installation

```
pip install torchradio
```

## Example

Below is a simple example that trains two radios via backpropagation to communicate over the same noisy channel:

```python
# Define the training environment
from torchradio import Transmitter, Receiver
from torchradio.algorithm.example import DenseRadio
from torchradio.env.null import RandomAWGNEnvironment

n_radios = 2
radio0 = DenseRadio(n_input_bits=8, tx_length_per_bit=4)
radio1 = DenseRadio(n_input_bits=8, tx_length_per_bit=4)
env = RandomAWGNEnvironment(p_min=0, p_max=1)
env.place(
    transmitters={"tx0": Transmitter(radio0.tx), "tx1": Transmitter(radio1.tx)},
    receivers={"rx0": Receiver(radio0.rx), "rx1": Receiver(radio1.rx)},
)


# Evaluate the initial radios
import numpy as np

def evaluate():
    simulation_logs = env.simulate(n_timesteps=10000, batch_size=10)
    tx_bits = {k: v.metadata["bits"] for k, v in simulation_logs.tx.items()}
    rx_bits = {k: v["bits"] for k, v in simulation_logs.rx.items()}
    for i in range(n_radios):
        print(f'radio{i} BER: {1 - float(np.mean((tx_bits[f"tx{i}"] == rx_bits[f"rx{i}"]).numpy())):.5f}')

evaluate()


# Define the training loop
import torch
from torch import nn

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam([*radio0.parameters(), *radio1.parameters()])

def train(
    n_timesteps: int,
    batch_size: int,
) -> float:
    optimizer.zero_grad()
    device_logs = env.simulate(n_timesteps, batch_size)
    tx_bits = {k: v.metadata["bits"] for k, v in device_logs.tx.items()}
    rx_outputs = {k: v["bit_probabilities"] for k, v in device_logs.rx.items()}
    loss = sum([
        loss_fn(rx_outputs[f"rx{i}"], tx_bits[f"tx{i}"].float())
        for i in range(n_radios)
    ])
    loss.backward()
    optimizer.step()
    return loss


# Train the radios
for i in range(1000):
    loss = train(n_timesteps=64, batch_size=10)
    if i % 100 == 0:
        print(f"Loss at iteration {i}: {loss:5f}")


# Evaluate the trained radios
evaluate()

```

See [our notebooks](https://dewcservices.github.io/torchradio/Examples/1_introduction/) for more in-depth examples.


## Assumptions
- All events take place at baseband.
- All devices have the same centre frequency and bandwidth.
- Torchradio is not intended to replace a high-fidelity simulation. Rather, it is a training ground for developing novel radios. If a radio looks promising, its parameters can be exported for testing in a high-fidelity simulation environment.


## Contributing

New contributors are always welcome! If you would like to contribute, it is recommended you set up your development environment using the following instructions.

Create a new Python virtual environment using your method of choice (e.g., venv, conda, pyenv etc.). Clone this repository and install using

```
pip install -e .[dev]
```

The above command will install Torchradio along with its core dependencies, as well as dev-specific dependencies for formatting, linting and testing. The `-e` flag installs Torchradio in editable mode, so you can quickly see the effects of local source code changes without reinstalling Torchradio. You can test that everything is working as expected by running

```
pytest
```

To save failing GitHub Actions due to styling issues, set up the project's git hooks using:

```
pre-commit install
pre-commit run --all-files
```

You can view the documentation locally anytime by running:

```
mkdocs serve
```
