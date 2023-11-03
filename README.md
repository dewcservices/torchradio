[![](https://github.com/dewcservices/torchradio/actions/workflows/python.yml/badge.svg)](https://github.com/dewcservices/torchradio/actions/workflows/python.yml)

# Torchradio

A differentiable RF simulation environment written in PyTorch.

> [!NOTE]
> `torchradio` is *not* affiliated with the official PyTorch project and should not be considered as a PyTorch-supported library like `torchvision` or `torchaudio`.

## Overview
`torchradio` provides an ecosystem for creating novel radios using machine learning. Specifically, `torchradio` lets engineers train parameterized radios with gradient-based methods using [PyTorch](https://pytorch.org/). `torchradio` accomplishes this by modelling common RF effects with auto-differentiable blocks that allow gradients to backpropagate from an objective function.

## Why are gradients important?
The gradient tells us in what direction we should adjust our parameters to immediately improve performance. Without this information, we can usually only improve performance via guess-and-check. That is, guess some new parameters, re-evaluate the objective function, and see whether performance improved. The gradient removes much of this guesswork by providing an explicit direction for the next parameter update. This information is critical when parameter space has thousands, millions, or even billions of dimensions.

Gradients are **not** essential for optimization. There exist many gradient-free optimization methods that do not require specialized simulation environments such as `torchradio`. Gradient-free methods allow users to optimize parameters with relatively simple techniques based on random search. Unfortunately these methods scale poorly as the number of parameters increases and are rarely used in conjunction with parameter-rich algorithms like deep neural networks (though there are some notable counter-examples such as [https://arxiv.org/pdf/1703.03864.pdf](https://arxiv.org/pdf/1703.03864.pdf)).

## Assumptions
- All events take place at baseband.
- All devices have the same centre frequency and bandwidth.
- `torchradio` is not intended to replace a high-fidelity simulation. Rather, it is a training ground for developing novel radios. If a radio looks promising, its parameters can be exported for testing in a high-fidelity simulation environment.

## Installation

Install the `torchradio` package by running:

```
pip install git+https://github.com/dewcservices/torchradio.git
```

## Contributing

New contributors are always welcome! If you would like to contribute, it is recommended you set up your development environment using the following instructions.

Create a new Python virtual environment using your method of choice (e.g., `venv`, `conda`, `pyenv` etc.). Clone this repository and install using

```
pip install -e .[dev]
```

The above command will install `torchradio` along with its core dependencies, as well as dev-specific dependencies for formatting, linting and testing. The `-e` flag installs `torchradio` in editable mode, so you can quickly see the effects of local source code changes without reinstalling `torchradio`. You can test that everything is working as expected by running

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
