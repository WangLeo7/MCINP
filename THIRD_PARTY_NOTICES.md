# Third-party notices

## Source distributed with MCINP

The MCINP source is released under the MIT license by Shenpei Wang and Hairong Lian. See LICENSE.

The forward method in [canonical_spectral.py](mcinp/methods/fno/canonical_spectral.py) adapts
[NeuralOperator 2.0.0](https://github.com/neuraloperator/neuraloperator/releases/tag/2.0.0)
(commit b307030, neuralop/layers/spectral_convolution.py). It replaces the real inverse FFT
with an explicit inverse transform along the complex axes followed by a real inverse FFT,
with real DC and Nyquist entries on the last axis. The network and spectral weights use
the installed NeuralOperator package. The upstream notice for the adapted code follows.

```text
MIT License

Copyright (c) 2023 NeuralOperator developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Separately installed dependencies

| Package | Version | License in the installed distribution |
| --- | --- | --- |
| PyTorch | 2.6.0 | BSD-3-Clause |
| NeuralOperator | 2.0.0 | MIT |
| NumPy | 2.4.6 | BSD-3-Clause and additional notices (0BSD, MIT, Zlib, CC0-1.0) |
| SciPy | 1.17.1 | BSD-3-Clause, with bundled notices |
| TensorLy | 0.9.0 | Modified BSD |
| TensorLy-Torch | 0.5.0 | Modified BSD |
| opt-einsum | 3.4.0 | MIT |
| threadpoolctl | 3.6.0 | BSD-3-Clause |
| configmypy | 0.2.0 | Modified BSD |
| wandb | 0.30.0 | MIT |
| ruamel.yaml | 0.19.1 | MIT |

Dependency licenses and bundled notices are supplied by the respective distributions.
The project includes local implementations of DeepONet and POD-linear. The manuscript
credits the underlying methods. This repository distributes source code and example
settings. The project license applies to its source code.
