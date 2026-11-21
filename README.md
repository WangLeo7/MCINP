# MCINP

Python implementation of the **Mean-coefficient Interaction Neural Propagator** for periodic
directional space-fractional diffusion. Includes the Fourier
reference solver and six methods: MCINP, Global-Q, MCP, FNO, DeepONet and POD-linear.

## Run

Use Python 3.11 and PyTorch 2.6.0. Install the PyTorch build for your CPU or CUDA platform first.

~~~sh
python -m pip install -r requirements.txt
python main.py run --root runs/example
python main.py run --dimension 2 --method DeepONet --root runs/example_2d
~~~

Each run uses one method and seed. You can call the stages separately as **generate**, **train**
and **evaluate**, with the same settings and output directory. Use **--config** for a JSON override,
or **--method**, **--dimension**, **--seed** and **--device** for individual overrides.
Use a fresh output directory after changing the source, data settings or training configuration.
Methods with the same data settings can share generated reference data.

## Edit the example

- **Grid, query times, sample counts, and training settings** are in [settings.json](settings.json).
  The defaults are a small runnable example. Set **test_N** to evaluate on another grid. Training
  is performed on **N**. 
- **Initial state, positive diffusion fields and directional orders** are defined by
  **sample_id_1d** and **sample_id_2d** in [datasets.py](mcinp/runtime/datasets.py). These functions
  implement the paper's central (ID) distributions on the endpoint-excluded periodic grid.
  For a custom initial condition, supply your own center, width and amplitude to **periodic_gaussian**,
  or use **random_fourier** with your own modes.
- **Perturbed observations or order changes** can be configured in **make_case**. Apply changes to
  the test split. For noise, retain clean fields for the reference solve and keep the observed
  coefficients positive. For switching, set a switch time within the observation interval and one
  post-switch order per direction. Predictions after the switch start from the predicted switch state.

## Output and citation

Each run writes **per_instance_metrics.csv** under its directory in **slots/**. The raw trajectory
and terminal errors use the unprojected predictions. The trajectory error combines the positive-time
states before normalization. Iterative training writes **history.csv**. Per-case NPZ files contain
predictions, reference fields and observation times. The forward timing column measures individual
evaluation calls.

Please cite *MCINP: Fourier Neural Propagator with Mean-Coefficient Interaction for Spatial Fractional Diffusion Equations*. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
