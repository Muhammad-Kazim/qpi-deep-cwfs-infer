# notebooks/
Explanatory notebooks, saved with their outputs (figures). Open with `pip install -e ".[dev]"` (development extra) and `jupyter lab`.

| File | Purpose |
|---|---|
| `opd_sign_convention_explained.ipynb` | why OPD = +px²/z·∫d (textbook, higher RI positive) for every method: BPM through a bead, phase-vs-OPD sign conventions, speckle simulation + DIC, Deep CWFS labels = −d, summary table |
| `integration_boundary_conditions_explained.ipynb` | the four integrators (`poisson_neumann` default, `poisson_zero_boundary`, `fourier_mirror`, `fourier_asdi`) on synthetic phases with known truth and on a real pair; why mirror padding loses tilts and ASDI does not; how to remove piston/tilt bias before an RMSE, and gradient vs OPD error |
