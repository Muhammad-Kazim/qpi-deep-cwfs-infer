# tests/
Run with `pytest -v`. Deep-CWFS tests skip automatically without checkpoints or data
(`../data/…`; override the data folder with the `SPECKLE_OPD_DATA` environment variable).
`SPECKLE_OPD_DEVICE=cuda` runs the tests on the GPU (default cpu); `SPECKLE_OPD_ORIGINAL_REPO` points to the original
qpi-deep-cwfs-infer code if `archive/` is absent (see `gpu_server/`).

| File | Purpose |
|---|---|
| `conftest.py` | helpers: synthetic speckle, known phase bump, forward warp, data paths, test device |
| `reference_numpy_implementations_validated.py` | frozen NumPy/SciPy copy of the validated DIC and integrators (before the torch port); reference only, do not edit |
| `test_torch_ports_match_validated_numpy_reference.py` | torch DIC grid, DIC retriever, grid interpolation, integrators and padding reproduce the NumPy reference (≤1e-10) |
| `test_core_integration_setup_and_result.py` | integrators (exact Neumann, zero boundary, names), `Setup` scaling, result saving |
| `test_dic_zncc_displacement_on_synthetic_speckle.py` | DIC core accuracy on known shifts and amplitude-only changes |
| `test_opd_sign_convention_physical_speckle_simulation.py` | OPD sign from physics: diffuser + thin object, angular-spectrum propagation, DIC → higher RI positive, lower RI negative (textbook) |
| `test_dic_opd_retriever_end_to_end.py` | DIC via the common interface: phase in → OPD out (sign, scale), validation |
| `test_method_registry_and_command_line.py` | registry (lazy), `get_method`/`retrieve`, CLI with config + `--param` |
| `test_deep_cwfs_patch_tiling_and_stitching.py` | tiling coverage, exact stitching, padding |
| `test_deep_cwfs_opd_retriever.py` | Deep CWFS via the interface (random SEA-RAFT weights), checkpoint resolution, regression vs the original script (default checkpoint only), **sign agreement with DIC**; patches padded to 500 × 500 before the network |
