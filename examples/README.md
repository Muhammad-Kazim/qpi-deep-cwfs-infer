# examples/
| File | Purpose |
|---|---|
| `example_compare_methods_on_one_speckle_pair.py` | run any set of methods (`--methods dic deep_cwfs`) on one reference/specimen pair through the common interface and plot the OPD maps side by side (`--integrator` to switch) |
| `example_deep_cwfs_stitching_and_integrators.ipynb` | Deep CWFS with one patch vs 3×3 stitched patches (`mean` / `feather` blending, seams), then the same displacement integrated with all four integrators, and switching `integrator=` on a method; run it on a GPU (`gpu_server/README.md`, step 5) |
