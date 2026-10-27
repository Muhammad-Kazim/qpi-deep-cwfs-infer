# methods/dic/
| File | Purpose |
|---|---|
| `__init__.py` | exports `DIC`, `zncc_displacement_grid`, `local_normalise`, `grid_to_full_resolution` |
| `dic_opd_retriever.py` | `DIC` retriever: grid displacement → bilinear upsampling to full resolution → OPD via the common interface |
| `zncc_displacement_estimation.py` | DIC core: local intensity normalisation, saturation mask, exact weighted ZNCC per lag, sub-pixel peak, median outlier test |
