# core/
| File | Purpose |
|---|---|
| `__init__.py` | re-exports the core API |
| `optical_setup_and_retrieval_result.py` | `Setup` (pixel size, distance, OPD scale factors) and `RetrievalResult` (opd, opd_gradient, displacement, amplitude, params, info; `.save()`) |
| `retriever_interface_and_method_registry.py` | `OPDRetriever` base class (`retrieve`, `retrieve_many`), `MethodOutput`, and the lazy method registry (`get_method`, `available_methods`, `register`, `retrieve`) |
| `poisson_integration_gradient_to_opd.py` | displacement → OPD integrators in torch: `poisson_neumann` (default; DCT least squares), `poisson_zero_boundary` (original Deep CWFS CG), `fourier_mirror`, `fourier_asdi`; `integrate`, `available_integrators`, `DEFAULT_INTEGRATOR` |
| `torch_device_and_tensor_helpers.py` | `resolve_device` (default CUDA if available, else CPU), working dtype (float64; float32 on MPS), NumPy ↔ tensor conversion |
| `image_io_load_crop_save.py` | `load_image` (tiff/npy/npz/mat/png), `crop`, `save_result` (tiff/npy/png + meta.json) |
