# src/speckle_opd_retrieval/
| File / folder | Purpose |
|---|---|
| `__init__.py` | public API: `Setup`, `get_method`, `available_methods`, `retrieve`, `RetrievalResult`, `load_image`, … |
| `core/` | pieces shared by all methods: types, the retriever interface + registry, integration, I/O |
| `methods/` | one subfolder per OPD retrieval method |
| `command_line_interface.py` | the `speckle-opd-retrieval` command (`methods`, `run`) |
