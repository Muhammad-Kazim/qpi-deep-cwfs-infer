# methods/
One subfolder per method. Each defines an `OPDRetriever` subclass and is registered in
`core/retriever_interface_and_method_registry.py` (`METHODS`). All methods are written in torch and run on
the retriever's `device` (CPU or GPU).

| Folder | Method |
|---|---|
| `__init__.py` | (package marker; methods are imported lazily) |
| `dic/` | Digital image correlation (ZNCC), robust to amplitude changes |
| `deep_cwfs/` | Deep CWFS with CWFS-trained RAFT / SEA-RAFT networks |

To add a method (e.g. classical CWFS): create `methods/<name>/`, subclass `OPDRetriever`,
implement `_estimate(reference, specimen) -> MethodOutput` in torch (inputs are tensors on `self.device`
in `self.dtype`; return tensors there), add one line to `METHODS`, add tests.
