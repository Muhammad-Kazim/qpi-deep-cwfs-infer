# methods/
One subfolder per method. Each defines an `OPDRetriever` subclass and is registered in
`core/retriever_interface_and_method_registry.py` (`METHODS`).

| Folder | Method | Needs |
|---|---|---|
| `__init__.py` | (package marker; methods are imported lazily) | — |
| `dic/` | Digital image correlation, robust to amplitude changes | numpy, scipy |
| `deep_cwfs/` | Deep CWFS with CWFS-trained RAFT / SEA-RAFT networks | torch, torchvision (`[deep]`) |

To add a method (e.g. classical CWFS): create `methods/<name>/`, subclass `OPDRetriever`,
implement `_estimate(reference, specimen) -> MethodOutput`, add one line to `METHODS`, add tests.
