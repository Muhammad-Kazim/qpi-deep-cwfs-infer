# methods/deep_cwfs/
| File / folder | Purpose |
|---|---|
| `__init__.py` | exports `DeepCWFS` |
| `deep_cwfs_opd_retriever.py` | `DeepCWFS` retriever: tiling → patches reflect-padded to 500 × 500 (`pad_size`, the training input size) → network flow → displacement = −flow (labels = −d) → OPD |
| `optical_flow_network_builders.py` | builds RAFT (torchvision) / SEA-RAFT for 1-channel speckle and loads CWFS checkpoints; guesses the model from the checkpoint name |
| `patch_tiling_and_stitching.py` | tile grid for any image shape, weighted stitching (`mean` / `feather`), mirror (reflect) padding of small images and patches |
| `sea_raft/` | vendored SEA-RAFT network code (BSD-3-Clause, lightly modified) |
