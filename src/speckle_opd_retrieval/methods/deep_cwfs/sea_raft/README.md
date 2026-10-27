# methods/deep_cwfs/sea_raft/
Vendored from https://github.com/princeton-vl/SEA-RAFT (BSD-3-Clause). Changes: relative imports,
no Hugging Face Hub mixin, ImageNet backbone download made opt-in (`args.init_weight`).

| File | Purpose |
|---|---|
| `__init__.py` | exports `RAFT` (the SEA-RAFT model) |
| `raft.py` | SEA-RAFT model: context/feature encoders, iterative update, convex upsampling |
| `extractor.py` | ResNet-FPN feature extractor |
| `corr.py` | correlation volume and lookup |
| `update.py` | motion encoder and update block |
| `sea_raft_layer.py` | building blocks (ConvNeXt block, residual block, conv helpers) |
