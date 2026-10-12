# Pair-native V3 release

This release is split deliberately:

- GitHub contains source, tests, documentation, and reproducible exporters.
- The model hub contains inference and resumable-training checkpoints.
- The dataset hub contains the exact audited replay window used by the promoted checkpoint.

Large binary artifacts must not be committed to Git history.

## Build the model package

```bash
python -m experiments.pair_native.export_release \
  --checkpoint /path/to/promoted/best.pth \
  --state /path/to/run/state.json \
  --through-round 452 \
  --output-dir /path/to/release/model
```

The output contains a compact inference checkpoint, the original training checkpoint,
model configuration, a model card, a machine-readable manifest, and SHA256 checksums.

## Build the exact replay package

```bash
python -m experiments.pair_native.prepare_public_dataset \
  --state /path/to/run/state.json \
  --through-round 452 \
  --output-dir /path/to/release/dataset
```

The exporter selects replay directories from the immutable successful promotion entry,
not from the latest files in the run directory. Every source shard and copied public shard
is strictly audited. Host paths and CPU-affinity details are removed from public manifests.

## Verify before upload

```bash
cd /path/to/release/model && sha256sum -c SHA256SUMS
cd /path/to/release/dataset && sha256sum -c SHA256SUMS
python -m pytest -q
```

Upload only after the checkpoint hash, aggregate game/turn counts, cards, licenses, and
destination repository names have been reviewed. Creating repositories, tags, releases,
or pushing files is a separate explicitly authorized step.
