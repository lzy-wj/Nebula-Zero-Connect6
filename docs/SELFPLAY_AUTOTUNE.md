# Self-play hardware autotuning

The tuner keeps MCTS and sampling semantics fixed. It changes only execution
parameters: workers per GPU, persistent contexts, CPU threads, batch size,
NUMA affinity, and capacity layout.

Quick hardware check:

```bash
experiments/pair_native/autotune_capacity.sh \
  --checkpoint /path/to/model.pth \
  --output-dir /path/to/autotune-quick \
  --capacities 1,2,4,8 \
  --mode quick
```

Production profiles with audited complete-game validation:

```bash
experiments/pair_native/autotune_capacity.sh \
  --checkpoint /path/to/model.pth \
  --output-dir /path/to/autotune-full \
  --capacities 1,2,4,8 \
  --mode full
```

By default the tool selects idle, homogeneous GPUs. Use `--gpu-ids 0,1,2,3`
to pin a specific set. Results are written as `profiles/1gpu.json`,
`profiles/2gpu.json`, and so on. A profile can be passed directly to
`selfplay --search-profile` or `async_loop --search-profile`.

`quick` is a screening result. Use `full` before a long production run: it
warms every persistent pool, measures later replay shards, audits every shard,
and rejects insufficient game diversity. Benchmark output is isolated from
the training replay tree and is bound to the model, hardware, source, and
search configuration by a request manifest.
