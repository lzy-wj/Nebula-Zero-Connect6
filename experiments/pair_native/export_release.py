"""Build a self-contained public model package from a Pair checkpoint.

The source training checkpoint is never modified.  The exported inference file
contains only the model definition, weights, seed, and scalar training metrics;
the optimizer checkpoint is retained separately for reproducible fine-tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from experiments.pair_native.model import load_checkpoint, parameter_count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def model_card(manifest: dict) -> str:
    config = manifest["model"]["config"]
    promotion = manifest.get("promotion")
    promotion_text = ""
    if promotion:
        promotion_text = f"""
## Internal promotion evidence

- Through-round: `{promotion['through_round']}`
- Paired gate score: `{promotion['score']:.4f}`
- One-sided lower bound: `{promotion['lower_bound']:.4f}`
- Gate: `{promotion['gate_config']['pairs']}` opening pairs at
  `{promotion['gate_config']['simulations']}` simulations per complete turn

This gate compares the candidate with its immediate accepted baseline; it is not
an external rating.
"""
    return f"""---
license: mit
library_name: pytorch
tags:
- connect6
- alphazero
- mcts
- board-game
- self-play
---

# Nebula Zero Connect6 — Pair-native V3

面向六子棋的原生双落子策略价值网络。模型从随机初始化开始，通过自对弈训练；
一次策略动作直接表示无序棋子对，而不是把一回合拆成两次独立决策。

This is a native unordered-pair policy/value network for Connect6, trained from
random initialization with self-play. One policy action represents a complete
two-stone turn.

## Model

- Parameters: `{manifest['model']['parameters']:,}`
- Input: `{config['input_planes']} × 19 × 19`
- Sparse trunk: width `{config['feature_dim']}`, depth `{config['depth']}`,
  at most `{config['max_tokens']}` board tokens
- Pair rank: `{config['pair_rank']}`
- License: MIT

`pair_native_v3_inference.pth` is the recommended inference artifact.
`pair_native_v3_training.pth` additionally contains optimizer state for research
continuation. Search strength depends strongly on the MCTS budget; the production
self-play range was 400–1200 simulations per complete turn.

## Load

```python
from experiments.pair_native.model import load_checkpoint

model, metadata = load_checkpoint("pair_native_v3_inference.pth")
model.eval()
```

The implementation lives in
`experiments/pair_native/` of
[Nebula-Zero-Connect6](https://github.com/lzy-wj/Nebula-Zero-Connect6).
{promotion_text}

## Scope and limitations

The internal promotion score recorded in this package is a paired self-play gate,
not an external tournament rating. Results are specific to the stated model,
search configuration, hardware, and opening protocol. The checkpoint should not
be interpreted as solved-play or professional-tournament strength.
"""


def export_model(
    checkpoint_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    state_path: Path | None = None,
    through_round: int | None = None,
) -> dict:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_checkpoint(checkpoint_path)
    source_sha256 = sha256_file(checkpoint_path)
    inference_path = output_dir / "pair_native_v3_inference.pth"
    training_path = output_dir / "pair_native_v3_training.pth"
    config_path = output_dir / "config.json"

    inference_checkpoint = {
        "schema_version": int(checkpoint.get("schema_version", 1)),
        "model_config": dict(checkpoint["model_config"]),
        "seed": int(checkpoint.get("seed", 0)),
        "epoch": int(checkpoint.get("epoch", 0)),
        "metrics": json_value(checkpoint.get("metrics", {})),
        "model_state_dict": checkpoint["model_state_dict"],
        "release": {
            "name": "pair-native-v3",
            "source_checkpoint_sha256": source_sha256,
        },
    }
    torch.save(inference_checkpoint, inference_path)
    shutil.copy2(checkpoint_path, training_path)
    config_path.write_text(
        json.dumps(checkpoint["model_config"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifacts = {}
    for path in (inference_path, training_path, config_path):
        artifacts[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": 1,
        "release": "pair-native-v3",
        "source": {
            "checkpoint_sha256": source_sha256,
        },
        "model": {
            "parameters": parameter_count(model),
            "config": dict(checkpoint["model_config"]),
            "epoch": int(checkpoint.get("epoch", 0)),
            "metrics": json_value(checkpoint.get("metrics", {})),
        },
        "artifacts": artifacts,
    }
    if state_path is not None:
        state = json.loads(state_path.expanduser().resolve().read_text(encoding="utf-8"))
        entries = [
            item for item in state.get("promotions", [])
            if item.get("promote")
            and (through_round is None or int(item.get("through_round", -1)) == through_round)
        ]
        if not entries:
            raise ValueError("requested promoted entry is missing from state")
        entry = entries[-1]
        if entry.get("candidate_sha256") != source_sha256:
            raise ValueError("promotion checkpoint hash does not match source checkpoint")
        manifest["promotion"] = {
            "through_round": int(entry["through_round"]),
            "score": float(entry["score"]),
            "standard_error": float(entry["standard_error"]),
            "lower_bound": float(entry["lower_bound"]),
            "required_lower_bound": float(entry["required_lower_bound"]),
            "promotion_policy": str(entry["promotion_policy"]),
            "promotion_reason": str(entry["promotion_reason"]),
            "baseline_sha256": str(entry["baseline_sha256"]),
            "candidate_sha256": str(entry["candidate_sha256"]),
            "gate_config": json_value(entry["gate_config"]),
        }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text(model_card(manifest), encoding="utf-8")

    checksum_paths = sorted(
        path for path in output_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--state")
    parser.add_argument("--through-round", type=int)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    manifest = export_model(
        Path(arguments.checkpoint),
        Path(arguments.output_dir),
        overwrite=arguments.overwrite,
        state_path=Path(arguments.state) if arguments.state else None,
        through_round=arguments.through_round,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
