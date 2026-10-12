"""Bind a legacy TensorRT engine to the checkpoint used to build it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.pair_native.pair_replay import sha256_file


def default_manifest_path(engine: Path) -> Path:
    return engine.with_suffix(engine.suffix + ".manifest.json")


def write_manifest(
    checkpoint: Path,
    engine: Path,
    output: Path,
    onnx: Path | None = None,
    metadata: dict | None = None,
) -> dict:
    for path in (checkpoint, engine):
        if not path.is_file():
            raise FileNotFoundError(path)
    if onnx is not None and not onnx.is_file():
        raise FileNotFoundError(onnx)
    payload = {
        "schema_version": 1,
        "kind": "legacy-c6transnet-tensorrt",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "engine": str(engine.resolve()),
        "engine_sha256": sha256_file(engine),
        "onnx": str(onnx.resolve()) if onnx is not None else None,
        "onnx_sha256": sha256_file(onnx) if onnx is not None else None,
        "metadata": dict(metadata or {}),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return payload


def validate_manifest(manifest_path: Path, checkpoint: Path, engine: Path) -> dict:
    for path in (manifest_path, checkpoint, engine):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError("unsupported legacy engine manifest schema")
    if payload.get("kind") != "legacy-c6transnet-tensorrt":
        raise RuntimeError("unexpected legacy engine manifest kind")
    checkpoint_sha256 = sha256_file(checkpoint)
    engine_sha256 = sha256_file(engine)
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("legacy checkpoint does not match engine manifest")
    if payload.get("engine_sha256") != engine_sha256:
        raise RuntimeError("legacy TensorRT engine does not match engine manifest")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--onnx", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--batch-size", type=int, default=64)
    arguments = parser.parse_args()
    checkpoint = Path(arguments.checkpoint).expanduser().resolve()
    engine = Path(arguments.engine).expanduser().resolve()
    onnx = Path(arguments.onnx).expanduser().resolve() if arguments.onnx else None
    output = (
        Path(arguments.output).expanduser().resolve()
        if arguments.output
        else default_manifest_path(engine)
    )
    payload = write_manifest(
        checkpoint,
        engine,
        output,
        onnx=onnx,
        metadata={
            "architecture": "C6TransNet-17plane",
            "precision": arguments.precision,
            "max_batch_size": arguments.batch_size,
            "fused_selfplay": True,
        },
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
