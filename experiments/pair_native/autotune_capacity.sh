#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! "${PYTHON_BIN}" -c 'import torch, pyarrow' >/dev/null 2>&1; then
  for candidate in \
    "${PWD}/.venv/bin/python" \
    "${HOME}/miniconda3/envs/train/bin/python" \
    "${HOME}/anaconda3/envs/train/bin/python"; do
    if [[ -x "${candidate}" ]] && "${candidate}" -c 'import torch, pyarrow' >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi
if ! "${PYTHON_BIN}" -c 'import torch, pyarrow' >/dev/null 2>&1; then
  echo "未找到同时安装 torch 和 pyarrow 的 Python；请激活训练环境或设置 PYTHON_BIN。" >&2
  exit 2
fi
exec "${PYTHON_BIN}" -u -m experiments.pair_native.autotune_capacity "$@"
