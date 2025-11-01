#!/usr/bin/env bash
set -euo pipefail

python -m src.trainer_prompt \
  --config config.yaml \
  --dataset rest
