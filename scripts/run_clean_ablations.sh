#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Clean staged runner for comparable ablation phases.
exec bash "$ROOT/scripts/run_staged_ablations.sh" "$@"
