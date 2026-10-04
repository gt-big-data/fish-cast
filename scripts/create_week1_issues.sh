#!/usr/bin/env bash
# Superseded by scripts/setup_github.py, which creates every semester task
# (not only week 1) and the project board. Kept so old instructions still work.
set -euo pipefail
exec python3 "$(dirname "$0")/setup_github.py" "$@"
