#!/usr/bin/env bash

# Prevent sourcing: run as an executable script so shell options don't leak.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "[error] Please run this script instead of sourcing it: bash run.sh" >&2
  return 1
fi

set -euo pipefail

# Determine repository root so the script is location-independent.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate local virtual environment if it exists.
if [[ -d "${REPO_DIR}/.venv" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_DIR}/.venv/bin/activate"
else
  echo "[warn] No virtual environment at ${REPO_DIR}/.venv; falling back to current interpreter."
fi

# ===== CONFIG SELECTION =====
# Require the YAML config path as the first argument.
if [[ $# -eq 0 || "$1" == --* ]]; then
  echo "Usage: bash run.sh <config.yaml> [extra args...]" >&2
  echo "Example: bash run.sh training_config/tllada_52m_phase2.yaml" >&2
  exit 1
fi
CONFIG_FILE="$1"
shift
CONFIG_FILE="$(cd "${REPO_DIR}" && realpath "${CONFIG_FILE}")"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "[error] Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

# Ensure the repo is importable when running helper scripts.
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Delegate to the Python launcher that understands config files.
python3 "${REPO_DIR}/training_launcher.py" \
  --config "${CONFIG_FILE}" \
  --repo-dir "${REPO_DIR}" \
  "$@"
