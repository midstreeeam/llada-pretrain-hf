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
DEFAULT_CONFIG="${REPO_DIR}/training_config/tllada_52m_dl.yaml"
if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG_FILE="$1"
  shift
else
  CONFIG_FILE="${DEFAULT_CONFIG}"
fi
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
