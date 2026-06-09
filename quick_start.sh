#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yangxp/Capstone"
FASTAPI_DIR="${PROJECT_ROOT}/fastapi"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
WORKERS="${WORKERS:-1}"
OPEN_BROWSER="${OPEN_BROWSER:-1}"

if [[ -f "${HOME}/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate capstone
elif [[ -x "${HOME}/anaconda3/bin/conda" ]]; then
  eval "$("${HOME}/anaconda3/bin/conda" shell.bash hook)"
  conda activate capstone
fi

echo "Starting Capstone FastAPI UI..."
echo "URL: http://${HOST}:${PORT}/"

if [[ "${OPEN_BROWSER}" == "1" ]]; then
  (
    sleep 2
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open "http://${HOST}:${PORT}/" >/dev/null 2>&1 || true
    fi
  ) &
fi

exec env HOST="${HOST}" PORT="${PORT}" WORKERS="${WORKERS}" "${FASTAPI_DIR}/run_uvicorn.sh"
