#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
UI_HOST="${UI_HOST:-127.0.0.1}"
UI_PORT="${UI_PORT:-5173}"
OPEN_BROWSER="${OPEN_BROWSER:-0}"

cd "${PROJECT_ROOT}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ ! -d "frontend/node_modules" ]]; then
  echo "frontend/node_modules is missing."
  echo "Run: npm --prefix frontend install"
  exit 1
fi

cleanup() {
  if [[ -n "${API_PID:-}" ]]; then
    kill "${API_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${UI_PID:-}" ]]; then
    kill "${UI_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting Capstone API at http://${HOST}:${PORT}"
python -m uvicorn main:app --app-dir mevid_fastapi --host "${HOST}" --port "${PORT}" &
API_PID=$!

echo "Starting Capstone UI at http://${UI_HOST}:${UI_PORT}"
npm --prefix frontend run dev -- --host "${UI_HOST}" --port "${UI_PORT}" &
UI_PID=$!

if [[ "${OPEN_BROWSER}" == "1" ]]; then
  (
    sleep 3
    if command -v open >/dev/null 2>&1; then
      open "http://${UI_HOST}:${UI_PORT}/" >/dev/null 2>&1 || true
    elif command -v xdg-open >/dev/null 2>&1; then
      xdg-open "http://${UI_HOST}:${UI_PORT}/" >/dev/null 2>&1 || true
    fi
  ) &
fi

wait "${API_PID}" "${UI_PID}"
