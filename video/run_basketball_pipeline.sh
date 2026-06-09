#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/yangxp/Capstone"
VIDEO_DIR="${ROOT_DIR}/video_data"
OUT_DIR="${ROOT_DIR}/data/basketball_output"
MODEL_NAME="${MODEL_NAME:-n}"
TRACKER_NAME="${TRACKER_NAME:-botsort_reid}"
RUN_REFINE="${RUN_REFINE:-1}"
REFINE_MODE="${REFINE_MODE:-vector}"
REFINE_CLIP_INDEX="${REFINE_CLIP_INDEX:-0}"
REFINE_NUM_FRAMES="${REFINE_NUM_FRAMES:-4}"
FORCE_RERUN="${FORCE_RERUN:-0}"

VIDEO_1="${VIDEO_DIR}/basketball_1.mp4"
VIDEO_2="${VIDEO_DIR}/basketball_2.mp4"

mkdir -p "${OUT_DIR}"

run_one_video() {
  local video_path="$1"
  local base_name
  base_name="$(basename "${video_path}" .mp4)"

  echo "=================================================="
  echo "Processing: ${video_path}"
  echo "Output dir: ${OUT_DIR}"
  echo "Model: ${MODEL_NAME} | Tracker: ${TRACKER_NAME} | Run refine: ${RUN_REFINE}"

  local events_json="${OUT_DIR}/${base_name}_events.json"
  local clips_json="${OUT_DIR}/${base_name}_clips.json"
  local vector_json="${OUT_DIR}/${base_name}_events_vector_flat.json"

  if [[ "${FORCE_RERUN}" != "1" && -f "${events_json}" && -f "${clips_json}" ]]; then
    echo "Skip video stage (existing outputs found): ${base_name}"
  else
    conda run -n capstone python -m video.factory.coordinator video \
      "${video_path}" \
      --out-dir "${OUT_DIR}" \
      --model "${MODEL_NAME}" \
      --tracker "${TRACKER_NAME}" \
      --conf 0.25 \
      --iou 0.25
  fi

  if [[ "${RUN_REFINE}" == "1" ]]; then
    if [[ "${FORCE_RERUN}" != "1" && -f "${vector_json}" ]]; then
      echo "Skip refine stage (existing output found): ${base_name}"
    else
      conda run -n capstone python -m video.factory.coordinator refine \
        --events "${events_json}" \
        --clips "${clips_json}" \
        --mode "${REFINE_MODE}" \
        --clip-index "${REFINE_CLIP_INDEX}" \
        --num-frames "${REFINE_NUM_FRAMES}" \
        --model qwen3-vl-flash \
        --temperature 0.1
    fi
  fi

  echo "Done: ${base_name}"
  echo "  events: ${events_json}"
  echo "  clips:  ${clips_json}"
  if [[ "${RUN_REFINE}" == "1" ]]; then
    echo "  vector: ${vector_json}"
  fi
}

if [[ ! -f "${VIDEO_1}" ]]; then
  echo "Missing input video: ${VIDEO_1}" >&2
  exit 1
fi

if [[ ! -f "${VIDEO_2}" ]]; then
  echo "Missing input video: ${VIDEO_2}" >&2
  exit 1
fi

run_one_video "${VIDEO_1}"
run_one_video "${VIDEO_2}"

echo "=================================================="
echo "All done."
echo "Final outputs are under: ${OUT_DIR}"
