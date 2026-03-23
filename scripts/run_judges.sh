#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ "$#" -gt 0 ]; then
  folders=("$@")
else
  folders=(
    "eval_results_deepeyes/SpecEyes"
    "eval_results_thyme/SpecEyes"
    "eval_results_deepeyes/SpecReason"
    "eval_results_thyme/SpecReason"
    "eval_results_qwen3vl-2b-Instruct"
  )
fi

for folder in "${folders[@]}"; do
  if [ ! -d "$folder" ]; then
    echo "Skip missing folder: $folder"
    continue
  fi

  echo "Processing folder: $folder"
  python judge_code/judge_vstar.py --input_folder "$folder"
  python judge_code/judge_hr.py --input_folder "$folder"
  python judge_code/judge_pope.py --input_folder "$folder"
done
