#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-72B-Instruct}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PORT="${PORT:-23333}"
TP_SIZE="${TP_SIZE:-4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen72b}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

export CUDA_VISIBLE_DEVICES
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve \
  --model "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --disable-log-requests \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max_num_seqs "${MAX_NUM_SEQS}"
