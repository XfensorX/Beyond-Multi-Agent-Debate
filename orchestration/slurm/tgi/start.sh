#!/usr/bin/env bash

set -euo pipefail

source "scripts/environment/.env.pascal" # FIXME: could make this adaptable

: "${MODEL_ID}"
LOG_DIR=$PROJECT_DIR/results/slurm-logs

LOGIN="pascal.l3s" # FIXME: could make this adaptable
SBATCH_FILE="${PROJECT_DIR}/scripts/slurm/tgi/tgi.job"

JOBID=$(ssh "$LOGIN" "mkdir -p ${LOG_DIR} && cd ${PROJECT_DIR} && source scripts/load_env.sh && export MODEL_ID=${MODEL_ID} && cd $LOG_DIR && sbatch --parsable $SBATCH_FILE")
echo "Submitted: $JOBID"
echo "Output in ${LOGIN}:${LOG_DIR}"

NODE=""
until [ -n "$NODE" ] && [ "$NODE" != "(null)" ]; do
  NODE=$(ssh "$LOGIN" "squeue -j $JOBID -h -o %N | head -n1" || true)
  sleep 1
done
echo "TGI running on: $NODE"


