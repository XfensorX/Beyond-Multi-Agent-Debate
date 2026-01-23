#!/usr/bin/env bash

set -euo pipefail

source "scripts/environment/.env.pascal" # FIXME: could make this adaptable
LOGIN="pascal.l3s" # FIXME: could make this adaptable
REMOTE_DIR="${PROJECT_DIR}/results/slurm-logs"
SBATCH_FILE="${PROJECT_DIR}/scripts/slurm/phoenix/phoenix.job"

JOBID=$(ssh "$LOGIN" "cd ${PROJECT_DIR} && source scripts/load_env.sh && cd $REMOTE_DIR && sbatch --parsable $SBATCH_FILE")
echo "Submitted: $JOBID"
echo "Output in ${LOGIN}:${REMOTE_DIR}"

NODE=""
until [ -n "$NODE" ] && [ "$NODE" != "(null)" ]; do
  NODE=$(ssh "$LOGIN" "squeue -j $JOBID -h -o %N | head -n1" || true)
  sleep 1
done
echo "Phoenix running on: $NODE"
echo "Opening tunnel on http://localhost:$PHOENIX_PORT"


ssh -N -L localhost:${PHOENIX_PORT}:localhost:${PHOENIX_PORT} $NODE