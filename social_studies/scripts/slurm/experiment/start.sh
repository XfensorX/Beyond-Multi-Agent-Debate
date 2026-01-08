#!/usr/bin/env bash

set -euo pipefail

: "${EXPERIMENT_NAME:=$@}"
: "${TGI_BASE_URL}"

source "scripts/environment/.env.pascal" # FIXME: could make this adaptable
LOGIN="pascal.l3s" # FIXME: could make this adaptable
REMOTE_DIR="${PROJECT_DIR}/results/slurm-logs"
SBATCH_FILE="${PROJECT_DIR}/scripts/slurm/experiment/experiment.job"

JOBID=$(ssh "$LOGIN" "cd ${PROJECT_DIR} && source scripts/load_env.sh && export TGI_BASE_URL=${TGI_BASE_URL} && export EXPERIMENT_NAME=$EXPERIMENT_NAME && cd $REMOTE_DIR && sbatch --parsable $SBATCH_FILE")

echo "Submitted: $JOBID"
echo "Output in ${LOGIN}:${REMOTE_DIR}"
