#!/usr/bin/env bash

set -euo pipefail

: "${EXPERIMENT_NAME:=$@}"

source "scripts/environment/.env.pascal" # FIXME: could make this adaptable
LOGIN="pascal.l3s" # FIXME: could make this adaptable
REMOTE_DIR="${PROJECT_DIR}/results/slurm-logs"
SBATCH_FILE="${PROJECT_DIR}/scripts/slurm/experiment/experiment.job"


JOBID=$(
  ssh "$LOGIN" "
    cd ${PROJECT_DIR}
    source scripts/load_env.sh
    export EXPERIMENT_NAME=${EXPERIMENT_NAME}

    cd ${REMOTE_DIR}
    alias squ='squeue -O jobid:8,name:32,state:16,timeused:10,reasonlist:18,tres-per-job:14,tres-alloc:0 --me'
    export TGI_HOST=\$(squ | awk '\$2==\"tgi\" && \$3==\"RUNNING\" {print \$5}')
    export PHOENIX_HOST=\$(squ | awk '\$2==\"phoenix\" && \$3==\"RUNNING\" {print \$5}')

    sbatch --parsable ${SBATCH_FILE}
  "
)

echo "Submitted: $JOBID"
echo "Output in ${LOGIN}:${REMOTE_DIR}"
