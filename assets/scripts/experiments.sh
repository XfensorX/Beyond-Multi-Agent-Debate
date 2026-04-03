set -euo pipefail

ITEMS=(
#'final/baseline'
#'final/baseline_tool'
#'final/baseline_structured_output'
'final/baseline_qwen35'
#'final/baseline_tool_qwen35'
#'final/baseline_structured_output_qwen35'
#'final/baseline_ministral3'
#'final/baseline_tool_ministral3'
#'final/baseline_structured_output_ministral3'
)


for ITEM in "${ITEMS[@]}"; do
    echo "Running: $ITEM"
    uv run orch start neumann experiment -m -e "$ITEM" &
done




