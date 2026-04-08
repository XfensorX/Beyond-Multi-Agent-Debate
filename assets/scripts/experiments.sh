set -euo pipefail

ITEMS=(
'final/changed_order_mad2'
'final/changed_order_mad2_ministral3'
'final/changed_order_mad2_qwen35'
'final/changed_order_mad3'
'final/changed_order_mad3_ministral3'
'final/changed_order_mad3_qwen35'
)

INTERVAL=10

TIME=0
for ITEM in "${ITEMS[@]}"; do
    printf '[T+%-3d] Running: %s.\n' $TIME $ITEM
    TIME=$((TIME + INTERVAL))
done

for ITEM in "${ITEMS[@]}"; do
    uv run orch start neumann experiment -m -e "$ITEM" &
    sleep $INTERVAL
done





# --------------------------- OLD ----------------------------

#'final/baseline'
#'final/baseline_tool'
#'final/baseline_structured_output'
#'final/baseline_qwen35'
#'final/baseline_tool_qwen35'
#'final/baseline_structured_output_qwen35'
#'final/baseline_ministral3'
#'final/baseline_tool_ministral3'
#'final/baseline_structured_output_ministral3'