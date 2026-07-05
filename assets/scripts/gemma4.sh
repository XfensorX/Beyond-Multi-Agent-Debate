set -euo pipefail

ITEMS=(
"google/gemma-4-E2B-it --gpus 3"
"google/gemma-4-E4B-it --gpus 3"
"google/gemma-4-31B-it --gpus 2"
)

for ITEM in "${ITEMS[@]}"; do
    echo "Running: $ITEM"
    uv run orch start neumann vllm --llm $ITEM &
done

