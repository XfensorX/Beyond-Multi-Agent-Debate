set -euo pipefail

ITEMS=(
"mistralai/Ministral-3-3B-Reasoning-2512 --gpus 1"
"mistralai/Ministral-3-8B-Reasoning-2512 --gpus 1"
"mistralai/Ministral-3-14B-Reasoning-2512 --gpus 1"

"mistralai/Ministral-3-3B-Instruct-2512 --gpus 1"
"mistralai/Ministral-3-8B-Instruct-2512 --gpus 1"
"mistralai/Ministral-3-14B-Instruct-2512 --gpus 1"
)


for ITEM in "${ITEMS[@]}"; do
    echo "Running: $ITEM"
    uv run orch start neumann vllm --llm $ITEM &
done

