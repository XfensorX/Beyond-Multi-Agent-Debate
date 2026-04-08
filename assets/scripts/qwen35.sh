set -euo pipefail

ITEMS=(
"Qwen/Qwen3.5-0.8B --gpus 1"
"Qwen/Qwen3.5-4B --gpus 1"
"Qwen/Qwen3.5-9B --gpus 1"
)


for ITEM in "${ITEMS[@]}"; do
    echo "Running: $ITEM"
    uv run orch start neumann vllm --llm $ITEM &
done

