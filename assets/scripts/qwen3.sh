set -euo pipefail

ITEMS=(
"Qwen/Qwen3-0.6B --gpus 1"
"Qwen/Qwen3-4B --gpus 2"
"Qwen/Qwen3-14B --gpus 5"
)


for ITEM in "${ITEMS[@]}"; do
    echo "Running: $ITEM"
    uv run orch start neumann vllm --llm $ITEM &
done

