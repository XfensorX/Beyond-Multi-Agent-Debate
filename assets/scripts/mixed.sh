set -euo pipefail

ITEMS=(
#"mistralai/Ministral-3-3B-Reasoning-2512 --gpus 1"
#"mistralai/Ministral-3-8B-Reasoning-2512 --gpus 1"
"mistralai/Ministral-3-14B-Reasoning-2512 --gpus 1"
#"Qwen/Qwen3-0.6B --gpus 1"
#"Qwen/Qwen3-4B --gpus 1"
"Qwen/Qwen3-14B --gpus 1"
#"Qwen/Qwen3.5-0.8B --gpus 1"
#"Qwen/Qwen3.5-4B --gpus 1"
"Qwen/Qwen3.5-9B --gpus 1"
)

for ITEM in "${ITEMS[@]}"; do
    echo "Running: $ITEM"
    uv run orch start neumann vllm --llm $ITEM &
done

