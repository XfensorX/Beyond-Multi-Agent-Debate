set -euo pipefail

ITEMS=(
# ----- Finished:
#'final/diversity_param_sweep_mad3_ministral3'
#'final/diversity_param_sweep_mad2_qwen35'
#'final/diversity_param_sweep_mad2_ministral3'
#'final/diversity_param_sweep_mad2_qwen3'
#'final/diversity_param_sweep_mad3_qwen3'
#'final/diversity_param_sweep_mad3_qwen35'

#'final/thinking_mad3_qwen3'
#'final/thinking_mad2_qwen35'
#'final/thinking_mad2_qwen3'
#'final/thinking_mad2_ministral3'
#'final/thinking_mad3_ministral3'
#'final/thinking_mad3_qwen35'

# ------ Not Loaded:


# ------ Running:

'final_gemma/single_model_baseline'
'final_gemma/no_discussion_voting_base'
'final_gemma/multi_agent_debate'
# -------- To Do:




#'tribal/tribal_council_standard'


#'final/different_families_mad2_mixedMQ'
#'final/different_families_mad3_mixedMQ'
#
#'final/encouraging_divergent_thinking_qwen3'
#'final/encouraging_divergent_thinking_qwen35'
#'final/encouraging_divergent_thinking_ministral3'


)

INTERVAL=20

TIME=0
for ITEM in "${ITEMS[@]}"; do
    printf '[T+%-3d] Running: %s.\n' $TIME $ITEM
    TIME=$((TIME + INTERVAL))
done

for ITEM in "${ITEMS[@]}"; do
    uv run orch start neumann experiment -m -e "$ITEM" &
    sleep $INTERVAL
done


# ------- Not Sorted:

#'final/different_families_mad2_L'
#'final/different_families_mad3_L'
#'final/different_families_mad2_M'
#'final/different_families_mad3_M'
#'final/different_families_mad2_H'
#'final/different_families_mad3_H'




# --------------------------- OLD ----------------------------

#'final/small_mad1_qwen3'
#'final/small_mad2_qwen3'
#'final/small_mad3_qwen3'
#'final/small_mad4_qwen3'

#'final/small_mad1_ministral3'
#'final/small_mad1_qwen35'
#'final/small_mad2_ministral3'
#'final/small_mad2_qwen35'
#'final/small_mad3_qwen35'
#'final/small_mad3_ministral3'
#'final/small_mad4_qwen35'
#'final/small_mad4_ministral3'


#'final/changed_prompt_mad3_ministral3'
#'final/changed_prompt_mad3_qwen35'
#'final/changed_prompt_mad2_ministral3'
#'final/changed_prompt_mad2_qwen35'
#'final/changed_prompt_mad2_qwen3'
#'final/changed_prompt_mad3_qwen3'

# --------------- Done --------------
#'final/changed_order_mad3_ministral3'
#'final/changed_order_mad3_qwen3'
#'final/changed_order_mad3_qwen35'
#'final/changed_order_mad2_qwen35'
#'final/changed_order_mad2_qwen3'
#'final/changed_order_mad2_ministral3'


#'final/baseline_qwen35'
#'final/baseline_structured_output_qwen35'
#'final/baseline_tool_qwen35'

#'final/baseline'
#'final/baseline_structured_output'
#'final/baseline_tool'

#'final/baseline_tool_ministral3'
#'final/baseline_ministral3'
#'final/baseline_structured_output_ministral3'



#'final/no_discussion_voting_ministral3'
#'final/no_discussion_voting_qwen3'
#'final/no_discussion_voting_qwen35'