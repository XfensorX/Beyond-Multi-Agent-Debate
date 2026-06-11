"""
This Decision Scheme shall be the same as https://github.com/Skytliang/Multi-Agents-Debate
This is not own work, it is adapted from the given GitHub page to fit the architecture of this project.
This file shall be available under GPL3.0 as the original source.
"""

import ast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from social_groups.general.utils.standard_library import BaseModelWithExtraFields
from social_groups.trialrunner.config import BackendInfo, LLMConfig, get_llm
from social_groups.trialrunner.decision_schemes.base import (
    DecisionScheme,
    ExampleInput,
    ExampleOutput,
)
from social_groups.trialrunner.experiment.main_registry import register_decision_scheme
from social_groups.trialrunner.utils.llm_calls import retrieve_single_answer_info
from social_groups.trialrunner.utils.phoenix import phoenix_log_span


class EncouragingAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.memory_lst: list[BaseMessage] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def add_system_message(self, meta_prompt: str):
        self.memory_lst.append(SystemMessage(meta_prompt))

    def add_human_message(self, event: str):
        self.memory_lst.append(HumanMessage(event))

    def add_ai_message(self, memory: str):
        self.memory_lst.append(AIMessage(memory))

    def ask(self) -> str:
        answer = retrieve_single_answer_info(self.llm.invoke(self.memory_lst))

        self.total_input_tokens += answer.input_tokens
        self.total_output_tokens += answer.output_tokens

        return answer.response


class DebatePrompts(BaseModel):
    debate_topic: str
    base_answer: str
    debate_answer: str
    player_meta_prompt: str
    moderator_meta_prompt: str
    affirmative_prompt: str
    negative_prompt: str
    moderator_prompt: str
    judge_prompt_last1: str
    judge_prompt_last2: str
    debate_prompt: str


class EncouragingDivergentThinkingConfiguration(BaseModelWithExtraFields):
    backend: BackendInfo
    llm: LLMConfig
    max_round: int
    use_thinking: bool
    prompts: DebatePrompts


round_dct = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
}


def init_prompt(prompt_config):
    def prompt_replace(key):
        prompt_config[key] = prompt_config[key].replace(
            "##debate_topic##", prompt_config["debate_topic"]
        )

    prompt_replace("player_meta_prompt")
    prompt_replace("moderator_meta_prompt")
    prompt_replace("affirmative_prompt")
    prompt_replace("judge_prompt_last2")


def evaluate_mod_answer(mod_ans):
    try:
        mod_ans_dict = ast.literal_eval(mod_ans)
        # Optional: verify it's actually a dict
        if not isinstance(mod_ans_dict, dict):
            mod_ans_dict = {"debate_answer": ""}
    except (ValueError, SyntaxError, TypeError):
        mod_ans_dict = {"debate_answer": ""}

    return mod_ans_dict


def init_agents(
    affirmative: EncouragingAgent,
    negative: EncouragingAgent,
    moderator: EncouragingAgent,
    debate_prompt_config,
):
    affirmative.add_system_message(debate_prompt_config["player_meta_prompt"])
    negative.add_system_message(debate_prompt_config["player_meta_prompt"])
    moderator.add_system_message(debate_prompt_config["moderator_meta_prompt"])

    affirmative.add_human_message(debate_prompt_config["affirmative_prompt"])
    aff_ans = affirmative.ask()
    affirmative.add_ai_message(aff_ans)
    debate_prompt_config["base_answer"] = aff_ans

    negative.add_human_message(
        debate_prompt_config["negative_prompt"].replace("##aff_ans##", aff_ans)
    )
    neg_ans = negative.ask()
    negative.add_ai_message(neg_ans)

    moderator.add_human_message(
        debate_prompt_config["moderator_prompt"]
        .replace("##aff_ans##", aff_ans)
        .replace("##neg_ans##", neg_ans)
        .replace("##round##", "first")
    )
    mod_ans = moderator.ask()
    moderator.add_ai_message(mod_ans)

    return evaluate_mod_answer(mod_ans), neg_ans, aff_ans


def run(
    mod_ans,
    neg_ans,
    aff_ans,
    affirmative: EncouragingAgent,
    negative: EncouragingAgent,
    moderator: EncouragingAgent,
    debate_prompt_config,
    max_round: int,
):
    for r in range(max_round - 1):
        if mod_ans["debate_answer"] != "":
            break
        else:
            with phoenix_log_span(f"Debate Round {r + 2}"):
                affirmative.add_human_message(
                    debate_prompt_config["debate_prompt"].replace(
                        "##oppo_ans##", neg_ans
                    )
                )
                aff_ans = affirmative.ask()
                affirmative.add_ai_message(aff_ans)

                negative.add_human_message(
                    debate_prompt_config["debate_prompt"].replace(
                        "##oppo_ans##", aff_ans
                    )
                )
                neg_ans = negative.ask()
                negative.add_ai_message(neg_ans)

                moderator.add_human_message(
                    debate_prompt_config["moderator_prompt"]
                    .replace("##aff_ans##", aff_ans)
                    .replace("##neg_ans##", neg_ans)
                    .replace("##round##", round_dct[r + 2])
                )
                mod_ans = moderator.ask()
                moderator.add_ai_message(mod_ans)
                mod_ans = evaluate_mod_answer(mod_ans)

    if mod_ans["debate_answer"] != "":
        debate_prompt_config.update(mod_ans)
        debate_prompt_config["success"] = True

    else:
        judge_player = EncouragingAgent(moderator.llm)
        aff_ans = affirmative.memory_lst[2].content
        neg_ans = negative.memory_lst[2].content

        judge_player.add_system_message(debate_prompt_config["moderator_meta_prompt"])

        # extract answer candidates
        judge_player.add_human_message(
            debate_prompt_config["judge_prompt_last1"]
            .replace("##aff_ans##", aff_ans)
            .replace("##neg_ans##", neg_ans)
        )
        ans = judge_player.ask()
        judge_player.add_ai_message(ans)

        # select one from the candidates
        judge_player.add_system_message(debate_prompt_config["judge_prompt_last2"])
        ans = judge_player.ask()
        judge_player.add_ai_message(ans)

        ans = evaluate_mod_answer(ans)
        if ans["debate_answer"] != "":
            debate_prompt_config["success"] = True

        debate_prompt_config.update(ans)
        return judge_player

    return None


@register_decision_scheme("encouraging-divergent-thinking")
class EncouragingDivergentThinking(
    DecisionScheme[EncouragingDivergentThinkingConfiguration]
):
    def run_example(self, example_input: ExampleInput) -> ExampleOutput:
        debate_prompt_config = self.config.prompts.model_dump()

        llm = get_llm(self.config.llm, self.config.backend, self.config.use_thinking)
        affirmative = EncouragingAgent(llm)
        negative = EncouragingAgent(llm)
        moderator = EncouragingAgent(llm)

        init_prompt(debate_prompt_config)
        mod_ans, neg_ans, aff_ans = init_agents(
            affirmative, negative, moderator, debate_prompt_config
        )

        judge = run(
            mod_ans,
            neg_ans,
            aff_ans,
            affirmative,
            negative,
            moderator,
            debate_prompt_config,
            self.config.max_round,
        )

        total_input_tokens = (
            negative.total_input_tokens
            + moderator.total_input_tokens
            + affirmative.total_input_tokens
        )
        total_output_tokens = (
            negative.total_output_tokens
            + moderator.total_output_tokens
            + affirmative.total_output_tokens
        )

        if judge:
            total_input_tokens += judge.total_input_tokens
            total_output_tokens += judge.total_output_tokens

        return ExampleOutput(
            used_input_tokens=total_input_tokens,
            used_output_tokens=total_output_tokens,
            final_answer=debate_prompt_config["debate_answer"],
            history=[  # TODO:
            ],
        )
