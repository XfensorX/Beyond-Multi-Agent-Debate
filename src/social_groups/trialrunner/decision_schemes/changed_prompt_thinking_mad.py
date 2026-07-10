from __future__ import annotations

from copy import deepcopy

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from social_groups.general.utils.standard_library import BaseModelWithExtraFields
from social_groups.trialrunner.config import BackendInfo, DebateAgent, get_llm
from social_groups.trialrunner.decision_schemes.base import (
    DecisionScheme,
    ExampleInput,
    ExampleOutput,
    HistoryMessage,
)
from social_groups.trialrunner.experiment.main_registry import register_decision_scheme
from social_groups.trialrunner.utils.llm_calls import strip_out_thinking_process
from social_groups.trialrunner.utils.phoenix import phoenix_log_span


class ChangedPromptThinkingMadConfiguration(BaseModelWithExtraFields):
    debate_agents: list[DebateAgent]
    number_of_rounds: int
    openly_thinking_models: set[BackendInfo]
    system_message: str
    human_messsage_before_other_answers: str
    human_messsage_after_other_answers: str
    use_few_shot_prompting: bool


@register_decision_scheme("changed_prompt_thinking_mad")
class ChangedPromptThinkingMad(DecisionScheme[ChangedPromptThinkingMadConfiguration]):
    def run_example(self, example_input: ExampleInput) -> ExampleOutput:
        llms, agent_can_think_openly = self._build_agents()
        no_of_agents = len(llms)

        history: list[HistoryMessage] = []
        total_used_input_tokens = 0
        total_used_output_tokens = 0

        def call_llm(messages: list[BaseMessage], agent_id: int) -> str:
            nonlocal total_used_input_tokens, total_used_output_tokens

            ai_msg = llms[agent_id].invoke(messages)
            history.append(
                HistoryMessage(
                    input_context=messages,
                    answer=ai_msg.content,
                    agent_id=agent_id,
                )
            )

            total_used_input_tokens += ai_msg.usage_metadata["input_tokens"]
            total_used_output_tokens += ai_msg.usage_metadata["output_tokens"]

            return ai_msg.content

        last_agent_answers: list[str] = []
        answers_at_beginning: list[str] = []
        answers_at_end: list[str] = []

        for round_number in range(1, self.config.number_of_rounds + 1):
            messages = self._build_round_messages(
                example_input=example_input,
                last_agent_answers=last_agent_answers,
                agent_can_think_openly=agent_can_think_openly,
                no_of_agents=no_of_agents,
            )

            with phoenix_log_span("One Round Over."):
                last_agent_answers = [
                    call_llm(agent_messages, agent_id)
                    for agent_id, agent_messages in messages.items()
                ]

            if round_number == 1:
                answers_at_beginning = deepcopy(last_agent_answers)

            if round_number == self.config.number_of_rounds:
                answers_at_end = deepcopy(last_agent_answers)

        return ExampleOutput(
            used_input_tokens=total_used_input_tokens,
            used_output_tokens=total_used_output_tokens,
            history=history,
            answers_at_beginning=answers_at_beginning,
            answers_at_end=answers_at_end,
        )

    def _build_agents(self) -> tuple[list[BaseChatModel], list[bool]]:
        llms: list[BaseChatModel] = []
        agent_can_think_openly: list[bool] = []

        for agent in self.config.debate_agents:
            for _ in range(agent.number_of_agents):
                llms.append(get_llm(agent.params, agent.backend))
                agent_can_think_openly.append(
                    agent.backend in self.config.openly_thinking_models
                )

        return llms, agent_can_think_openly

    def _build_round_messages(
        self,
        example_input: ExampleInput,
        last_agent_answers: list[str],
        agent_can_think_openly: list[bool],
        no_of_agents: int,
    ) -> dict[int, list[BaseMessage]]:
        question = self._question_with_examples(example_input)

        if not last_agent_answers:
            return {
                agent_id: [
                    SystemMessage(self.config.system_message),
                    HumanMessage(question),
                ]
                for agent_id in range(no_of_agents)
            }

        combined_answers = self._combine_answers_for_next_round(
            last_agent_answers=last_agent_answers,
            agent_can_think_openly=agent_can_think_openly,
        )

        debate_prompt = (
            self.config.human_messsage_before_other_answers
            + combined_answers
            + self.config.human_messsage_after_other_answers
        )

        return {
            agent_id: [
                SystemMessage(self.config.system_message),
                HumanMessage(question),
                AIMessage(last_agent_answers[agent_id]),
                HumanMessage(debate_prompt),
            ]
            for agent_id in range(no_of_agents)
        }

    def _question_with_examples(self, example_input: ExampleInput) -> str:
        if self.config.use_few_shot_prompting:
            return example_input.presolved_questions + "\n\n" + example_input.question
        else:
            return example_input.question

    @staticmethod
    def _combine_answers_for_next_round(
        last_agent_answers: list[str],
        agent_can_think_openly: list[bool],
    ) -> str:
        return "\n\n".join(
            answer if can_think_openly else "\n".join(answer.splitlines()[-2:])
            for can_think_openly, answer in zip(
                agent_can_think_openly, last_agent_answers
            )
        )
