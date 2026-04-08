from copy import deepcopy

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

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


class ThinkingMadConfiguration(BaseModel):
    debate_agents: list[DebateAgent]
    number_of_rounds: int
    openly_thinking_models: set[BackendInfo]


@register_decision_scheme("thinking-mad")
class ThinkingMad(DecisionScheme[ThinkingMadConfiguration]):
    def run_example(self, example_input: ExampleInput) -> ExampleOutput:
        llms = [
            get_llm(agent.params, agent.backend)
            for agent in self.config.debate_agents
            for _ in range(agent.number_of_agents)
        ]

        agent_allowed_to_think_openly = [
            (agent.backend in self.config.openly_thinking_models)
            for agent in self.config.debate_agents
            for _ in range(agent.number_of_agents)
        ]

        no_of_agents = len(llms)

        history = []

        total_used_input_tokens = 0
        total_used_output_tokens = 0

        def call_llm(messages, agent_id):
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

        last_agent_answers = []
        answers_at_beginning = []
        answers_at_end = []

        for r in range(1, self.config.number_of_rounds + 1):
            if not last_agent_answers:
                messages = {
                    a: [
                        SystemMessage(
                            "Can you answer the following question as accurately as possible? Explain your answer, putting the answer in the form (X) at the end of your response"
                        ),
                        HumanMessage(
                            example_input.presolved_questions
                            + "\n\n"
                            + example_input.question
                        ),
                    ]
                    for a in range(no_of_agents)
                }

            else:
                combined_answers = "\n\n".join(
                    (
                        response
                        if can_think_openly
                        else strip_out_thinking_process(response)
                    )
                    for can_think_openly, response in zip(
                        agent_allowed_to_think_openly, last_agent_answers
                    )
                )

                messages = {
                    a: [
                        SystemMessage(
                            "Can you answer the following question as accurately as possible? Explain your answer, putting the answer in the form (X) at the end of your response"
                        ),
                        HumanMessage(
                            example_input.presolved_questions
                            + "\n\n"
                            + example_input.question
                        ),
                        AIMessage(last_agent_answers[a]),
                        HumanMessage(
                            f"These are the solutions to the problem from other agents: {combined_answers} Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents. Put your answer in the form (X) at the end of your response."
                        ),
                    ]
                    for a in range(no_of_agents)
                }
            with phoenix_log_span("One Round Over."):
                last_agent_answers = [
                    call_llm(messages[a], a) for a in range(no_of_agents)
                ]

            if r == 1:
                answers_at_beginning = deepcopy(last_agent_answers)
            elif r == self.config.number_of_rounds:
                answers_at_end = deepcopy(last_agent_answers)

        return ExampleOutput(
            used_input_tokens=total_used_input_tokens,
            used_output_tokens=total_used_output_tokens,
            history=history,
            answers_at_beginning=answers_at_beginning,
            answers_at_end=answers_at_end,
        )
