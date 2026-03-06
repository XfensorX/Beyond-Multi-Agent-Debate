from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from social_groups.trialrunner.config import get_llm
from social_groups.trialrunner.decision_schemes.base import (
    DecisionScheme,
    ExampleInput,
    ExampleOutput,
    HistoryMessage,
)
from social_groups.trialrunner.decision_schemes.single_agent import (
    SingleAgentConfiguration,
)
from social_groups.trialrunner.experiment.main_registry import register_decision_scheme
from social_groups.trialrunner.utils.llm_calls import retrieve_single_answer_info
from social_groups.trialrunner.utils.tool_calls import (
    InvalidToolCallException,
    NoToolCallsException,
    parse_tool_call_arguments,
)


class SingleAgentToolConfiguration(SingleAgentConfiguration):
    retries_on_invalid_tool_call: int


@register_decision_scheme("single-agent-tool")
class SingleAgentToolBaseline(DecisionScheme[SingleAgentToolConfiguration]):
    def run_example(self, example_input: ExampleInput) -> ExampleOutput:
        messages = [
            SystemMessage(
                "You are an knowledge expert, you are supposed to answer "
                "the multi-choice question to derive your final answer. "
                "Think about the solution first, then submit your answer using the given tool."
                "You have to Submit your final answer using the provided tool."
            ),
            HumanMessage(
                example_input.presolved_questions + "\n\n" + example_input.question
            ),
        ]

        @tool
        def submit_answer(answer: str):
            """
            You MUST use this tool.
            Submit your final answer to the question.
            :param answer: The Answer that is correct in the form "(X)"
            """
            pass

        total_input_tokens = 0
        total_output_tokens = 0
        history = []

        for _ in range(self.config.retries_on_invalid_tool_call):
            ai_msg = (
                get_llm(self.config.llm, self.config.backend)
                .bind_tools([submit_answer], tool_choice=submit_answer.name)
                .invoke(messages)
            )
            answer_info = retrieve_single_answer_info(ai_msg)
            total_input_tokens += answer_info.input_tokens
            total_output_tokens += answer_info.output_tokens

            history.append(
                HistoryMessage(
                    input_context=messages, answer=answer_info.response, agent_id=0
                )
            )
            try:
                given_answer = parse_tool_call_arguments(ai_msg)["answer"]
                return ExampleOutput(
                    used_input_tokens=total_input_tokens,
                    used_output_tokens=total_output_tokens,
                    final_answer=given_answer,
                    history=history,
                )

            except (NoToolCallsException, InvalidToolCallException, KeyError):
                continue

        raise InvalidToolCallException(
            f"Even After {self.config.retries_on_invalid_tool_call} retries."
        )
