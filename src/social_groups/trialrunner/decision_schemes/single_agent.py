from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from social_groups.trialrunner.config import BackendInfo, LLMConfig, get_llm
from social_groups.trialrunner.decision_schemes.base import (
    DecisionScheme,
    ExampleInput,
    ExampleOutput,
    HistoryMessage,
)
from social_groups.trialrunner.experiment.main_registry import register_decision_scheme
from social_groups.trialrunner.utils.llm_calls import retrieve_single_answer_info


class SingleAgentConfiguration(BaseModel):
    llm: LLMConfig
    backend: BackendInfo


@register_decision_scheme("single-agent")
class SingleAgentBaseline(DecisionScheme[SingleAgentConfiguration]):
    def run_example(self, example_input: ExampleInput) -> ExampleOutput:
        messages = [
            SystemMessage(
                "You are an knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`."
            ),
            HumanMessage(
                example_input.presolved_questions + "\n\n" + example_input.question
            ),
        ]

        ai_msg = get_llm(self.config.llm, self.config.backend).invoke(messages)

        answer_info = retrieve_single_answer_info(ai_msg)
        return ExampleOutput(
            used_input_tokens=answer_info.input_tokens,
            used_output_tokens=answer_info.output_tokens,
            final_answer=answer_info.response,
            history=[
                HistoryMessage(
                    input_context=messages, answer=answer_info.response, agent_id=0
                )
            ],
        )
