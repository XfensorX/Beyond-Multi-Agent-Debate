from config import BackendInfo, LLMConfig, get_llm
from decision_schemes.base import (
    DecisionScheme,
    ExampleInput,
    ExampleOutput,
    HistoryMessage,
)
from experiment.main_registry import register_decision_scheme
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel


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

        return ExampleOutput(
            used_input_tokens=ai_msg.usage_metadata["input_tokens"],
            used_output_tokens=ai_msg.usage_metadata["output_tokens"],
            final_answer=ai_msg.content,
            history=[
                HistoryMessage(
                    input_context=messages,
                    answer=ai_msg.content,
                    agent_id=0,
                )
            ],
        )
