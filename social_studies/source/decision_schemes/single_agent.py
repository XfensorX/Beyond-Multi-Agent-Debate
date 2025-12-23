from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from config import get_llm, Backend
from decision_schemes.base import (
    DecisionScheme,
    ExampleInput,
    ExampleOutput,
    HistoryMessage,
)
from experiment.main_registry import register_decision_scheme


class SingleAgentConfiguration(BaseModel):
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]

    backend: Backend
    model_name: str

    def create_llm_options(self):
        options = {}

        if self.max_tokens:
            options["max_tokens"] = self.max_tokens
        if self.temperature:
            options["temperature"] = self.temperature
        if self.top_p:
            options["top_p"] = self.top_p

        return options


@register_decision_scheme("single-agent")
class SingleAgentBaseline(DecisionScheme[SingleAgentConfiguration]):
    def run_example(self, example_input: ExampleInput) -> ExampleOutput:
        used_options = self.config.create_llm_options()

        messages = [
            SystemMessage(
                "You are an knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`."
            ),
            HumanMessage(
                example_input.presolved_questions + "\n\n" + example_input.question
            ),
        ]

        ai_msg = get_llm(self.config.backend, model_name=self.config.model_name).invoke(
            messages, **used_options
        )

        return ExampleOutput(
            used_input_tokens=ai_msg.usage_metadata["input_tokens"],
            used_output_tokens=ai_msg.usage_metadata["output_tokens"],
            final_answer=ai_msg.content,
            history=[
                HistoryMessage(
                    input_context=messages,
                    answer=ai_msg.content,
                    model_name=ai_msg.response_metadata["model_name"],
                    agent_id=0,
                    options=used_options,
                )
            ],
        )
