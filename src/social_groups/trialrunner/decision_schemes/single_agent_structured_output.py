from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from social_groups.general.utils.standard_library import BaseModelWithExtraFields
from social_groups.trialrunner.config import (
    BackendInfo,
    LLMConfig,
    get_llm,
    get_structured_output_method,
)
from social_groups.trialrunner.decision_schemes.base import (
    DecisionScheme,
    ExampleInput,
    ExampleOutput,
    HistoryMessage,
)
from social_groups.trialrunner.experiment.main_registry import register_decision_scheme
from social_groups.trialrunner.utils.llm_calls import (
    InvalidResponseException,
    retrieve_single_answer_info,
)


class AnswerResponseFormat(BaseModel):
    response: str = Field(
        ..., description="The answer to the question in the form 'The answer is (X)'."
    )


class SingleAgentStructuredOutputConfiguration(BaseModelWithExtraFields):
    llm: LLMConfig
    backend: BackendInfo

    use_few_shot_prompting: bool
    use_thinking: bool | None = None


@register_decision_scheme("single-agent-structured-output")
class SingleAgentStructuredOutputBaseline(
    DecisionScheme[SingleAgentStructuredOutputConfiguration]
):
    def run_example(self, example_input: ExampleInput) -> ExampleOutput:
        messages = [
            SystemMessage(
                "You are an knowledge expert, you are supposed to answer the multi-choice "
                "question to derive your final answer as `The answer is ...` using the given schema only!"
            ),
            HumanMessage(
                (example_input.presolved_questions + "\n\n" + example_input.question)
                if self.config.use_few_shot_prompting
                else example_input.question
            ),
        ]

        output = (
            get_llm(self.config.llm, self.config.backend, self.config.use_thinking)
            .with_structured_output(
                AnswerResponseFormat,
                strict=True,
                include_raw=True,
                method=get_structured_output_method(self.config.backend.model_name),
            )
            .invoke(messages)
        )
        structured_output, ai_msg, error = (
            output["parsed"],
            output["raw"],
            output["parsing_error"],
        )

        if error:
            raise InvalidResponseException() from error

        if structured_output is None:
            raise InvalidResponseException("The structured output returned is None")

        answer_info = retrieve_single_answer_info(ai_msg)

        return ExampleOutput(
            used_input_tokens=answer_info.input_tokens,
            used_output_tokens=answer_info.output_tokens,
            final_answer=structured_output.response,
            history=[
                HistoryMessage(
                    input_context=messages, answer=answer_info.response, agent_id=0
                )
            ],
        )
