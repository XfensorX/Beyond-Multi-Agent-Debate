from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field, ValidationError


class SingleAnswerInfo(BaseModel):
    response: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class InvalidResponseException(Exception):
    pass


def retrieve_single_answer_info(ai_msg: AIMessage) -> SingleAnswerInfo:
    """Checks that the response has a single string as content and the metadata is correct.
    Raises InvalidResponseException if not.
    """

    if not isinstance(ai_msg.content, str):
        raise InvalidResponseException("Response content is not a string.")

    if ai_msg.usage_metadata is None:
        raise InvalidResponseException("Usage metadata is not set.")

    try:
        return SingleAnswerInfo(
            response=ai_msg.content,
            input_tokens=ai_msg.usage_metadata["input_tokens"],
            output_tokens=ai_msg.usage_metadata["output_tokens"],
        )

    except ValidationError as e:
        raise InvalidResponseException(str(e)) from e

    except KeyError as e:
        raise InvalidResponseException(str(e)) from e
