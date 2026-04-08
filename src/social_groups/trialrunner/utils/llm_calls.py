import re

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


THINKING_TAGS = {
    "think",
    "thinking",
    "reasoning",
    "step",
    "steps",
    "thought",
    "thoughts",
}


def strip_out_thinking_process(response: str):
    """
    Very fast version using regex.
    Removes everything between any combination of the listed thinking tags.
    """
    if not THINKING_TAGS:
        return response

    etags = [re.escape(tag) for tag in THINKING_TAGS]

    # Build pattern like: <think>.*?</think>|<thinking>.*?</thinking>|...
    tags_pattern = "|".join(f"<{tag}>.*?</{tag}>" for tag in etags)

    # (?s) = dot matches newline, *? = non-greedy
    pattern = re.compile(f"(?s){tags_pattern}")

    # Remove all matches repeatedly until none left (handles nesting & multiple types)
    prev_len = -1
    while len(response) != prev_len:
        prev_len = len(response)
        response = pattern.sub("", response)

    # repeat for only closing tags that are left.
    closing_only_pattern = re.compile(
        r"(?s)" + "|".join(f"^.*?</{tag}>" for tag in etags)
    )

    prev_len = -1
    while len(response) != prev_len:
        prev_len = len(response)
        response = closing_only_pattern.sub("", response)

    return response.strip()
