import json
import re
from json import JSONDecodeError
from typing import Any

import bs4
from langchain_core.messages import (
    AIMessage,
)
from pydantic import ValidationError

from social_groups.config import TOOL_CALL_TAGS


class InvalidToolCallException(Exception):
    pass


class NoToolCallsException(Exception):
    pass


def parse_tool_call_arguments(ai_msg: AIMessage) -> dict[str, str]:
    if not ai_msg.tool_calls:
        try:
            tool_calls = retrieve_tool_call(ai_msg.content)

            if not tool_calls:
                raise NoToolCallsException()

            return tool_calls[0]["arguments"]

        except (KeyError, ValidationError):
            raise InvalidToolCallException()

    try:
        return ai_msg.tool_calls[0]["args"]
    except KeyError as e:
        raise InvalidToolCallException() from e


def retrieve_tool_call(response: str) -> list[dict[str, Any]]:
    """
    Very fast version using regex.
    """

    def fix_latex_backslashes(text: str) -> str:
        def replacer(match):
            esc_char = match.group(1)
            # Valid JSON escapes — keep them as-is
            if esc_char in r'"\\/bfnrtu':
                return match.group(0)
            # Invalid ones (most LaTeX cases) → make them literal \\
            return "\\" + match.group(0)  # turns \c → \\c

        # Replace \ followed by anything that's not a valid escape starter
        return re.sub(r'\\([^"\\/bfnrtu])', replacer, text)

    def fix_wired_whitespace(text: str) -> str:
        return text.translate(
            str.maketrans(
                {
                    "\u00a0": " ",  # non-breaking space
                    "\u200b": "",  # zero-width space → remove
                    "\u200c": "",  # zero-width non-joiner
                    "\u200d": "",  # zero-width joiner
                    "\ufeff": "",  # BOM / zero-width no-break space
                    "\u202f": " ",  # narrow no-break space
                    "\u205f": " ",  # medium mathematical space
                }
            )
        )

    response = fix_latex_backslashes(response)
    response = fix_wired_whitespace(response)
    bs = bs4.BeautifulSoup(response, "html.parser")
    try:
        return [
            (json.loads(x.text.strip()))
            for pattern in TOOL_CALL_TAGS
            for x in bs.find_all(pattern)
        ]
    except JSONDecodeError as e:
        raise InvalidToolCallException() from e
