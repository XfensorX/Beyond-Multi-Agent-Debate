import json
import re
from json import JSONDecodeError
from typing import Any

import bs4
from langchain_core.messages import AIMessage

from social_groups.config import TOOL_CALL_TAGS


class InvalidToolCallException(Exception):
    pass


class NoToolCallsException(Exception):
    pass


def parse_tool_call_arguments(ai_msg: AIMessage) -> dict[str, str]:
    """Parse tool call arguments from AIMessage, supporting both LangChain tool_calls and raw Mistral/vLLM output."""
    if ai_msg.tool_calls:
        try:
            args = ai_msg.tool_calls[0]["args"]
            if isinstance(args, str):
                return json.loads(args)
            return args

        except (KeyError, IndexError, TypeError) as e:
            raise InvalidToolCallException() from e

    # Fallback: No tool_calls → try to extract manually from content
    if not isinstance(ai_msg.content, str):
        raise NoToolCallsException()

    tool_calls = retrieve_tool_call(ai_msg.content)
    if not tool_calls:
        raise NoToolCallsException()

    try:
        return tool_calls[0]["arguments"]
    except KeyError as e:
        raise InvalidToolCallException() from e
    except IndexError as e:
        raise NoToolCallsException() from e


def retrieve_tool_call(response: str) -> list[dict[str, Any]]:
    """
    Extract tool calls from raw model output.
    Supports:
      - Your original TOOL_CALL_TAGS (via BeautifulSoup)
      - Mistral/vLLM [TOOL_CALLS] format (e.g. [TOOL_CALLS]propose_solution{...})
    """
    if not response or not isinstance(response, str):
        return []

    response = fix_latex_backslashes(response)
    response = fix_wired_whitespace(response)

    tool_calls = []

    mistral_matches = extract_mistral_tool_calls(response)
    tool_calls.extend(mistral_matches)

    if not tool_calls:
        bs_tool_calls = extract_via_tags(response)
        tool_calls.extend(bs_tool_calls)

    return tool_calls


def extract_mistral_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract tool calls in the format: [TOOL_CALLS]tool_name{json arguments}"""
    # This regex captures: toolname followed by a JSON object (allowing newlines)
    pattern = r"\[TOOL_CALLS\]\s*(\w+)\s*(\{.*?\})\s*(?=\[TOOL_CALLS\]|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    results = []
    for tool_name, args_str in matches:
        try:
            args_str = args_str.strip()
            # Sometimes models put extra text after the JSON → take only until last }
            if "}" in args_str:
                args_str = args_str[: args_str.rfind("}") + 1]

            arguments = json.loads(args_str)
            results.append({"name": tool_name, "arguments": arguments})
        except (JSONDecodeError, TypeError):
            try:
                # Remove any text before the first {
                cleaned = re.search(r"(\{.*\})", args_str, re.DOTALL)
                if cleaned:
                    arguments = json.loads(cleaned.group(1))
                    results.append({"name": tool_name, "arguments": arguments})
            except (JSONDecodeError, TypeError):
                continue  # skip bad ones

    return results


def extract_via_tags(response: str) -> list[dict[str, Any]]:
    bs = bs4.BeautifulSoup(response, "html.parser")
    try:
        return [
            json.loads(x.text.strip())
            for pattern in TOOL_CALL_TAGS
            for x in bs.find_all(pattern)
        ]
    except JSONDecodeError as e:
        raise InvalidToolCallException() from e


def fix_latex_backslashes(text: str) -> str:
    def replacer(match):
        esc_char = match.group(1)
        if esc_char in r'"\\/bfnrtu':
            return match.group(0)
        return "\\" + match.group(0)

    return re.sub(r'\\([^"\\/bfnrtu])', replacer, text)


def fix_wired_whitespace(text: str) -> str:
    return text.translate(
        str.maketrans(
            {
                "\u00a0": " ",
                "\u200b": "",
                "\u200c": "",
                "\u200d": "",
                "\ufeff": "",
                "\u202f": " ",
                "\u205f": " ",
            }
        )
    )
