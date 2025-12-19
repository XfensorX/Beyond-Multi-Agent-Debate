import os

import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from data_connectors.mmlu_pro import (
    ExperimentInput,
    ExperimentOutput,
    HistoryMessage,
)


# TODO: move this into a global configuration
dotenv.load_dotenv("../.env")

USED_ENDPOINT = "L3S-TGI"
# USED_ENDPOINT = "LMStudio"


ALL_LLMs = {
    "interweb": ChatOpenAI(
        model="gemma3:1b",
        base_url="https://interweb.l3s.uni-hannover.de/v1",
        api_key=os.getenv("INTERWEB_API_KEY"),
    ),
    "LMStudio": ChatOpenAI(
        model="qwen2.5-0.5b-instruct", base_url="http://127.0.0.1:1234/v1", api_key=""
    ),
    "L3S-TGI": ChatOpenAI(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        base_url="http://localhost:8000/v1",
        api_key="",
    ),
}


def single_model_baseline(input: ExperimentInput):
    used_options = {"max_tokens": 1024, "temperature": 0.0, "top_p": 0.0001}
    llm = ALL_LLMs[USED_ENDPOINT]
    messages = [
        SystemMessage(
            "You are an knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`."
        ),
        HumanMessage(input.example_questions + "\n\n" + input.question),
    ]
    ai_msg = llm.invoke(
        messages, **used_options
    )  # FIXME: different from original use temperature 0.0 and top_k 1

    return ExperimentOutput(
        number_of_agents=1,
        used_input_tokens=ai_msg.usage_metadata["input_tokens"],
        used_output_tokens=ai_msg.usage_metadata["output_tokens"],
        used_rounds=1,
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
