from types import SimpleNamespace


class thinking_models(SimpleNamespace):
    nobody = "nobody"
    everyone = "everyone"


MODEL_NAME_TO_LETTER_MAPPING = {
    "Qwen/Qwen3-14B": "H",
    "Qwen/Qwen3-4B": "M",
    "Qwen/Qwen3-0.6B": "L",
}
