from types import SimpleNamespace


class thinking_models(SimpleNamespace):
    nobody = "nobody"
    everyone = "everyone"


MODEL_NAME_TO_LETTER_MAPPING: dict[str, str] = {
    "Qwen/Qwen3-14B": "H",
    "Qwen/Qwen3-4B": "M",
    "Qwen/Qwen3-0.6B": "L",
    "mistralai/Ministral-3-14B-Reasoning-2512": "H",
    "mistralai/Ministral-3-8B-Reasoning-2512": "M",
    "mistralai/Ministral-3-3B-Reasoning-2512": "L",
    "Qwen/Qwen3.5-0.8B": "L",
    "Qwen/Qwen3.5-4B": "M",
    "Qwen/Qwen3.5-9B": "H",
}
