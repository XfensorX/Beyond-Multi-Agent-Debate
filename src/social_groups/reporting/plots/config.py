from pyfonts import load_google_font

TEXT_FONT = dict(
    family=load_google_font("Libre Baskerville", weight="bold").get_name(),
    size=12,
    color="rgb(34,32,32)",
    shadow="none",
)

PAPER_COLORSCALE = [
    [0.0, "#B5524A"],  # red
    # [0.5, "#EFE9DD"],  # paper
    [1.0, "#6FAF8A"],  # green
]


PAPER_BG = "#EFE9DD"


MODEL_NAME_TO_LETTER_MAPPING = {
    "Qwen/Qwen3-14B": "H",
    "Qwen/Qwen3-4B": "M",
    "Qwen/Qwen3-0.6B": "L",
}
