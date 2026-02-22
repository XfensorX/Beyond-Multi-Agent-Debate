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
