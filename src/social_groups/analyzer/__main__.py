from rich import print
from typer import Typer

app = Typer(no_args_is_help=True)


@app.command("parse", help="Parse results to produce parquet files.")
def produce_parquet():
    print("Producing Parquet files ...")


@app.command("show", help="Show nothing.")
def show_results():
    print("There is nothing to show.")


def main():
    app()


if __name__ == "__main__":
    main()
