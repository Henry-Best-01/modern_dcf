"""Console script for modern_dcf."""

import typer
from rich.console import Console

from modern_dcf import utils

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for modern_dcf."""
    console.print("Replace this message by putting your code into "
               "modern_dcf.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
