"""``python -m ybook`` entry point."""

from __future__ import annotations

from .cli import app


def main() -> None:
    """Run the Typer application."""

    app()


if __name__ == "__main__":
    main()
