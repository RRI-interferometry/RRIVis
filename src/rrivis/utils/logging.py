"""Logging utilities for RRIvis.

Provides consistent logging configuration with rich console output.
"""

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Custom theme for RRIvis
RRIVIS_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "highlight": "bold magenta",
    "dim": "dim white",
})

# Global console instance
console = Console(theme=RRIVIS_THEME)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    rich_tracebacks: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration for RRIvis with rich console output.

    Args:
        level: Logging level (default: logging.INFO)
        log_file: Optional file path to write logs to
        format_string: Optional custom format string (for file handler)
        rich_tracebacks: Enable rich tracebacks for exceptions

    Returns:
        Configured logger instance
    """
    # Get root logger for rrivis
    logger = logging.getLogger("rrivis")
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Rich console handler with beautiful formatting
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=rich_tracebacks,
        tracebacks_show_locals=False,
        markup=True,
    )
    rich_handler.setLevel(level)
    logger.addHandler(rich_handler)

    # File handler (optional) - uses plain format for log files
    if log_file:
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        file_formatter = logging.Formatter(format_string)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Module name (e.g., "rrivis.core.visibility")

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Default logger instance
logger = get_logger("rrivis")


# =============================================================================
# Rich Console Utilities
# =============================================================================

def print_header(title: str, subtitle: Optional[str] = None) -> None:
    """Print a styled header panel."""
    from rich.panel import Panel
    from rich.text import Text

    content = Text(title, style="bold cyan")
    if subtitle:
        content.append(f"\n{subtitle}", style="dim")

    console.print(Panel(content, border_style="cyan", padding=(0, 2)))


def print_success(message: str) -> None:
    """Print a success message with checkmark."""
    console.print(f"[success]✓[/success] {message}", highlight=False)


def print_error(message: str) -> None:
    """Print an error message with X mark."""
    console.print(f"[error]✗[/error] {message}", highlight=False)


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[warning]⚠[/warning] {message}", highlight=False)


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[info]ℹ[/info] {message}", highlight=False)


def print_table(title: str, data: dict, title_style: str = "bold cyan") -> None:
    """Print a formatted table from a dictionary."""
    from rich.table import Table

    table = Table(title=title, title_style=title_style, show_header=False)
    table.add_column("Key", style="dim")
    table.add_column("Value", style="white")

    for key, value in data.items():
        table.add_row(str(key), str(value))

    console.print(table)


def get_progress(**kwargs):
    """
    Get a rich Progress instance for tracking long operations.

    Usage:
        with get_progress() as progress:
            task = progress.add_task("Processing...", total=100)
            for i in range(100):
                progress.update(task, advance=1)

    Returns:
        Progress context manager
    """
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
    )

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        **kwargs,
    )


def status(message: str):
    """
    Get a rich Status context manager for showing a spinner.

    Usage:
        with status("Loading data..."):
            load_data()

    Returns:
        Status context manager
    """
    return console.status(message, spinner="dots")
