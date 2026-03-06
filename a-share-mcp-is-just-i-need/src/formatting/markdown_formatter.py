"""
Markdown formatting utilities for A-Share MCP Server.
"""
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Configuration
# Common number of trading days per year. Max rows to display in Markdown output
MAX_MARKDOWN_ROWS = 250


def _df_to_markdown_fallback(df: pd.DataFrame) -> str:
    """
    Lightweight DataFrame -> Markdown table conversion.
    Used when optional dependencies (e.g. `tabulate`) are unavailable.
    """
    # Convert values to strings and normalize newlines/pipes to keep table shape
    df_str = df.copy()
    for col in df_str.columns:
        df_str[col] = (
            df_str[col]
            .map(lambda v: "" if v is None else str(v))
            .str.replace("\n", " ", regex=False)
            .str.replace("|", "\\|", regex=False)
        )

    headers = [str(c) for c in df_str.columns.tolist()]
    rows = df_str.values.tolist()
    col_widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def _fmt_row(items: list[str]) -> str:
        return "| " + " | ".join(items[i].ljust(col_widths[i]) for i in range(len(items))) + " |"

    header_line = _fmt_row(headers)
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    body_lines = [_fmt_row([r[i] for i in range(len(headers))]) for r in rows]
    return "\n".join([header_line, sep_line, *body_lines])


def format_df_to_markdown(df: pd.DataFrame, max_rows: int = None) -> str:
    """Formats a Pandas DataFrame to a Markdown string with row truncation.

    Args:
        df: The DataFrame to format
        max_rows: Maximum rows to include in output. Defaults to MAX_MARKDOWN_ROWS if None.

    Returns:
        A markdown formatted string representation of the DataFrame
    """
    if df.empty:
        logger.warning("Attempted to format an empty DataFrame to Markdown.")
        return "(No data available to display)"

    # Default max_rows to the configured limit if not provided
    if max_rows is None:
        max_rows = MAX_MARKDOWN_ROWS
        logger.debug(f"max_rows defaulted to MAX_MARKDOWN_ROWS: {max_rows}")

    original_rows = df.shape[0]  # Only need original_rows now
    truncated = False
    truncation_notes = []

    # Determine the actual number of rows to display, capped by max_rows
    rows_to_show = min(original_rows, max_rows)

    # Always apply the row limit
    df_display = df.head(rows_to_show)

    # Check if actual row truncation occurred (only if original_rows > rows_to_show)
    if original_rows > rows_to_show:
        truncation_notes.append(
            f"rows truncated to the limit of {rows_to_show} (from {original_rows})")
        truncated = True

    try:
        markdown_table = df_display.to_markdown(index=False)
    except ImportError as e:
        # pandas.DataFrame.to_markdown requires optional dependency `tabulate`
        logger.warning(
            f"Optional dependency missing while converting DataFrame to Markdown: {e}. Falling back to a basic Markdown table."
        )
        markdown_table = _df_to_markdown_fallback(df_display)
    except Exception as e:
        logger.error(f"Error converting DataFrame to Markdown: {e}", exc_info=True)
        # As a last resort, show a readable text table instead of failing the whole workflow
        try:
            return "```text\n" + df_display.to_string(index=False) + "\n```"
        except Exception:
            return "Error: Could not format data into Markdown table."

    if truncated:
        # Note: 'truncated' is now only True if rows were truncated
        notes = "; ".join(truncation_notes)
        logger.debug(
            f"Markdown table generated with truncation notes: {notes}")
        return f"Note: Data truncated ({notes}).\n\n{markdown_table}"
    else:
        logger.debug("Markdown table generated without truncation.")
        return markdown_table
