"""
Error handling infrastructure for Noeta DSL.

This module provides production-quality error messages with:
- Rich context (line numbers, source code snippets, arrows)
- Helpful hints and suggestions
- Color-coded output for terminals
- Plain text for Jupyter notebooks
"""

import sys
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
from difflib import SequenceMatcher


class ErrorCategory(Enum):
    """Categories of errors in Noeta compilation."""
    LEXER = "Lexical Error"
    SYNTAX = "Syntax Error"
    SEMANTIC = "Semantic Error"
    RUNTIME = "Runtime Error"
    TYPE = "Type Error"


@dataclass
class ErrorContext:
    """Context information for an error location."""
    line: int
    column: int
    length: int = 1
    source_line: str = ""

    def __post_init__(self):
        """Ensure length is at least 1."""
        if self.length < 1:
            self.length = 1


class NoetaError(Exception):
    """
    Unified error class for Noeta DSL with rich formatting.

    Attributes:
        message: Primary error message
        category: Error category (lexer, syntax, semantic, etc.)
        context: Optional error context with line/column info
        hint: Optional helpful hint
        suggestion: Optional code suggestion for fix
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYNTAX,
        context: Optional[ErrorContext] = None,
        hint: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        self.message = message
        self.category = category
        self.context = context
        self.hint = hint
        self.suggestion = suggestion

        # Format the complete error message
        formatted_message = ErrorFormatter.format(self)
        super().__init__(formatted_message)


class ErrorFormatter:
    """Formats NoetaError instances with rich context and color."""

    # ANSI color codes
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    @classmethod
    def _supports_color(cls) -> bool:
        """Check if terminal supports ANSI colors."""
        # Check if output is a terminal and not redirected
        return hasattr(sys.stderr, 'isatty') and sys.stderr.isatty()

    @classmethod
    def _colorize(cls, text: str, color: str) -> str:
        """Apply color to text if terminal supports it."""
        if cls._supports_color():
            return f"{color}{text}{cls.RESET}"
        return text

    @classmethod
    def format(cls, error: NoetaError) -> str:
        """
        Format a NoetaError with rich context.

        Output format:
            Syntax Error at line 2, column 14:
                2 | select sales 123 as result
                               ^^^ Expected column name, got number

            Hint: Column names must be identifiers, not numbers
            Did you mean: select sales with price as result?
        """
        lines = []

        # Header: "Syntax Error at line 2, column 14:"
        if error.context:
            header = f"{error.category.value} at line {error.context.line}, column {error.context.column}:"
            lines.append(cls._colorize(header, cls.RED + cls.BOLD))
        else:
            header = f"{error.category.value}:"
            lines.append(cls._colorize(header, cls.RED + cls.BOLD))

        # Source code context with arrow
        if error.context and error.context.source_line:
            lines.append(cls._format_source_context(error.context))

        # Main error message
        lines.append(f"    {cls._colorize(error.message, cls.RED)}")

        # Hint (if provided)
        if error.hint:
            lines.append("")
            lines.append(cls._colorize(f"Hint: {error.hint}", cls.YELLOW))

        # Suggestion (if provided)
        if error.suggestion:
            lines.append(cls._colorize(f"Did you mean: {error.suggestion}", cls.GREEN))

        return "\n".join(lines)

    @classmethod
    def _format_source_context(cls, context: ErrorContext) -> str:
        """
        Format source code line with arrow pointing to error.

        Example:
            2 | select sales 123 as result
                           ^^^ Expected column name
        """
        lines = []

        # Line number and source
        line_num_str = f"{context.line:4d}"
        source_line = f"    {line_num_str} | {context.source_line}"
        lines.append(source_line)

        # Arrow pointing to error location
        # Calculate spaces: 4 (indent) + len(line_num_str) + 3 (" | ") + column - 1
        spaces = 4 + len(line_num_str) + 3 + context.column - 1
        arrow = " " * spaces + cls._colorize("^" * context.length, cls.CYAN)
        lines.append(arrow)

        return "\n".join(lines)


# Utility functions for error suggestions

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.

    Used for "did you mean?" suggestions.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def suggest_similar(
    attempted: str,
    available: List[str],
    max_suggestions: int = 3,
    max_distance: int = 3
) -> List[str]:
    """
    Suggest similar names from available options.

    Args:
        attempted: The name that was attempted
        available: List of available names
        max_suggestions: Maximum number of suggestions to return
        max_distance: Maximum Levenshtein distance to consider

    Returns:
        List of suggested names, sorted by similarity
    """
    if not available:
        return []

    # Calculate distances
    distances = []
    for name in available:
        dist = levenshtein_distance(attempted.lower(), name.lower())
        if dist <= max_distance:
            distances.append((dist, name))

    # Sort by distance and return top suggestions
    distances.sort(key=lambda x: x[0])
    return [name for _, name in distances[:max_suggestions]]


def get_token_type_description(token_type_name: str) -> str:
    """
    Get human-friendly description for a token type.

    Args:
        token_type_name: Name of the token type (e.g., "IDENTIFIER", "STRING_LITERAL")

    Returns:
        Human-friendly description
    """
    descriptions = {
        'IDENTIFIER': 'dataset or column name',
        'STRING_LITERAL': 'string value',
        'NUMERIC_LITERAL': 'number',
        'INTEGER': 'integer number',
        'FLOAT': 'decimal number',
        'BOOLEAN': 'true or false',
        'NONE': 'None value',
        'LPAREN': 'opening parenthesis (',
        'RPAREN': 'closing parenthesis )',
        'LBRACE': 'opening brace {',
        'RBRACE': 'closing brace }',
        'LBRACKET': 'opening bracket [',
        'RBRACKET': 'closing bracket ]',
        'COMMA': 'comma',
        'DOT': 'dot',
        'EQUALS': 'equals sign =',
        'COLON': 'colon :',
        'SEMICOLON': 'semicolon',
        'AS': '"as" keyword',
        'WITH': '"with" keyword',
        'WHERE': '"where" keyword',
        'COLUMN': '"column" keyword',
        'COLUMNS': '"columns" keyword',
        'EOF': 'end of file',
    }

    return descriptions.get(token_type_name, token_type_name.lower().replace('_', ' '))


def get_operation_hint(operation: str) -> Optional[str]:
    """
    Get a helpful hint for a specific operation.

    Args:
        operation: Operation name (e.g., "select", "filter")

    Returns:
        Helpful hint or None if no hint available
    """
    hints = {
        'select': 'Use "select <dataset> with <col1>, <col2> as <alias>" to select columns',
        'filter': 'Use "filter <dataset> where <condition> as <alias>" to filter rows',
        'load': 'Use "load <file_path> as <alias>" or "load csv <file_path> as <alias>"',
        'save': 'Use "save <dataset> <file_path>" or "save csv <dataset> <file_path>"',
        'groupby': 'Use "groupby <dataset> by <column> as <alias>" to group data',
        'join': 'Use "join <left> with <right> on <condition> as <alias>" to join datasets',
        'merge': 'Use "merge <left> with <right> on <key> as <alias>" to merge datasets',
        'head': 'Use "head <dataset> with n=<number> as <alias>" to get first rows',
        'tail': 'Use "tail <dataset> with n=<number> as <alias>" to get last rows',
        'describe': 'Use "describe <dataset>" to show statistical summary',
        'info': 'Use "info <dataset>" to show dataset information',
        'sort': 'Use "sort <dataset> by <column> as <alias>" to sort data',
        'rename': 'Use "rename <dataset> column <old> to <new> as <alias>" to rename columns',
        'drop': 'Use "drop <dataset> columns <col1>, <col2> as <alias>" to remove columns',
        'fillna': 'Use "fillna <dataset> with <value> as <alias>" to fill missing values',
        'dropna': 'Use "dropna <dataset> as <alias>" to remove rows with missing values',
        'astype': 'Use "astype <dataset> column <col> type=<type> as <alias>" to convert types',
        'unique': 'Use "unique <dataset> column <column>" to get unique values',
        'value_counts': 'Use "value_counts <dataset> column <column>" to count value frequencies',
    }

    return hints.get(operation.lower())


def format_did_you_mean(suggestions: List[str], context: str = "operation") -> Optional[str]:
    """
    Format "did you mean?" message with suggestions.

    Args:
        suggestions: List of suggested names
        context: What kind of thing we're suggesting (e.g., "operation", "column", "dataset")

    Returns:
        Formatted suggestion message or None if no suggestions
    """
    if not suggestions:
        return None

    if len(suggestions) == 1:
        return f"{suggestions[0]}"
    elif len(suggestions) == 2:
        return f"{suggestions[0]} or {suggestions[1]}"
    else:
        return f"{', '.join(suggestions[:-1])}, or {suggestions[-1]}"


def create_syntax_error(
    message: str,
    line: int,
    column: int,
    source_line: str = "",
    length: int = 1,
    hint: Optional[str] = None,
    suggestion: Optional[str] = None
) -> NoetaError:
    """
    Convenience function to create a syntax error.

    Args:
        message: Error message
        line: Line number
        column: Column number
        source_line: Source code line
        length: Length of error span
        hint: Optional hint
        suggestion: Optional suggestion

    Returns:
        NoetaError instance
    """
    context = ErrorContext(line, column, length, source_line)
    return NoetaError(message, ErrorCategory.SYNTAX, context, hint, suggestion)


def create_semantic_error(
    message: str,
    line: int = 0,
    column: int = 0,
    source_line: str = "",
    length: int = 1,
    hint: Optional[str] = None,
    suggestion: Optional[str] = None
) -> NoetaError:
    """
    Convenience function to create a semantic error.

    Args:
        message: Error message
        line: Line number (0 if unknown)
        column: Column number (0 if unknown)
        source_line: Source code line
        length: Length of error span
        hint: Optional hint
        suggestion: Optional suggestion

    Returns:
        NoetaError instance
    """
    context = ErrorContext(line, column, length, source_line) if line > 0 else None
    return NoetaError(message, ErrorCategory.SEMANTIC, context, hint, suggestion)


def create_type_error(
    message: str,
    line: int,
    column: int,
    source_line: str = "",
    length: int = 1,
    hint: Optional[str] = None,
    suggestion: Optional[str] = None
) -> NoetaError:
    """
    Convenience function to create a type error.

    Args:
        message: Error message
        line: Line number
        column: Column number
        source_line: Source code line
        length: Length of error span
        hint: Optional hint
        suggestion: Optional suggestion

    Returns:
        NoetaError instance
    """
    context = ErrorContext(line, column, length, source_line)
    return NoetaError(message, ErrorCategory.TYPE, context, hint, suggestion)
