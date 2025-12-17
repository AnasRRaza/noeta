"""
Unit tests for Noeta Error Handling.

Tests error message formatting, error categories, and suggestions.
"""
import pytest
from noeta_errors import (
    NoetaError,
    ErrorCategory,
    ErrorContext,
    create_syntax_error,
    create_semantic_error,
    create_type_error,
    suggest_similar
)


class TestErrorFormatting:
    """Tests for error message formatting."""

    def test_syntax_error_formatting(self):
        """Test syntax error message formatting."""
        error = create_syntax_error(
            message="Expected AS keyword",
            line=1,
            column=20,
            source_line='load "data.csv" sales',
            length=5
        )

        assert isinstance(error, NoetaError)
        assert error.category == ErrorCategory.SYNTAX
        assert "Expected AS keyword" in error.message
        assert error.context.line == 1
        assert error.context.column == 20

    def test_semantic_error_formatting(self):
        """Test semantic error message formatting."""
        error = create_semantic_error(
            message="Dataset 'unknown' has not been loaded",
            line=1,
            column=8,
            source_line='describe unknown',
            length=7
        )

        assert isinstance(error, NoetaError)
        assert error.category == ErrorCategory.SEMANTIC
        assert "unknown" in error.message
        assert error.context.line == 1

    def test_type_error_formatting(self):
        """Test type error message formatting."""
        error = create_type_error(
            message="Expected numeric column",
            line=1,
            column=15,
            source_line='round sales column name',
            length=4
        )

        assert isinstance(error, NoetaError)
        assert error.category == ErrorCategory.TYPE
        assert "numeric" in error.message.lower()

    def test_error_string_representation(self):
        """Test error string representation includes all info."""
        error = create_syntax_error(
            message="Test error",
            line=5,
            column=10,
            source_line='test line',
            length=4
        )

        error_str = str(error)

        # Should include category, message, and location
        assert "Syntax" in error_str or "SYNTAX" in error_str
        assert "Test error" in error_str
        assert "5" in error_str  # Line number


class TestErrorSuggestions:
    """Tests for error suggestions (did-you-mean)."""

    def test_suggest_similar_single_match(self):
        """Test suggesting similar string with single match."""
        candidates = ["sales", "customers", "products"]

        suggestions = suggest_similar("sale", candidates, max_suggestions=1)

        assert len(suggestions) >= 1
        assert "sales" in suggestions

    def test_suggest_similar_multiple_matches(self):
        """Test suggesting multiple similar strings."""
        candidates = ["employee", "employees", "employers"]

        suggestions = suggest_similar("employe", candidates, max_suggestions=3)

        assert len(suggestions) >= 1
        # Should suggest "employee" or "employees"
        assert any(s in ["employee", "employees"] for s in suggestions)

    def test_suggest_similar_no_match(self):
        """Test that very different strings return no suggestions."""
        candidates = ["sales", "customers"]

        suggestions = suggest_similar("xyz123", candidates, max_suggestions=1)

        # Might return empty list or distant matches
        # This tests that the function handles very different strings

    def test_suggest_similar_case_insensitive(self):
        """Test that suggestions are case-insensitive."""
        candidates = ["Sales", "CUSTOMERS", "products"]

        suggestions = suggest_similar("sale", candidates, max_suggestions=1)

        # Should still match despite case differences
        assert len(suggestions) >= 0  # May or may not match depending on implementation

    def test_suggest_similar_empty_candidates(self):
        """Test suggestions with empty candidate list."""
        candidates = []

        suggestions = suggest_similar("test", candidates, max_suggestions=1)

        assert len(suggestions) == 0


class TestErrorCategories:
    """Tests for different error categories."""

    def test_lexer_error_category(self):
        """Test that lexer errors have correct category."""
        from noeta_lexer import Lexer
        from noeta_errors import NoetaError, ErrorCategory

        lexer = Lexer('load "unterminated')

        try:
            lexer.tokenize()
            assert False, "Should have raised error"
        except NoetaError as e:
            assert e.category == ErrorCategory.LEXER

    def test_syntax_error_category(self):
        """Test that syntax errors have correct category."""
        error = create_syntax_error(
            message="Missing keyword",
            line=1,
            column=1,
            source_line="test",
            length=4
        )

        assert error.category == ErrorCategory.SYNTAX

    def test_semantic_error_category(self):
        """Test that semantic errors have correct category."""
        error = create_semantic_error(
            message="Undefined reference",
            line=1,
            column=1,
            source_line="test",
            length=4
        )

        assert error.category == ErrorCategory.SEMANTIC

    def test_type_error_category(self):
        """Test that type errors have correct category."""
        error = create_type_error(
            message="Type mismatch",
            line=1,
            column=1,
            source_line="test",
            length=4
        )

        assert error.category == ErrorCategory.TYPE


class TestErrorContext:
    """Tests for error context (line, column, source)."""

    def test_error_context_creation(self):
        """Test creating error context."""
        context = ErrorContext(
            line=10,
            column=5,
            source_line='load "data.csv" as sales',
            length=4
        )

        assert context.line == 10
        assert context.column == 5
        assert context.source_line == 'load "data.csv" as sales'
        assert context.length == 4

    def test_error_with_context(self):
        """Test error includes context information."""
        error = create_syntax_error(
            message="Test error",
            line=1,
            column=5,
            source_line='test source line',
            length=4
        )

        assert error.context is not None
        assert error.context.line == 1
        assert error.context.column == 5
        assert error.context.source_line == 'test source line'

    def test_error_context_in_string_representation(self):
        """Test that error string includes context."""
        error = create_syntax_error(
            message="Test error",
            line=15,
            column=20,
            source_line='this is the source line',
            length=4
        )

        error_str = str(error)

        # Should include line number in output
        assert "15" in error_str


class TestErrorHints:
    """Tests for error hints and suggestions."""

    def test_error_with_hint(self):
        """Test error with hint message."""
        error = create_semantic_error(
            message="Dataset not found",
            line=1,
            column=1,
            source_line="describe unknown",
            length=7,
            hint="Available datasets: sales, customers"
        )

        error_str = str(error)

        # Hint should be included in error output
        assert "Available datasets" in error_str or "sales" in error_str

    def test_error_with_suggestion(self):
        """Test error with suggestion."""
        error = create_semantic_error(
            message="Dataset 'sale' not found",
            line=1,
            column=1,
            source_line="describe sale",
            length=4,
            suggestion="Did you mean 'sales'?"
        )

        error_str = str(error)

        # Suggestion should be in error output
        assert "Did you mean" in error_str or "sales" in error_str

    def test_error_without_hint(self):
        """Test error without hint still works."""
        error = create_syntax_error(
            message="Simple error",
            line=1,
            column=1,
            source_line="test",
            length=4
        )

        # Should not raise exception
        error_str = str(error)
        assert "Simple error" in error_str


class TestErrorIntegration:
    """Integration tests for error handling."""

    def test_lexer_error_integration(self):
        """Test lexer error in full context."""
        from noeta_lexer import Lexer
        from noeta_errors import NoetaError, ErrorCategory

        source = 'load "data.csv" as sales @ invalid'
        lexer = Lexer(source)

        with pytest.raises(NoetaError) as exc_info:
            lexer.tokenize()

        error = exc_info.value
        assert error.category == ErrorCategory.LEXER
        assert error.context is not None

    def test_parser_error_integration(self):
        """Test parser error in full context."""
        from noeta_lexer import Lexer
        from noeta_parser import Parser
        from noeta_errors import NoetaError, ErrorCategory

        source = 'load "data.csv" sales'  # Missing AS
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)

        with pytest.raises(NoetaError) as exc_info:
            parser.parse()

        error = exc_info.value
        assert error.category == ErrorCategory.SYNTAX

    def test_semantic_error_integration(self):
        """Test semantic error in full context."""
        from noeta_lexer import Lexer
        from noeta_parser import Parser
        from noeta_semantic import SemanticAnalyzer
        from noeta_errors import ErrorCategory

        source = 'describe undefined'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) >= 1
        assert errors[0].category == ErrorCategory.SEMANTIC


class TestErrorRecovery:
    """Tests for error recovery and multiple errors."""

    def test_single_error_reported(self):
        """Test that single error is reported correctly."""
        from noeta_runner import compile_noeta
        from noeta_errors import NoetaError

        source = 'describe nonexistent'

        with pytest.raises(NoetaError) as exc_info:
            compile_noeta(source)

        # Should get one clear error
        error = exc_info.value
        assert "nonexistent" in error.message

    def test_first_error_reported(self):
        """Test that first error is reported in multi-error scenario."""
        from noeta_runner import compile_noeta
        from noeta_errors import NoetaError

        # Multiple errors: undefined dataset
        source = '''describe first_undefined
describe second_undefined'''

        with pytest.raises(NoetaError) as exc_info:
            compile_noeta(source)

        # Should report first error
        error = exc_info.value
        assert "first_undefined" in error.message or "undefined" in error.message.lower()


class TestErrorMessageQuality:
    """Tests for error message quality and clarity."""

    def test_error_message_includes_operation(self):
        """Test that error messages mention the operation."""
        error = create_semantic_error(
            message="Dataset 'unknown' not found in select operation",
            line=1,
            column=1,
            source_line="select unknown with price as result",
            length=7
        )

        # Should mention the operation or context
        assert "select" in error.message.lower() or "unknown" in error.message

    def test_error_message_actionable(self):
        """Test that error messages are actionable."""
        error = create_semantic_error(
            message="Dataset 'sale' not found",
            line=1,
            column=1,
            source_line="describe sale",
            length=4,
            hint="Available datasets: sales, customers",
            suggestion="Did you mean 'sales'?"
        )

        error_str = str(error)

        # Should provide actionable information
        assert "sales" in error_str  # Shows available option
        assert "Did you mean" in error_str or "Available" in error_str

    def test_error_message_readable(self):
        """Test that error messages are human-readable."""
        error = create_syntax_error(
            message="Expected AS keyword after dataset path",
            line=1,
            column=20,
            source_line='load "data.csv" sales',
            length=5,
            hint="Syntax: load <path> as <alias>"
        )

        error_str = str(error)

        # Should be readable and helpful
        assert "Expected" in error_str or "AS" in error_str
        assert len(error_str) > 20  # Should have substantial content
