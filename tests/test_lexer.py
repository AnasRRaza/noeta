"""
Unit tests for Noeta Lexer.

Tests token recognition, position tracking, and error handling.
"""
import pytest
from noeta_lexer import Lexer, TokenType
from noeta_errors import NoetaError, ErrorCategory


class TestLexerBasicTokenization:
    """Basic lexer tokenization tests."""

    def test_tokenize_simple_load(self):
        """Test tokenizing a simple load statement."""
        lexer = Lexer('load "data.csv" as sales')
        tokens = lexer.tokenize()

        assert len(tokens) >= 4
        assert tokens[0].type == TokenType.LOAD
        assert tokens[1].type == TokenType.STRING_LITERAL
        assert tokens[1].value == "data.csv"
        assert tokens[2].type == TokenType.AS
        assert tokens[3].type == TokenType.IDENTIFIER
        assert tokens[3].value == "sales"

    def test_tokenize_select_with_columns(self):
        """Test tokenizing select with column list."""
        lexer = Lexer('select sales with price, quantity as subset')
        tokens = lexer.tokenize()

        assert tokens[0].type == TokenType.SELECT
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[1].value == "sales"
        assert tokens[2].type == TokenType.WITH
        assert tokens[3].type == TokenType.IDENTIFIER
        assert tokens[3].value == "price"
        assert tokens[4].type == TokenType.COMMA
        assert tokens[5].type == TokenType.IDENTIFIER
        assert tokens[5].value == "quantity"

    def test_tokenize_filter_where(self):
        """Test tokenizing filter with where clause."""
        lexer = Lexer('filter sales where price > 100 as expensive')
        tokens = lexer.tokenize()

        assert tokens[0].type == TokenType.FILTER
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[2].type == TokenType.WHERE
        # price > 100 gets tokenized
        assert tokens[6].type == TokenType.AS
        assert tokens[7].type == TokenType.IDENTIFIER
        assert tokens[7].value == "expensive"

    def test_tokenize_describe(self):
        """Test tokenizing describe statement."""
        lexer = Lexer('describe sales')
        tokens = lexer.tokenize()

        assert len(tokens) >= 2
        assert tokens[0].type == TokenType.DESCRIBE
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[1].value == "sales"

    def test_tokenize_groupby(self):
        """Test tokenizing groupby statement."""
        lexer = Lexer('groupby sales by product agg sum as totals')
        tokens = lexer.tokenize()

        assert tokens[0].type == TokenType.GROUPBY
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[2].type == TokenType.BY
        assert tokens[3].type == TokenType.IDENTIFIER
        assert tokens[4].type == TokenType.AGG


class TestLexerDataTypes:
    """Tests for different data type recognition."""

    def test_tokenize_string_double_quotes(self):
        """Test string with double quotes."""
        lexer = Lexer('load "data.csv" as d')
        tokens = lexer.tokenize()

        string_tokens = [t for t in tokens if t.type == TokenType.STRING_LITERAL]
        assert len(string_tokens) == 1
        assert string_tokens[0].value == "data.csv"

    def test_tokenize_string_single_quotes(self):
        """Test string with single quotes."""
        lexer = Lexer("load 'data.csv' as d")
        tokens = lexer.tokenize()

        string_tokens = [t for t in tokens if t.type == TokenType.STRING_LITERAL]
        assert len(string_tokens) == 1
        assert string_tokens[0].value == "data.csv"

    def test_tokenize_integer(self):
        """Test integer literal."""
        lexer = Lexer('head sales with n=10')
        tokens = lexer.tokenize()

        number_tokens = [t for t in tokens if t.type == TokenType.NUMERIC_LITERAL]
        assert len(number_tokens) == 1
        assert number_tokens[0].value == "10"

    def test_tokenize_float(self):
        """Test float literal."""
        lexer = Lexer('filter sales where price > 99.99 as result')
        tokens = lexer.tokenize()

        number_tokens = [t for t in tokens if t.type == TokenType.NUMERIC_LITERAL]
        assert len(number_tokens) == 1
        assert number_tokens[0].value == "99.99"

    def test_tokenize_boolean_true(self):
        """Test boolean True."""
        lexer = Lexer('fillna sales with value=0 inplace=True as result')
        tokens = lexer.tokenize()

        bool_tokens = [t for t in tokens if t.type == TokenType.BOOLEAN_LITERAL]
        assert len(bool_tokens) == 1
        assert bool_tokens[0].value == "True"

    def test_tokenize_boolean_false(self):
        """Test boolean False."""
        lexer = Lexer('fillna sales with value=0 inplace=False as result')
        tokens = lexer.tokenize()

        bool_tokens = [t for t in tokens if t.type == TokenType.BOOLEAN_LITERAL]
        assert len(bool_tokens) == 1
        assert bool_tokens[0].value == "False"

    def test_tokenize_none(self):
        """Test None literal."""
        lexer = Lexer('fillna sales with value=None as result')
        tokens = lexer.tokenize()

        none_tokens = [t for t in tokens if t.type == TokenType.NONE_LITERAL]
        assert len(none_tokens) == 1


class TestLexerOperators:
    """Tests for operator tokenization."""

    def test_tokenize_comparison_operators(self):
        """Test comparison operators."""
        lexer = Lexer('filter d where x > 5 and y < 10 and z >= 3 and w <= 7')
        tokens = lexer.tokenize()

        operators = [t for t in tokens if t.type in (
            TokenType.GREATER_THAN,
            TokenType.LESS_THAN,
            TokenType.GREATER_EQUAL,
            TokenType.LESS_EQUAL
        )]
        assert len(operators) >= 4

    def test_tokenize_equality_operators(self):
        """Test equality operators."""
        lexer = Lexer('filter d where x == 5 and y != 10')
        tokens = lexer.tokenize()

        eq_tokens = [t for t in tokens if t.type in (TokenType.EQUALS_EQUALS, TokenType.NOT_EQUALS)]
        assert len(eq_tokens) >= 2

    def test_tokenize_logical_operators(self):
        """Test logical operators."""
        lexer = Lexer('filter d where x > 5 and y < 10 or z == 3')
        tokens = lexer.tokenize()

        logical_tokens = [t for t in tokens if t.type in (TokenType.AND, TokenType.OR)]
        assert len(logical_tokens) >= 2

    def test_tokenize_arithmetic_operators(self):
        """Test arithmetic operators."""
        lexer = Lexer('filter d where x + y - z * w / 2')
        tokens = lexer.tokenize()

        arithmetic_tokens = [t for t in tokens if t.type in (
            TokenType.PLUS,
            TokenType.MINUS,
            TokenType.MULTIPLY,
            TokenType.DIVIDE
        )]
        assert len(arithmetic_tokens) >= 4


class TestLexerKeywords:
    """Tests for keyword recognition."""

    def test_all_load_keywords(self):
        """Test load-related keywords."""
        keywords = ['load', 'load_csv', 'load_json', 'load_excel']

        for keyword in keywords:
            lexer = Lexer(f'{keyword} "file" as data')
            tokens = lexer.tokenize()
            assert tokens[0].type.name.startswith('LOAD')

    def test_selection_keywords(self):
        """Test selection keywords."""
        keywords = ['select', 'head', 'tail', 'sample']

        for keyword in keywords:
            lexer = Lexer(f'{keyword} data as result')
            tokens = lexer.tokenize()
            # Should tokenize without error
            assert len(tokens) >= 3

    def test_transformation_keywords(self):
        """Test transformation keywords."""
        keywords = ['filter', 'sort', 'groupby', 'join', 'merge']

        for keyword in keywords:
            lexer = Lexer(f'{keyword} data')
            tokens = lexer.tokenize()
            assert len(tokens) >= 1

    def test_case_insensitive_keywords(self):
        """Test that keywords are case-insensitive."""
        variations = ['LOAD', 'Load', 'load', 'LoAd']

        for variation in variations:
            lexer = Lexer(f'{variation} "file" as data')
            tokens = lexer.tokenize()
            assert tokens[0].type == TokenType.LOAD


class TestLexerPositionTracking:
    """Tests for line and column tracking."""

    def test_single_line_positions(self):
        """Test position tracking on single line."""
        lexer = Lexer('load "data.csv" as sales')
        tokens = lexer.tokenize()

        # All tokens should be on line 1
        for token in tokens:
            assert token.line == 1

    def test_multi_line_positions(self):
        """Test position tracking across multiple lines."""
        source = '''load "data.csv" as sales
select sales with price as subset'''

        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # First tokens should be on line 1
        line1_tokens = [t for t in tokens if t.line == 1]
        assert len(line1_tokens) > 0
        assert line1_tokens[0].type == TokenType.LOAD

        # Later tokens should be on line 2
        line2_tokens = [t for t in tokens if t.line == 2]
        assert len(line2_tokens) > 0
        assert line2_tokens[0].type == TokenType.SELECT

    def test_column_tracking(self):
        """Test column position tracking."""
        lexer = Lexer('load "data.csv" as sales')
        tokens = lexer.tokenize()

        # load starts at column 0 (or 1 depending on implementation)
        assert tokens[0].column >= 0

        # Later tokens should have higher column numbers
        for i in range(1, len(tokens)):
            # Each token should have a valid column position
            assert tokens[i].column >= 0


class TestLexerComplexExpressions:
    """Tests for complex expression tokenization."""

    def test_multiple_statements(self):
        """Test tokenizing multiple statements."""
        source = '''load "data.csv" as sales
select sales with price, quantity as subset
describe subset'''

        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # Should have tokens for all three statements
        load_tokens = [t for t in tokens if t.type == TokenType.LOAD]
        select_tokens = [t for t in tokens if t.type == TokenType.SELECT]
        describe_tokens = [t for t in tokens if t.type == TokenType.DESCRIBE]

        assert len(load_tokens) == 1
        assert len(select_tokens) == 1
        assert len(describe_tokens) == 1

    def test_complex_where_clause(self):
        """Test complex where clause tokenization."""
        lexer = Lexer('filter d where (x > 5 and y < 10) or (z == 3 and w != 7)')
        tokens = lexer.tokenize()

        # Should have parentheses
        lparen_tokens = [t for t in tokens if t.type == TokenType.LPAREN]
        rparen_tokens = [t for t in tokens if t.type == TokenType.RPAREN]

        assert len(lparen_tokens) >= 2
        assert len(rparen_tokens) >= 2

    def test_parameter_assignment(self):
        """Test parameter=value syntax."""
        lexer = Lexer('head sales with n=10')
        tokens = lexer.tokenize()

        # Should have EQUALS token
        equals_tokens = [t for t in tokens if t.type == TokenType.EQUALS]
        assert len(equals_tokens) >= 1

    def test_column_list(self):
        """Test comma-separated column list."""
        lexer = Lexer('select d with col1, col2, col3 as result')
        tokens = lexer.tokenize()

        # Should have COMMA tokens
        comma_tokens = [t for t in tokens if t.type == TokenType.COMMA]
        assert len(comma_tokens) == 2


class TestLexerEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_string(self):
        """Test lexing empty string."""
        lexer = Lexer('')
        tokens = lexer.tokenize()

        # Should return empty list or just EOF
        assert len(tokens) <= 1

    def test_whitespace_only(self):
        """Test lexing whitespace-only string."""
        lexer = Lexer('   \n\t  ')
        tokens = lexer.tokenize()

        # Should return empty list or just EOF
        assert len(tokens) <= 1

    def test_string_with_spaces(self):
        """Test string containing spaces."""
        lexer = Lexer('load "my data file.csv" as data')
        tokens = lexer.tokenize()

        string_tokens = [t for t in tokens if t.type == TokenType.STRING_LITERAL]
        assert len(string_tokens) == 1
        assert string_tokens[0].value == "my data file.csv"

    def test_string_with_special_chars(self):
        """Test string with special characters."""
        lexer = Lexer('load "path/to/data-2024_v1.csv" as data')
        tokens = lexer.tokenize()

        string_tokens = [t for t in tokens if t.type == TokenType.STRING_LITERAL]
        assert len(string_tokens) == 1
        assert "path/to/data-2024_v1.csv" in string_tokens[0].value

    def test_identifier_with_underscore(self):
        """Test identifier with underscore."""
        lexer = Lexer('select sales_data with price as result')
        tokens = lexer.tokenize()

        # Should have identifier with underscore
        identifiers = [t for t in tokens if t.type == TokenType.IDENTIFIER and '_' in t.value]
        assert len(identifiers) >= 1

    def test_identifier_with_numbers(self):
        """Test identifier with numbers."""
        lexer = Lexer('select data2024 with price as result')
        tokens = lexer.tokenize()

        # Should have identifier with number
        identifiers = [t for t in tokens if t.type == TokenType.IDENTIFIER and 'data2024' == t.value]
        assert len(identifiers) == 1


class TestLexerErrorHandling:
    """Tests for lexer error detection and handling."""

    def test_unclosed_string_double_quote(self):
        """Test error on unclosed double-quoted string."""
        lexer = Lexer('load "data.csv as sales')

        # Should raise NoetaError for unterminated string
        with pytest.raises(NoetaError) as exc_info:
            lexer.tokenize()

        error = exc_info.value
        assert error.category == ErrorCategory.LEXER

    def test_unclosed_string_single_quote(self):
        """Test error on unclosed single-quoted string."""
        lexer = Lexer("load 'data.csv as sales")

        with pytest.raises(NoetaError) as exc_info:
            lexer.tokenize()

        error = exc_info.value
        assert error.category == ErrorCategory.LEXER

    def test_invalid_character(self):
        """Test error on invalid character."""
        lexer = Lexer('load "data.csv" as sales @ test')

        with pytest.raises(NoetaError) as exc_info:
            lexer.tokenize()

        error = exc_info.value
        assert error.category == ErrorCategory.LEXER
        assert "@" in error.message or "invalid" in error.message.lower()

    def test_error_includes_position(self):
        """Test that errors include line and column info."""
        lexer = Lexer('load "data.csv" as sales @ test')

        with pytest.raises(NoetaError) as exc_info:
            lexer.tokenize()

        error = exc_info.value
        assert error.context.line >= 1
        assert error.context.column >= 0


class TestLexerIntegration:
    """Integration tests using fixture test cases."""

    def test_lexer_test_cases(self, lexer_test_cases):
        """Test all lexer test cases from fixture."""
        for name, source in lexer_test_cases.items():
            lexer = Lexer(source)
            tokens = lexer.tokenize()

            # All test cases should tokenize successfully
            assert len(tokens) > 0, f"Failed to tokenize: {name}"

            # Should have at least one non-EOF token
            non_eof_tokens = [t for t in tokens if t.type != TokenType.EOF]
            assert len(non_eof_tokens) > 0, f"No tokens generated for: {name}"
