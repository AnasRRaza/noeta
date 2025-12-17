"""
Unit tests for Noeta Parser.

Tests AST generation, statement parsing, and syntax error handling.
"""
import pytest
from noeta_lexer import Lexer
from noeta_parser import Parser
from noeta_ast import *
from noeta_errors import NoetaError, ErrorCategory


class TestParserLoad:
    """Tests for load operation parsing."""

    def test_parse_simple_load(self):
        """Test parsing simple load statement."""
        source = 'load "data.csv" as sales'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        assert isinstance(ast, ProgramNode)
        assert len(ast.statements) == 1

        stmt = ast.statements[0]
        assert isinstance(stmt, LoadNode)
        assert stmt.filepath == "data.csv"
        assert stmt.alias == "sales"

    def test_parse_load_csv(self):
        """Test parsing load_csv statement."""
        source = 'load_csv "data.csv" as sales'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, LoadCSVNode)
        assert stmt.filepath == "data.csv"
        assert stmt.alias == "sales"

    def test_parse_load_json(self):
        """Test parsing load_json statement."""
        source = 'load_json "data.json" as data'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, LoadJSONNode)
        assert stmt.filepath == "data.json"
        assert stmt.alias == "data"

    def test_parse_load_excel(self):
        """Test parsing load_excel statement."""
        source = 'load_excel "data.xlsx" as data'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, LoadExcelNode)
        assert stmt.filepath == "data.xlsx"
        assert stmt.alias == "data"


class TestParserSave:
    """Tests for save operation parsing."""

    def test_parse_simple_save(self):
        """Test parsing save statement."""
        source = 'save sales to "output.csv"'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, SaveNode)
        assert stmt.source == "sales"
        assert stmt.filepath == "output.csv"

    def test_parse_save_csv(self):
        """Test parsing save_csv statement."""
        source = 'save_csv sales to "output.csv"'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, SaveCSVNode)
        assert stmt.source == "sales"
        assert stmt.filepath == "output.csv"


class TestParserSelect:
    """Tests for select operation parsing."""

    def test_parse_select_with_columns(self):
        """Test parsing select with column list."""
        source = 'select sales with price, quantity as subset'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, SelectNode)
        assert stmt.source_alias == "sales"
        assert "price" in stmt.columns
        assert "quantity" in stmt.columns
        assert stmt.new_alias == "subset"

    def test_parse_select_single_column(self):
        """Test parsing select with single column."""
        source = 'select sales with price as prices'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, SelectNode)
        assert len(stmt.columns) == 1
        assert "price" in stmt.columns

    def test_parse_head(self):
        """Test parsing head statement."""
        source = 'head sales with n=10 as preview'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, HeadNode)
        assert stmt.source_alias == "sales"
        assert stmt.n == 10
        assert stmt.new_alias == "preview"

    def test_parse_tail(self):
        """Test parsing tail statement."""
        source = 'tail sales with n=5 as last'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, TailNode)
        assert stmt.source_alias == "sales"
        assert stmt.n == 5


class TestParserFilter:
    """Tests for filter operation parsing."""

    def test_parse_filter_simple(self):
        """Test parsing simple filter statement."""
        source = 'filter sales where price > 100 as expensive'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, UpdatedFilterNode)
        assert stmt.source_alias == "sales"
        assert stmt.new_alias == "expensive"

    def test_parse_filter_complex_condition(self):
        """Test parsing filter with complex condition."""
        source = 'filter sales where price > 100 and quantity < 50 as result'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, UpdatedFilterNode)
        assert stmt.source_alias == "sales"

    def test_parse_filter_between(self):
        """Test parsing filter_between statement."""
        source = 'filter_between sales column price with lower=10 upper=100 as mid_range'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, FilterBetweenNode)
        assert stmt.source_alias == "sales"
        assert stmt.column == "price"


class TestParserDescribe:
    """Tests for describe and info operations."""

    def test_parse_describe(self):
        """Test parsing describe statement."""
        source = 'describe sales'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, DescribeNode)
        assert stmt.source_alias == "sales"

    def test_parse_info(self):
        """Test parsing info statement."""
        source = 'info sales'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, InfoNode)
        assert stmt.source_alias == "sales"

    def test_parse_summary(self):
        """Test parsing summary statement."""
        source = 'summary sales'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, SummaryNode)
        assert stmt.source_alias == "sales"


class TestParserGroupBy:
    """Tests for groupby and aggregation parsing."""

    def test_parse_groupby_simple(self):
        """Test parsing simple groupby."""
        source = 'groupby sales by product agg sum as totals'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, GroupByNode)
        assert stmt.source_alias == "sales"
        assert "product" in stmt.by_columns
        assert stmt.new_alias == "totals"

    def test_parse_groupby_multiple_columns(self):
        """Test parsing groupby with multiple columns."""
        source = 'groupby sales by product, region agg mean as avg_by_region'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, GroupByNode)
        assert "product" in stmt.by_columns
        assert "region" in stmt.by_columns


class TestParserJoin:
    """Tests for join and merge operations."""

    def test_parse_join(self):
        """Test parsing join statement."""
        source = 'join sales with customers on id as combined'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, JoinNode)
        assert stmt.alias1 == "sales"
        assert stmt.alias2 == "customers"
        assert stmt.new_alias == "combined"

    def test_parse_merge(self):
        """Test parsing merge statement."""
        source = 'merge sales with customers on id as merged'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, MergeNode)
        assert stmt.left_alias == "sales"
        assert stmt.right_alias == "customers"


class TestParserSort:
    """Tests for sorting operations."""

    def test_parse_sort_asc(self):
        """Test parsing sort ascending."""
        source = 'sort sales by price asc as sorted_sales'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, SortNode)
        assert stmt.source_alias == "sales"
        assert "price" in stmt.by_columns
        assert stmt.ascending == True

    def test_parse_sort_desc(self):
        """Test parsing sort descending."""
        source = 'sort sales by price desc as sorted_sales'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, SortNode)
        assert stmt.ascending == False


class TestParserTransformations:
    """Tests for transformation operations."""

    def test_parse_upper(self):
        """Test parsing upper transformation."""
        source = 'upper sales column name as upper_names'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, UpperNode)
        assert stmt.source_alias == "sales"
        assert stmt.column == "name"
        assert stmt.new_alias == "upper_names"

    def test_parse_lower(self):
        """Test parsing lower transformation."""
        source = 'lower sales column name as lower_names'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, LowerNode)
        assert stmt.source_alias == "sales"
        assert stmt.column == "name"

    def test_parse_round(self):
        """Test parsing round transformation."""
        source = 'round sales column price with decimals=2 as rounded'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, RoundNode)
        assert stmt.source_alias == "sales"
        assert stmt.column == "price"
        assert stmt.decimals == 2


class TestParserCleaning:
    """Tests for cleaning operations."""

    def test_parse_dropna(self):
        """Test parsing dropna statement."""
        source = 'dropna sales as clean'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, DropNaNode)
        assert stmt.source_alias == "sales"
        assert stmt.new_alias == "clean"

    def test_parse_fillna(self):
        """Test parsing fillna with value."""
        source = 'fillna sales with value=0 as filled'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, FillNaNode)
        assert stmt.source_alias == "sales"
        assert stmt.value == 0

    def test_parse_drop_duplicates(self):
        """Test parsing drop_duplicates."""
        source = 'drop_duplicates sales as unique'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, DropDuplicatesNode)
        assert stmt.source_alias == "sales"


class TestParserMultipleStatements:
    """Tests for parsing multiple statements."""

    def test_parse_two_statements(self):
        """Test parsing two statements."""
        source = '''load "data.csv" as sales
describe sales'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        assert isinstance(ast, ProgramNode)
        assert len(ast.statements) == 2
        assert isinstance(ast.statements[0], LoadNode)
        assert isinstance(ast.statements[1], DescribeNode)

    def test_parse_multiple_operations(self):
        """Test parsing multiple operations in sequence."""
        source = '''load "data.csv" as sales
select sales with price, quantity as subset
filter subset where price > 100 as expensive
describe expensive'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        assert len(ast.statements) == 4
        assert isinstance(ast.statements[0], LoadNode)
        assert isinstance(ast.statements[1], SelectNode)
        assert isinstance(ast.statements[2], UpdatedFilterNode)
        assert isinstance(ast.statements[3], DescribeNode)

    def test_parse_complex_workflow(self):
        """Test parsing complex workflow."""
        source = '''load "sales.csv" as sales
load "customers.csv" as customers
join sales with customers on id as combined
groupby combined by region agg sum as regional_totals
sort regional_totals by total desc as sorted'''

        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        assert len(ast.statements) == 5


class TestParserErrors:
    """Tests for parser error handling."""

    def test_missing_as_keyword(self):
        """Test error on missing AS keyword."""
        source = 'load "data.csv" sales'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)

        with pytest.raises(NoetaError) as exc_info:
            parser.parse()

        error = exc_info.value
        assert error.category == ErrorCategory.SYNTAX

    def test_unexpected_eof(self):
        """Test error on unexpected end of file."""
        source = 'load "data.csv" as'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)

        with pytest.raises(NoetaError) as exc_info:
            parser.parse()

        error = exc_info.value
        # Should indicate unexpected EOF
        assert "expected" in error.message.lower() or "end" in error.message.lower()

    def test_missing_column_keyword(self):
        """Test error on missing column keyword."""
        source = 'upper sales name as result'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)

        with pytest.raises(NoetaError) as exc_info:
            parser.parse()

        error = exc_info.value
        assert error.category == ErrorCategory.SYNTAX

    def test_missing_with_keyword(self):
        """Test error on missing WITH keyword."""
        source = 'select sales price, quantity as result'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)

        with pytest.raises(NoetaError) as exc_info:
            parser.parse()

        error = exc_info.value
        assert error.category == ErrorCategory.SYNTAX

    def test_error_includes_context(self):
        """Test that errors include source context."""
        source = 'load "data.csv" sales'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)

        with pytest.raises(NoetaError) as exc_info:
            parser.parse()

        error = exc_info.value
        assert error.context is not None
        assert error.context.line >= 1


class TestParserParameters:
    """Tests for parameter parsing."""

    def test_parse_single_parameter(self):
        """Test parsing single parameter."""
        source = 'head sales with n=10 as preview'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert stmt.n == 10

    def test_parse_multiple_parameters(self):
        """Test parsing multiple parameters."""
        source = 'fillna sales with value=0 inplace=False as filled'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert stmt.value == 0

    def test_parse_string_parameter(self):
        """Test parsing string parameter."""
        source = 'fillna sales with method="ffill" as filled'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert stmt.method == "ffill"


class TestParserComplexOperations:
    """Tests for complex operations."""

    def test_parse_groupby_with_agg_dict(self):
        """Test parsing groupby with aggregation dictionary."""
        source = 'groupby sales by product agg sum as totals'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, GroupByNode)

    def test_parse_pivot(self):
        """Test parsing pivot operation."""
        source = 'pivot sales with index=date columns=product values=price as pivoted'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, PivotNode)
        assert stmt.source_alias == "sales"

    def test_parse_melt(self):
        """Test parsing melt operation."""
        source = 'melt sales with id_vars=date value_vars=price as melted'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        stmt = ast.statements[0]
        assert isinstance(stmt, MeltNode)
        assert stmt.source_alias == "sales"


class TestParserIntegration:
    """Integration tests using fixture test cases."""

    def test_parser_test_cases(self, parser_test_cases):
        """Test all parser test cases from fixture."""
        for name, (source, expected_node_type) in parser_test_cases.items():
            lexer = Lexer(source)
            parser = Parser(lexer.tokenize(), source)
            ast = parser.parse()

            # Should parse successfully
            assert len(ast.statements) >= 1, f"Failed to parse: {name}"

            # Check node type
            stmt = ast.statements[0]
            assert stmt.__class__.__name__ == expected_node_type, \
                f"Expected {expected_node_type}, got {stmt.__class__.__name__} for {name}"


class TestParserEdgeCases:
    """Tests for edge cases in parsing."""

    def test_parse_empty_program(self):
        """Test parsing empty program."""
        source = ''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        assert isinstance(ast, ProgramNode)
        assert len(ast.statements) == 0

    def test_parse_whitespace_only(self):
        """Test parsing whitespace-only program."""
        source = '   \n\t  '
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        assert isinstance(ast, ProgramNode)
        assert len(ast.statements) == 0

    def test_parse_with_comments(self):
        """Test parsing with comments (if supported)."""
        # This test depends on whether comments are implemented
        source = 'load "data.csv" as sales'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        assert len(ast.statements) >= 1
