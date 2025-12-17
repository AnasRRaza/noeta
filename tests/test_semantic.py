"""
Unit tests for Noeta Semantic Analyzer.

Tests semantic validation, symbol table management, and error detection.
"""
import pytest
from noeta_lexer import Lexer
from noeta_parser import Parser
from noeta_semantic import SemanticAnalyzer, SymbolTable, DatasetInfo, ColumnInfo
from noeta_errors import NoetaError, ErrorCategory


class TestSemanticUndefinedDataset:
    """Tests for undefined dataset detection."""

    def test_undefined_dataset_in_select(self):
        """Test error on undefined dataset in select."""
        source = 'select unknown with price as result'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 1
        assert errors[0].category == ErrorCategory.SEMANTIC
        assert "unknown" in errors[0].message
        assert "not been loaded" in errors[0].message.lower()

    def test_undefined_dataset_in_filter(self):
        """Test error on undefined dataset in filter."""
        source = 'filter nonexistent where price > 100 as result'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 1
        assert errors[0].category == ErrorCategory.SEMANTIC
        assert "nonexistent" in errors[0].message

    def test_undefined_dataset_in_describe(self):
        """Test error on undefined dataset in describe."""
        source = 'describe missing_data'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 1
        assert "missing_data" in errors[0].message

    def test_undefined_dataset_in_groupby(self):
        """Test error on undefined dataset in groupby."""
        source = 'groupby undefined by product agg sum as result'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 1
        assert "undefined" in errors[0].message

    def test_undefined_dataset_in_save(self):
        """Test error on undefined dataset in save."""
        source = 'save nonexistent to "output.csv"'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 1
        assert "nonexistent" in errors[0].message


class TestSemanticDatasetSuggestions:
    """Tests for dataset name suggestions."""

    def test_suggests_similar_dataset(self):
        """Test that analyzer suggests similar dataset names."""
        source = '''load "data.csv" as sales
select sale with price as result'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 1
        # Should suggest "sales" for "sale"
        assert errors[0].suggestion is not None or "sales" in str(errors[0])

    def test_suggests_closest_match(self):
        """Test that analyzer suggests the closest matching dataset."""
        source = '''load "data1.csv" as employees
load "data2.csv" as customers
select employes with name as result'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 1
        # Should suggest "employees" not "customers"
        error_text = str(errors[0])
        assert "employees" in error_text.lower()

    def test_no_suggestion_for_very_different_name(self):
        """Test no suggestion when name is very different."""
        source = '''load "data.csv" as sales
select xyz123 with price as result'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 1
        # Might not suggest anything for very different names


class TestSemanticSymbolTable:
    """Tests for symbol table management."""

    def test_symbol_table_registers_load(self):
        """Test that load operation registers dataset in symbol table."""
        source = 'load "data.csv" as sales'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 0
        assert analyzer.symbol_table.exists("sales")

    def test_symbol_table_registers_transformations(self):
        """Test that transformations register new datasets."""
        source = '''load "data.csv" as sales
select sales with price as prices'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 0
        assert analyzer.symbol_table.exists("sales")
        assert analyzer.symbol_table.exists("prices")

    def test_symbol_table_registers_filter(self):
        """Test that filter operation registers new dataset."""
        source = '''load "data.csv" as sales
filter sales where price > 100 as expensive'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 0
        assert analyzer.symbol_table.exists("expensive")

    def test_symbol_table_registers_multiple_datasets(self):
        """Test that symbol table tracks multiple datasets."""
        source = '''load "sales.csv" as sales
load "customers.csv" as customers
load "products.csv" as products'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 0
        assert analyzer.symbol_table.exists("sales")
        assert analyzer.symbol_table.exists("customers")
        assert analyzer.symbol_table.exists("products")


class TestSemanticValidOperations:
    """Tests for valid operation sequences."""

    def test_valid_simple_workflow(self):
        """Test valid simple workflow has no errors."""
        source = '''load "data.csv" as sales
describe sales'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 0

    def test_valid_chained_operations(self):
        """Test valid chained operations have no errors."""
        source = '''load "data.csv" as sales
select sales with price as prices
filter prices where price > 100 as expensive
describe expensive'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 0

    def test_valid_parallel_operations(self):
        """Test valid parallel operations on same dataset."""
        source = '''load "data.csv" as sales
select sales with price as prices
select sales with quantity as quantities'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 0

    def test_valid_join_operation(self):
        """Test valid join operation."""
        source = '''load "sales.csv" as sales
load "customers.csv" as customers
join sales with customers on id as combined'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 0

    def test_valid_groupby(self):
        """Test valid groupby operation."""
        source = '''load "data.csv" as sales
groupby sales by product agg sum as totals'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 0


class TestSemanticChainedOperations:
    """Tests for chained operation validation."""

    def test_chain_break_in_middle(self):
        """Test error when chain breaks in the middle."""
        source = '''load "data.csv" as data
filter data where price > 100 as filtered
select data_filtered with name as result'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 1
        assert "data_filtered" in errors[0].message

    def test_using_intermediate_result(self):
        """Test using intermediate result in chain."""
        source = '''load "data.csv" as data
select data with price as prices
filter prices where price > 100 as expensive'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 0

    def test_cannot_reuse_original_after_chain(self):
        """Test that original dataset can still be used after creating derived dataset."""
        source = '''load "data.csv" as data
select data with price as prices
select data with quantity as quantities'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        # Should be valid - can use original dataset multiple times
        assert len(errors) == 0


class TestSemanticMultipleDatasets:
    """Tests for operations involving multiple datasets."""

    def test_join_with_undefined_left(self):
        """Test error when left dataset in join is undefined."""
        source = '''load "customers.csv" as customers
join undefined with customers on id as result'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 1
        assert "undefined" in errors[0].message

    def test_join_with_undefined_right(self):
        """Test error when right dataset in join is undefined."""
        source = '''load "sales.csv" as sales
join sales with undefined on id as result'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 1
        assert "undefined" in errors[0].message

    def test_valid_multiple_dataset_workflow(self):
        """Test valid workflow with multiple datasets."""
        source = '''load "sales.csv" as sales
load "customers.csv" as customers
load "products.csv" as products
join sales with customers on customer_id as sales_customers
join sales_customers with products on product_id as full_data'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 0


class TestSemanticTransformations:
    """Tests for transformation operation validation."""

    def test_valid_upper_transformation(self):
        """Test valid upper transformation."""
        source = '''load "data.csv" as sales
upper sales column name as upper_names'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 0

    def test_undefined_dataset_in_transformation(self):
        """Test error on undefined dataset in transformation."""
        source = 'upper undefined column name as result'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 1
        assert "undefined" in errors[0].message

    def test_valid_round_transformation(self):
        """Test valid round transformation."""
        source = '''load "data.csv" as sales
round sales column price with decimals=2 as rounded'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 0


class TestSemanticCleaning:
    """Tests for cleaning operation validation."""

    def test_valid_dropna(self):
        """Test valid dropna operation."""
        source = '''load "data.csv" as sales
dropna sales as clean'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 0

    def test_valid_fillna(self):
        """Test valid fillna operation."""
        source = '''load "data.csv" as sales
fillna sales with value=0 as filled'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 0

    def test_undefined_dataset_in_cleaning(self):
        """Test error on undefined dataset in cleaning."""
        source = 'dropna undefined as clean'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)

        assert len(errors) == 1


class TestSemanticIntegration:
    """Integration tests using fixture test cases."""

    def test_semantic_test_cases(self, semantic_test_cases):
        """Test all semantic test cases from fixture."""
        for name, (source, should_error, error_contains) in semantic_test_cases.items():
            lexer = Lexer(source)
            parser = Parser(lexer.tokenize(), source)
            ast = parser.parse()

            analyzer = SemanticAnalyzer(source)
            errors = analyzer.analyze(ast)

            if should_error:
                assert len(errors) > 0, f"Expected error for test case: {name}"
                if error_contains:
                    error_messages = " ".join([str(e) for e in errors])
                    assert error_contains in error_messages, \
                        f"Expected '{error_contains}' in error for test case: {name}"
            else:
                assert len(errors) == 0, \
                    f"Unexpected error for test case: {name}: {errors[0] if errors else 'None'}"


class TestSymbolTableAPI:
    """Tests for SymbolTable API."""

    def test_define_and_lookup(self):
        """Test defining and looking up datasets."""
        table = SymbolTable()
        info = DatasetInfo(name="sales", columns={})

        table.define("sales", info)

        assert table.exists("sales")
        assert table.lookup("sales") == info

    def test_lookup_nonexistent(self):
        """Test looking up nonexistent dataset."""
        table = SymbolTable()

        assert not table.exists("nonexistent")
        assert table.lookup("nonexistent") is None

    def test_get_all_names(self):
        """Test getting all dataset names."""
        table = SymbolTable()

        table.define("sales", DatasetInfo(name="sales", columns={}))
        table.define("customers", DatasetInfo(name="customers", columns={}))

        names = table.get_all_names()
        assert "sales" in names
        assert "customers" in names
        assert len(names) == 2

    def test_history_tracking(self):
        """Test that symbol table tracks definition order."""
        table = SymbolTable()

        table.define("first", DatasetInfo(name="first", columns={}))
        table.define("second", DatasetInfo(name="second", columns={}))
        table.define("third", DatasetInfo(name="third", columns={}))

        assert len(table.history) == 3
        assert table.history[0] == "first"
        assert table.history[1] == "second"
        assert table.history[2] == "third"


class TestDatasetInfo:
    """Tests for DatasetInfo class."""

    def test_has_column(self):
        """Test checking if dataset has column."""
        info = DatasetInfo(
            name="sales",
            columns={
                "price": ColumnInfo(name="price"),
                "quantity": ColumnInfo(name="quantity")
            }
        )

        assert info.has_column("price")
        assert info.has_column("quantity")
        assert not info.has_column("nonexistent")

    def test_get_column_type(self):
        """Test getting column type."""
        from noeta_semantic import DataType

        info = DatasetInfo(
            name="sales",
            columns={
                "price": ColumnInfo(name="price", dtype=DataType.NUMERIC)
            }
        )

        assert info.get_column_type("price") == DataType.NUMERIC
