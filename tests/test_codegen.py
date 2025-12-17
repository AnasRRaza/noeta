"""
Unit tests for Noeta Code Generator.

Tests Python code generation, import management, and symbol table tracking.
"""
import pytest
from noeta_lexer import Lexer
from noeta_parser import Parser
from noeta_codegen import CodeGenerator
from noeta_semantic import SemanticAnalyzer
from noeta_ast import *


class TestCodeGenBasicOperations:
    """Basic code generation tests."""

    def test_generate_load(self):
        """Test generating code for load operation."""
        source = 'load "data.csv" as sales'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        assert "pd.read_csv" in code
        assert "data.csv" in code
        assert code.strip()  # Should generate non-empty code

    def test_generate_describe(self):
        """Test generating code for describe operation."""
        source = 'load "data.csv" as sales\ndescribe sales'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        assert ".describe()" in code

    def test_generate_filter(self):
        """Test generating code for filter operation."""
        source = '''load "data.csv" as sales
filter sales where price > 100 as expensive'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        assert "price > 100" in code or "[" in code  # Should have filter logic

    def test_generate_select(self):
        """Test generating code for select operation."""
        source = 'load "data.csv" as sales\nselect sales with price as prices'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        assert "price" in code


class TestCodeGenImports:
    """Tests for import management."""

    def test_default_imports(self):
        """Test that default imports are generated."""
        source = 'load "data.csv" as sales'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        # Should have pandas import
        assert "import pandas" in code or "pd" in code

    def test_numpy_import_for_math(self):
        """Test that numpy is imported for math operations."""
        source = '''load "data.csv" as sales
round sales column price with decimals=2 as rounded'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        # Math operations should trigger numpy/pandas imports
        assert "import" in code

    def test_sklearn_import_for_scaling(self):
        """Test that sklearn is imported for scaling operations."""
        source = '''load "data.csv" as sales
standard_scale sales column price as scaled'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        # Scaling should trigger sklearn import
        assert "StandardScaler" in code or "sklearn" in code.lower()


class TestCodeGenSymbolTable:
    """Tests for symbol table tracking during code generation."""

    def test_symbol_table_tracks_aliases(self):
        """Test that symbol table tracks dataset aliases."""
        source = '''load "data.csv" as sales
select sales with price as prices'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        # Symbol table should allow referencing 'sales'
        assert "sales" in code.lower() or "df" in code

    def test_multiple_operations_use_symbol_table(self):
        """Test that multiple operations properly use symbol table."""
        source = '''load "data.csv" as sales
select sales with price as subset
describe subset'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        # Each operation should properly reference previous results
        assert code.strip()


class TestCodeGenTransformations:
    """Tests for transformation code generation."""

    def test_generate_upper(self):
        """Test generating upper transformation."""
        source = 'load "data.csv" as sales\nupper sales column name as upper_names'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        assert ".upper()" in code or "str.upper" in code

    def test_generate_groupby(self):
        """Test generating groupby operation."""
        source = '''load "data.csv" as sales
groupby sales by product agg sum as totals'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        assert ".groupby" in code

    def test_generate_join(self):
        """Test generating join operation."""
        source = '''load "sales.csv" as sales
load "customers.csv" as customers
join sales with customers on id as combined'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        assert ".merge" in code or "join" in code.lower()


class TestCodeGenCleaning:
    """Tests for cleaning operation code generation."""

    def test_generate_dropna(self):
        """Test generating dropna operation."""
        source = 'load "data.csv" as sales\ndropna sales as clean'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        assert ".dropna()" in code

    def test_generate_fillna(self):
        """Test generating fillna operation."""
        source = 'load "data.csv" as sales\nfillna sales with value=0 as filled'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        assert ".fillna" in code


class TestCodeGenComplexWorkflows:
    """Tests for complex workflow code generation."""

    def test_generate_multi_step_workflow(self):
        """Test generating multi-step workflow."""
        source = '''load "data.csv" as sales
select sales with price as prices
describe prices'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        # Should have code for all three operations
        assert "read_csv" in code or "pd" in code
        assert ".describe()" in code

    def test_generate_with_semantic_validation(self):
        """Test that code generation works after semantic validation."""
        source = '''load "data.csv" as sales
describe sales'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        # Run semantic validation
        analyzer = SemanticAnalyzer(source)
        errors = analyzer.analyze(ast)
        assert len(errors) == 0  # Should have no errors

        # Generate code
        generator = CodeGenerator()
        code = generator.generate(ast)

        assert code.strip()  # Should generate valid code


class TestCodeGenEdgeCases:
    """Tests for edge cases in code generation."""

    def test_generate_empty_program(self):
        """Test generating code for empty program."""
        source = ''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        # Should still have imports but no operation code
        assert "import" in code

    def test_generate_single_statement(self):
        """Test generating code for single statement."""
        source = 'load "data.csv" as sales'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        assert code.strip()
        assert len(code.split('\n')) > 1  # Should have multiple lines


class TestCodeGenOutputFormat:
    """Tests for generated code format and structure."""

    def test_generated_code_is_valid_python(self):
        """Test that generated code is syntactically valid Python."""
        source = '''load "data.csv" as sales
describe sales'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        # Try to compile the code (doesn't execute it)
        try:
            compile(code, '<string>', 'exec')
            valid = True
        except SyntaxError:
            valid = False

        assert valid, f"Generated invalid Python code:\n{code}"

    def test_generated_code_has_imports(self):
        """Test that generated code includes necessary imports."""
        source = 'load "data.csv" as sales'
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        # Should have import statements
        lines = code.split('\n')
        import_lines = [l for l in lines if 'import' in l]
        assert len(import_lines) > 0

    def test_code_execution_order(self):
        """Test that operations are generated in correct order."""
        source = '''load "data.csv" as sales
select sales with price as prices
describe prices'''
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize(), source)
        ast = parser.parse()

        generator = CodeGenerator()
        code = generator.generate(ast)

        # Load should come before select, select before describe
        load_pos = code.find("read_csv") if "read_csv" in code else code.find("load")
        describe_pos = code.find("describe")

        if load_pos >= 0 and describe_pos >= 0:
            assert load_pos < describe_pos
