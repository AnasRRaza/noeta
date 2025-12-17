"""
Integration tests for Noeta DSL.

End-to-end tests covering the full compilation pipeline.
"""
import pytest
import pandas as pd
from noeta_runner import compile_noeta, execute_noeta
from noeta_errors import NoetaError, ErrorCategory


class TestEndToEndCompilation:
    """End-to-end compilation tests."""

    def test_compile_simple_load(self):
        """Test compiling simple load statement."""
        source = 'load "data.csv" as sales'

        code = compile_noeta(source)

        assert isinstance(code, str)
        assert "pd.read_csv" in code
        assert len(code) > 0

    def test_compile_with_describe(self):
        """Test compiling load and describe."""
        source = '''load "data.csv" as sales
describe sales'''

        code = compile_noeta(source)

        assert "pd.read_csv" in code
        assert ".describe()" in code

    def test_compile_with_filter(self):
        """Test compiling with filter operation."""
        source = '''load "data.csv" as sales
filter sales where price > 100 as expensive'''

        code = compile_noeta(source)

        assert "pd.read_csv" in code
        assert code.strip()

    def test_compile_complex_workflow(self):
        """Test compiling complex workflow."""
        source = '''load "sales.csv" as sales
select sales with price as prices
filter prices where price > 100 as expensive
describe expensive'''

        code = compile_noeta(source)

        assert "pd.read_csv" in code
        assert ".describe()" in code


class TestEndToEndSemanticValidation:
    """Tests for semantic validation in compilation."""

    def test_compile_catches_undefined_dataset(self):
        """Test that compilation catches undefined datasets."""
        source = 'select unknown with price as result'

        with pytest.raises(NoetaError) as exc_info:
            compile_noeta(source)

        error = exc_info.value
        assert error.category == ErrorCategory.SEMANTIC
        assert "unknown" in error.message

    def test_compile_valid_workflow(self):
        """Test that valid workflow compiles without errors."""
        source = '''load "data.csv" as sales
describe sales'''

        # Should not raise any errors
        code = compile_noeta(source)
        assert len(code) > 0

    def test_compile_catches_typo_in_dataset_name(self):
        """Test that compilation catches typos in dataset names."""
        source = '''load "data.csv" as sales
select sale with price as result'''

        with pytest.raises(NoetaError) as exc_info:
            compile_noeta(source)

        error = exc_info.value
        assert error.category == ErrorCategory.SEMANTIC


class TestCompilationPipeline:
    """Tests for the full compilation pipeline."""

    def test_pipeline_lexer_to_parser(self):
        """Test that lexer output feeds into parser correctly."""
        source = 'load "data.csv" as sales'

        # Should go through lexer -> parser -> semantic -> codegen
        code = compile_noeta(source)
        assert isinstance(code, str)

    def test_pipeline_with_semantic_validation(self):
        """Test that semantic validation is part of pipeline."""
        source = 'describe nonexistent'

        # Should fail at semantic validation stage
        with pytest.raises(NoetaError) as exc_info:
            compile_noeta(source)

        assert exc_info.value.category == ErrorCategory.SEMANTIC

    def test_pipeline_generates_executable_python(self):
        """Test that pipeline generates executable Python code."""
        source = 'load "data.csv" as sales'

        code = compile_noeta(source)

        # Code should be syntactically valid Python
        try:
            compile(code, '<string>', 'exec')
            valid = True
        except SyntaxError:
            valid = False

        assert valid


class TestFileExecution:
    """Tests for executing .noeta files."""

    def test_execute_with_csv_fixture(self, sample_csv):
        """Test executing Noeta code with real CSV file."""
        source = f'''load "{sample_csv}" as data
describe data'''

        # Should execute without errors (but might produce output)
        result = execute_noeta(source, verbose=False)

        # execute_noeta returns 0 on success, 1 on error
        assert result == 0 or result is None

    def test_execute_multi_operation_workflow(self, sample_csv):
        """Test executing workflow with multiple operations."""
        source = f'''load "{sample_csv}" as data
select data with name as names
describe names'''

        result = execute_noeta(source, verbose=False)

        assert result == 0 or result is None


class TestErrorPropagation:
    """Tests for error handling across pipeline stages."""

    def test_lexer_error_propagates(self):
        """Test that lexer errors propagate correctly."""
        source = 'load "unterminated'

        with pytest.raises(NoetaError) as exc_info:
            compile_noeta(source)

        assert exc_info.value.category == ErrorCategory.LEXER

    def test_syntax_error_propagates(self):
        """Test that syntax errors propagate correctly."""
        source = 'load "data.csv" sales'  # Missing AS

        with pytest.raises(NoetaError) as exc_info:
            compile_noeta(source)

        assert exc_info.value.category == ErrorCategory.SYNTAX

    def test_semantic_error_propagates(self):
        """Test that semantic errors propagate correctly."""
        source = 'describe undefined'

        with pytest.raises(NoetaError) as exc_info:
            compile_noeta(source)

        assert exc_info.value.category == ErrorCategory.SEMANTIC

    def test_error_has_context(self):
        """Test that errors include source context."""
        source = 'describe undefined_dataset'

        with pytest.raises(NoetaError) as exc_info:
            compile_noeta(source)

        error = exc_info.value
        assert error.context is not None
        assert error.context.line >= 1


class TestComplexWorkflows:
    """Tests for complex multi-step workflows."""

    def test_join_workflow(self):
        """Test workflow with join operation."""
        source = '''load "sales.csv" as sales
load "customers.csv" as customers
join sales with customers on id as combined'''

        code = compile_noeta(source)

        assert "pd.read_csv" in code
        assert code.count("pd.read_csv") >= 2  # Two loads

    def test_groupby_workflow(self):
        """Test workflow with groupby."""
        source = '''load "data.csv" as sales
groupby sales by product agg sum as totals'''

        code = compile_noeta(source)

        assert "pd.read_csv" in code
        assert ".groupby" in code

    def test_filter_select_chain(self):
        """Test chaining filter and select."""
        source = '''load "data.csv" as sales
filter sales where price > 100 as expensive
select expensive with name, price as result'''

        code = compile_noeta(source)

        # Should reference intermediate dataset
        assert "pd.read_csv" in code


class TestDataTransformations:
    """Tests for data transformation workflows."""

    def test_string_transformation(self):
        """Test string transformation workflow."""
        source = '''load "data.csv" as sales
upper sales column name as upper_names'''

        code = compile_noeta(source)

        assert "pd.read_csv" in code
        assert ".upper()" in code or "str.upper" in code

    def test_numeric_transformation(self):
        """Test numeric transformation workflow."""
        source = '''load "data.csv" as sales
round sales column price with decimals=2 as rounded'''

        code = compile_noeta(source)

        assert "pd.read_csv" in code

    def test_cleaning_workflow(self):
        """Test data cleaning workflow."""
        source = '''load "data.csv" as sales
dropna sales as clean
fillna clean with value=0 as filled'''

        code = compile_noeta(source)

        assert ".dropna()" in code
        assert ".fillna" in code


class TestMultipleDatasets:
    """Tests for workflows with multiple datasets."""

    def test_load_multiple_datasets(self):
        """Test loading multiple datasets."""
        source = '''load "sales.csv" as sales
load "customers.csv" as customers
load "products.csv" as products'''

        code = compile_noeta(source)

        # Should have three loads
        assert code.count("pd.read_csv") >= 3

    def test_operations_on_different_datasets(self):
        """Test operations on different datasets."""
        source = '''load "sales.csv" as sales
load "customers.csv" as customers
describe sales
describe customers'''

        code = compile_noeta(source)

        # Should have two describe operations
        assert code.count(".describe()") >= 2

    def test_join_multiple_datasets(self):
        """Test joining multiple datasets."""
        source = '''load "a.csv" as a
load "b.csv" as b
join a with b on id as ab'''

        code = compile_noeta(source)

        assert code.count("pd.read_csv") >= 2


class TestEdgeCases:
    """Tests for edge cases in integration."""

    def test_empty_program(self):
        """Test compiling empty program."""
        source = ''

        code = compile_noeta(source)

        # Should still generate imports
        assert "import" in code

    def test_single_statement(self):
        """Test compiling single statement."""
        source = 'load "data.csv" as sales'

        code = compile_noeta(source)

        assert "pd.read_csv" in code

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        source = '''
load "data.csv" as sales

describe sales
'''

        code = compile_noeta(source)

        assert "pd.read_csv" in code
        assert ".describe()" in code


class TestVerboseMode:
    """Tests for verbose mode execution."""

    def test_verbose_mode_shows_code(self, sample_csv, capsys):
        """Test that verbose mode displays generated code."""
        source = f'load "{sample_csv}" as data'

        execute_noeta(source, verbose=True)

        captured = capsys.readouterr()
        # Verbose mode should show generated code
        assert "Generated Python Code" in captured.out or "import" in captured.out


class TestRealDataWorkflows:
    """Tests with real data fixtures."""

    def test_workflow_with_sample_csv(self, sample_csv):
        """Test complete workflow with sample CSV."""
        source = f'''load "{sample_csv}" as employees
select employees with name, age as subset
filter subset where age > 25 as adults
describe adults'''

        code = compile_noeta(source)

        assert "pd.read_csv" in code
        assert sample_csv in code
        assert ".describe()" in code

    def test_workflow_with_valid_noeta_script(self, valid_noeta_with_csv):
        """Test workflow using valid Noeta script fixture."""
        code = compile_noeta(valid_noeta_with_csv)

        assert "pd.read_csv" in code
        assert ".describe()" in code or "describe" in code.lower()


class TestCompileVsExecute:
    """Tests comparing compile and execute functions."""

    def test_compile_returns_string(self):
        """Test that compile_noeta returns string."""
        source = 'load "data.csv" as sales'

        result = compile_noeta(source)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_compile_error_raises_exception(self):
        """Test that compilation errors raise exceptions."""
        source = 'describe undefined'

        with pytest.raises(NoetaError):
            compile_noeta(source)


class TestFullPipelineValidation:
    """Tests for complete pipeline validation."""

    def test_pipeline_order(self):
        """Test that pipeline executes stages in correct order."""
        source = '''load "data.csv" as data
select unknown with price as result'''

        # Should fail at semantic validation (after parsing, before codegen)
        with pytest.raises(NoetaError) as exc_info:
            compile_noeta(source)

        # Should be semantic error, not runtime error
        assert exc_info.value.category == ErrorCategory.SEMANTIC

    def test_semantic_validation_before_codegen(self):
        """Test that semantic validation happens before code generation."""
        source = 'select nonexistent with column as result'

        with pytest.raises(NoetaError) as exc_info:
            compile_noeta(source)

        # Error should be caught at compile time, not runtime
        error = exc_info.value
        assert error.category == ErrorCategory.SEMANTIC
        assert "nonexistent" in error.message


class TestScalingOperations:
    """Tests for scaling and encoding operations."""

    def test_standard_scaling(self):
        """Test standard scaling operation."""
        source = '''load "data.csv" as sales
standard_scale sales column price as scaled'''

        code = compile_noeta(source)

        assert "StandardScaler" in code

    def test_encoding_operation(self):
        """Test encoding operation."""
        source = '''load "data.csv" as sales
one_hot_encode sales column category as encoded'''

        code = compile_noeta(source)

        # Should have sklearn or pandas get_dummies
        assert "get_dummies" in code or "sklearn" in code.lower()


class TestStatisticalOperations:
    """Tests for statistical operations."""

    def test_describe_operation(self):
        """Test describe operation."""
        source = '''load "data.csv" as sales
describe sales'''

        code = compile_noeta(source)

        assert ".describe()" in code

    def test_summary_operation(self):
        """Test summary operation."""
        source = '''load "data.csv" as sales
summary sales'''

        code = compile_noeta(source)

        # Should have some summary code
        assert code.strip()

    def test_info_operation(self):
        """Test info operation."""
        source = '''load "data.csv" as sales
info sales'''

        code = compile_noeta(source)

        assert ".info()" in code
