"""
Shared pytest fixtures for Noeta test suite.

This module provides reusable test fixtures including:
- Sample CSV data files
- Valid and invalid Noeta scripts
- Temporary directories for output files
"""
import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_csv(tmp_path):
    """
    Create a temporary CSV file with sample data for testing.

    Returns:
        str: Path to the temporary CSV file
    """
    csv_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000, 60000, 70000, 55000, 65000],
        'department': ['Sales', 'Engineering', 'Sales', 'HR', 'Engineering']
    })
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_sales_csv(tmp_path):
    """
    Create a temporary CSV file with sales data for testing.

    Returns:
        str: Path to the temporary sales CSV file
    """
    csv_path = tmp_path / "sales_data.csv"
    df = pd.DataFrame({
        'product': ['Widget A', 'Widget B', 'Widget A', 'Widget C', 'Widget B'],
        'price': [100.0, 150.0, 100.0, 200.0, 150.0],
        'quantity': [10, 5, 15, 3, 8],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
    })
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_json(tmp_path):
    """
    Create a temporary JSON file with sample data for testing.

    Returns:
        str: Path to the temporary JSON file
    """
    json_path = tmp_path / "test_data.json"
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'score': [95, 87, 92]
    })
    df.to_json(json_path, orient='records', indent=2)
    return str(json_path)


@pytest.fixture
def valid_noeta_script():
    """
    Return a valid Noeta script for testing.

    Returns:
        str: Valid Noeta source code
    """
    return '''load "data/sales.csv" as sales
select sales with price, quantity as subset
filter subset where price > 100 as filtered
describe filtered'''


@pytest.fixture
def valid_noeta_with_csv(sample_csv):
    """
    Return a valid Noeta script that uses a real CSV file.

    Args:
        sample_csv: Fixture providing path to sample CSV

    Returns:
        str: Valid Noeta source code using the sample CSV
    """
    return f'''load "{sample_csv}" as data
select data with name, age as subset
filter subset where age > 25 as filtered
describe filtered'''


@pytest.fixture
def invalid_noeta_script():
    """
    Return an invalid Noeta script for testing (undefined dataset).

    Returns:
        str: Invalid Noeta source code
    """
    return '''load "data/sales.csv" as sales
select unknown with price as result'''


@pytest.fixture
def invalid_syntax_script():
    """
    Return a Noeta script with syntax errors.

    Returns:
        str: Noeta source code with syntax errors
    """
    return '''load "data/sales.csv" sales
select sales with price'''


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Create a temporary directory for output files.

    Yields:
        Path: Path to the temporary output directory
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    yield output_dir
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def noeta_test_script_basic(tmp_path):
    """
    Create a basic Noeta test script file.

    Args:
        tmp_path: pytest's temporary directory fixture

    Returns:
        str: Path to the created .noeta file
    """
    script_path = tmp_path / "test_basic.noeta"
    script_content = '''load "data/test.csv" as data
describe data
head data with n=5 as preview'''

    script_path.write_text(script_content)
    return str(script_path)


@pytest.fixture
def noeta_test_script_complex(tmp_path, sample_csv):
    """
    Create a complex Noeta test script with multiple operations.

    Args:
        tmp_path: pytest's temporary directory fixture
        sample_csv: Sample CSV file fixture

    Returns:
        str: Path to the created .noeta file
    """
    script_path = tmp_path / "test_complex.noeta"
    script_content = f'''load "{sample_csv}" as employees
filter employees where age > 25 as adults
groupby adults by department agg mean as dept_stats
sort dept_stats by age desc as sorted_stats
describe sorted_stats'''

    script_path.write_text(script_content)
    return str(script_path)


@pytest.fixture
def sample_dataframe():
    """
    Return a pandas DataFrame for testing.

    Returns:
        pd.DataFrame: Sample DataFrame
    """
    return pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'c', 'd', 'e'],
        'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
    })


@pytest.fixture
def lexer_test_cases():
    """
    Return test cases for lexer testing.

    Returns:
        dict: Dictionary of test case name -> Noeta source code
    """
    return {
        'simple_load': 'load "data.csv" as sales',
        'select_with_columns': 'select sales with price, quantity as subset',
        'filter_where': 'filter sales where price > 100 as expensive',
        'groupby_agg': 'groupby sales by product agg sum as totals',
        'multiple_statements': '''load "data.csv" as sales
select sales with price as prices
describe prices''',
        'with_numbers': 'head sales with n=10',
        'with_strings': 'upper sales column name as upper_names',
        'with_booleans': 'fillna sales with value=0 inplace=false as filled'
    }


@pytest.fixture
def parser_test_cases():
    """
    Return test cases for parser testing.

    Returns:
        dict: Dictionary of test case name -> (source, expected_node_type)
    """
    return {
        'load': ('load "data.csv" as sales', 'LoadNode'),
        'select': ('select sales with price, quantity as subset', 'SelectNode'),
        'filter': ('filter sales where price > 100 as expensive', 'UpdatedFilterNode'),
        'describe': ('describe sales', 'DescribeNode'),
        'groupby': ('groupby sales by product agg sum as totals', 'GroupByNode'),
        'join': ('join sales with customers on id as combined', 'JoinNode'),
        'upper': ('upper sales column name as upper_names', 'UpperNode'),
        'fillna': ('fillna sales with value=0 as filled', 'FillNaNode')
    }


@pytest.fixture
def semantic_test_cases():
    """
    Return test cases for semantic analyzer testing.

    Returns:
        dict: Dictionary of test case name -> (source, should_error, error_message_contains)
    """
    return {
        'valid_simple': (
            'load "data.csv" as sales\nselect sales with price as prices',
            False,
            None
        ),
        'undefined_dataset': (
            'select unknown with price as result',
            True,
            'has not been loaded or created'
        ),
        'typo_suggestion': (
            'load "data.csv" as sales\nselect sale with price as result',
            True,
            'sales'  # Should suggest "sales"
        ),
        'valid_chain': (
            '''load "data.csv" as data
filter data where price > 100 as filtered
select filtered with name, price as result''',
            False,
            None
        ),
        'break_chain': (
            '''load "data.csv" as data
filter data where price > 100 as filtered
select data_filtered with name as result''',
            True,
            'data_filtered'
        )
    }
