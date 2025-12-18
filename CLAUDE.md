# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Noeta** is a production-ready Domain-Specific Language (DSL) for data analysis that compiles to Python/Pandas code. It provides an intuitive, natural language-like syntax for data manipulation, statistical analysis, and visualization tasks.

**Project Maturity**: Production Ready (67% of planned features implemented, 167/250 operations)
**Codebase Size**: ~9,100 lines of core implementation code + ~10,300 lines of documentation
**Last Major Update**: December 19, 2025 - **Multi-Error Reporting Added** ✅ (Dec 17: Semantic Validation)

**For comprehensive visual documentation of the entire system architecture and execution flow, see `FLOW_DIAGRAM.md` which contains 10 detailed Mermaid diagrams covering:**
- System architecture and component interactions
- Complete compilation pipeline with decision points
- CLI and Jupyter execution flows
- Symbol table and import management
- Error handling and operation categories

---

## Quick Reference for Common Tasks

| Task | Command | Location |
|------|---------|----------|
| Run a Noeta script | `python noeta_runner.py examples/demo_basic.noeta` | noeta_runner.py |
| Run inline Noeta code | `python noeta_runner.py -c 'load "data.csv" as d\ndescribe d'` | noeta_runner.py |
| Install Jupyter kernel | `python install_kernel.py` | install_kernel.py |
| Run basic tests | `python test_noeta.py` | test_noeta.py |
| View generated Python | Add `-v` flag or set `verbose=True` | noeta_runner.py |
| Add new operation | See "Adding New Operations" section below | All 4 core files |
| Check operation coverage | See `STATUS.md` | Documentation |
| View syntax examples | See `examples/` directory | 20+ example files |

---

## Core Architecture

Noeta follows a classic compiler pipeline architecture with **five main compilation stages**:

1. **Lexer** (`noeta_lexer.py`, 916 lines): Tokenizes Noeta source code into tokens
2. **Parser** (`noeta_parser.py`, 3,480 lines): Builds an Abstract Syntax Tree (AST) from tokens
3. **AST** (`noeta_ast.py`, 1,186 lines): Defines all AST node types using dataclasses
4. **Semantic Analyzer** (`noeta_semantic.py`, 1,717 lines): ⭐ **NEW!** Validates AST for semantic correctness
5. **Code Generator** (`noeta_codegen.py`, 1,795 lines): Converts AST to executable Python/Pandas code
6. **Runner** (`noeta_runner.py`, ~100 lines): Main execution script that orchestrates the compilation pipeline
7. **Kernel** (`noeta_kernel.py`, ~180 lines): Jupyter kernel implementation for notebook integration
8. **Error Handler** (`noeta_errors.py`, ~200 lines): Rich error formatting with position tracking

**Total Core Implementation**: 9,094 lines of code

### Compilation Pipeline

```
Noeta source → Lexer → Tokens → Parser → AST → SemanticAnalyzer → CodeGenerator → Python code → exec()
                                                      ↓
                                              Catch errors at
                                              compile-time! ✅
```

The code generator maintains a symbol table to track variable aliases and generates imports dynamically based on operations used.

### Code Statistics by Component

| Component | Lines | Token Types | AST Nodes | Parser Methods | Validator Methods | CodeGen Visitors |
|-----------|-------|-------------|-----------|----------------|-------------------|------------------|
| Lexer | 916 | 150+ | N/A | N/A | N/A | N/A |
| AST | 1,186 | N/A | 167 | N/A | N/A | N/A |
| Parser | 3,480 | N/A | N/A | 167+ | N/A | N/A |
| **Semantic** ⭐ | **1,717** | N/A | N/A | N/A | **138** | N/A |
| CodeGen | 1,795 | N/A | N/A | N/A | N/A | 167 |
| Runner | ~100 | N/A | N/A | N/A | N/A | N/A |
| Errors | ~200 | N/A | N/A | N/A | N/A | N/A |
| **TOTAL** | **~9,094** | **150+** | **167** | **167+** | **138** | **167** |

---

## Development Commands

### Running Noeta Scripts

Command line execution:
```bash
python noeta_runner.py examples/demo_basic.noeta
```

Inline code execution:
```bash
python noeta_runner.py -c 'load "data/sales_data.csv" as sales
describe sales'
```

Verbose mode (shows generated Python code):
```bash
python noeta_runner.py -c 'load "data.csv" as d
describe d' -v
```

### Testing

Basic functionality test:
```bash
python test_noeta.py
```

Run specific example:
```bash
python noeta_runner.py examples/test_phase11_all_26_operations.noeta
```

Run comprehensive test suite:
```bash
python noeta_runner.py examples/test_comprehensive_all_phases.noeta
```

### Jupyter Kernel

Install the Noeta Jupyter kernel:
```bash
python install_kernel.py
```

Start Jupyter and select "Noeta" kernel:
```bash
jupyter notebook
```

### Package Installation

Install Noeta as a package:
```bash
pip install -e .
```

Or install dependencies only:
```bash
pip install -r requirements.txt
```

---

## Language Syntax

Noeta uses a declarative syntax with operations that follow this general pattern:
```
<operation> <source> [<parameters>] as <alias>
```

### Complete Operation Categories (167 Total)

#### Data I/O (10 operations) - 100% Coverage ✅
- **Load**: `load`, `load_csv`, `load_json`, `load_excel`, `load_parquet`, `load_sql`
- **Save**: `save`, `save_csv`, `save_json`, `save_excel`, `save_parquet`

#### Selection & Projection (7 operations) - 100% Coverage ✅
- `select`, `select_by_type`, `head`, `tail`, `iloc`, `loc`, `rename`, `reorder`

#### Filtering (9 operations) - 100% Coverage ✅
- `filter`, `filter_between`, `filter_isin`, `filter_contains`, `filter_startswith`
- `filter_endswith`, `filter_regex`, `filter_null`, `filter_notnull`, `filter_duplicates`

#### Transformation (35 operations) - 100% Coverage ✅
**Math Operations (7)**
- `round`, `abs`, `sqrt`, `power`, `log`, `ceil`, `floor`

**String Operations (14)** - 88% Coverage
- `upper`, `lower`, `strip`, `replace`, `split`, `concat`, `substring`, `length`
- `title`, `capitalize`, `lstrip`, `rstrip`, `extract_regex`, `find`
- Missing: 8 advanced string operations

**Date/Time Operations (14)** - 93% Coverage
- `parse_datetime`, `extract_year`, `extract_month`, `extract_day`
- `extract_hour`, `extract_minute`, `extract_second`
- `extract_dayofweek`, `extract_dayofyear`, `extract_weekofyear`, `extract_quarter`
- `date_add`, `date_subtract`, `format_datetime`, `date_diff`
- Missing: 10 timezone and business day operations

#### Type & Encoding Operations (6 operations) - 100% Coverage ✅
- `astype`, `to_numeric`, `one_hot_encode`, `label_encode`, `ordinal_encode`, `target_encode`

#### Scaling & Normalization (4 operations) - 100% Coverage ✅
- `standard_scale`, `minmax_scale`, `robust_scale`, `maxabs_scale`

#### Cleaning Operations (13 operations) - 100% Coverage ✅
- `dropna`, `fillna`, `isnull`, `notnull`, `count_na`
- `fill_forward`, `fill_backward`, `fill_mean`, `fill_median`
- `interpolate`, `duplicated`, `count_duplicates`, `drop_duplicates`

#### Reshaping Operations (7 operations) - 100% Coverage ✅
- `pivot`, `melt`, `stack`, `unstack`, `transpose`, `explode`, `normalize`

#### Combining Operations (6 operations) - 100% Coverage ✅
- `join`, `merge`, `concat_vertical`, `concat_horizontal`, `append`, `cross_join`

#### Aggregation & Grouping (20 operations) - 85% Coverage
- `groupby`, `agg`, `sum`, `mean`, `median`, `min`, `max`, `count`, `std`, `var`
- `first`, `last`, `nth`, `nunique`, `quantile`, `rolling`, `expanding`
- Missing: 12 advanced aggregations (weighted mean, mode, skewness, kurtosis, etc.)

#### Apply/Map Operations (4 operations) - 100% Coverage ✅
- `apply`, `map`, `applymap`, `map_values`

#### Binning Operations (2 operations) - 100% Coverage ✅
- `binning`, `cut`

#### Cumulative Operations (4 operations) - 100% Coverage ✅
- `cumsum`, `cummax`, `cummin`, `cumprod`

#### Time Series Operations (3 operations) - 100% Coverage ✅
- `pct_change`, `diff`, `shift`

#### Validation Operations (3 operations) - 100% Coverage ✅
- `assert_unique`, `assert_no_nulls`, `assert_range`

#### Index Operations (5 operations) - 100% Coverage ✅
- `set_index`, `reset_index`, `sort_index`, `reindex`, `set_multiindex`

#### Boolean Operations (4 operations) - 100% Coverage ✅
- `any`, `all`, `count_true`, `compare`

#### Window Functions (14 operations) - 64% Coverage
- `rank`, `dense_rank`, `row_number`, `percent_rank`, `ntile`, `lag`, `lead`
- Missing: 8 advanced window operations

#### Statistical Operations (9 operations) - 47% Coverage
- `describe`, `summary`, `info`, `corr`, `cov`, `value_counts`, `unique`, `sample`
- Missing: 10 operations (hypothesis tests, regression, etc.)

#### Visualization Operations (5 operations) - 33% Coverage
- `boxplot`, `heatmap`, `pairplot`, `timeseries`, `pie`, `export_plot`
- Missing: 10 operations (scatter, line, bar, histogram, violin, etc.)

---

## Example Files (20+ Comprehensive Test Cases)

### Basic Examples
- `examples/demo_basic.noeta` - Simple load, select, filter workflow
- `examples/demo_advanced.noeta` - Advanced analysis with grouping and visualization

### Phase-Specific Test Files
- `examples/test_phase1_io.noeta` - Data I/O operations
- `examples/test_phase1_comprehensive.noeta` - Extended I/O test
- `examples/test_phase2_selection.noeta` - Selection & projection
- `examples/test_phase3_filtering.noeta` - Filtering operations
- `examples/test_phase4_math.noeta` - Math operations
- `examples/test_phase4_string.noeta` - String operations
- `examples/test_phase4_date.noeta` - Date/time operations
- `examples/test_phase4_type_encoding.noeta` - Type & encoding
- `examples/test_phase5_cleaning.noeta` - Cleaning operations

### Phase 11 & 12 Examples
- `examples/phase11_new_operations.noeta` - All 26 Phase 11 operations
- `examples/test_phase11_basic.noeta` - Basic Phase 11 test
- `examples/test_phase11_comprehensive.noeta` - Comprehensive Phase 11
- `examples/test_phase11_all_26_operations.noeta` - Complete Phase 11
- `examples/test_cumulative.noeta` - Cumulative operations
- `examples/test_new_ops_simple.noeta` - Simple integration test
- `examples/test_applymap_extract_regex.noeta` - Complex operations
- `examples/phase12_medium_priority_ops.noeta` - All 13 Phase 12 operations
- `examples/test_phase12_basic.noeta` - Basic Phase 12 test
- `examples/test_phase12_validation.noeta` - Validation operations

### Comprehensive Tests
- `examples/test_comprehensive_all_phases.noeta` - Full integration test

---

## Key Implementation Details

### Symbol Table Management

The code generator maintains `self.symbol_table` to track DataFrame aliases. When a new alias is created (via `as <name>`), it's registered in this table. This allows Noeta code to reference DataFrames by their aliases across statements.

```python
# In noeta_codegen.py
self.symbol_table[alias] = variable_name
```

### Import Management

The code generator uses `self.imports` (a set) to collect necessary imports dynamically based on operations used. Standard imports (pandas, numpy, matplotlib, seaborn, scipy) are added by default in `CodeGenerator.generate()`.

Dynamic imports are added for:
- **scikit-learn**: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`, `LabelEncoder`
- **numpy**: Math operations (`sqrt`, `power`, `log`, `ceil`, `floor`, trigonometric functions)
- **pandas**: All DataFrame operations
- **matplotlib/seaborn**: Visualization operations
- **scipy**: Statistical tests and advanced analysis

### Visualization Handling

The generator tracks whether visualization operations have been used via `self.last_plot`. If any plot operations are detected, it automatically adds `plt.show()` at the end of generated code. The Jupyter kernel has special handling to capture matplotlib figures and display them as PNG images.

### Parser State

The parser maintains position state (`self.pos`) and provides helper methods:
- `current_token()`: Returns token at current position
- `peek_token(offset)`: Looks ahead without advancing
- `expect(token_type)`: Consumes and validates expected token type
- `match(*token_types)`: Checks if current token matches any of the given types
- `parse_value()`: Parses literal values (strings, numbers, booleans, None)
- `parse_list_value()`: Parses Python lists
- `parse_dict_value()`: Parses Python dictionaries

### Error Handling

Noeta provides production-quality error handling with comprehensive compile-time validation:

#### Multi-Error Reporting (NEW: December 19, 2025) ✅

The compiler now shows **all errors at once** instead of stopping at the first error:

```python
# In noeta_runner.py
if errors:
    if len(errors) == 1:
        raise errors[0]  # Single error - use normal format
    else:
        raise create_multi_error(errors)  # Multiple errors - grouped format
```

**Features**:
- Shows all semantic errors in one pass
- Groups errors by category (Lexical, Syntax, Semantic, Type)
- Numbered error list for easy tracking
- Each error shows: line, column, source context, message, hint, suggestion
- Color-coded terminal output with ANSI colors

**Implementation**:
- `MultiErrorFormatter` class in `noeta_errors.py` handles formatting
- `create_multi_error()` function combines multiple errors into single exception
- See `examples/test_multi_error_reporting.noeta` for demonstration

#### Error Infrastructure

**Files**:
- `noeta_errors.py`: Error classes, formatters, and utilities
  - `NoetaError`: Base error class with rich formatting
  - `ErrorCategory`: Enum for error types (LEXER, SYNTAX, SEMANTIC, TYPE, RUNTIME)
  - `ErrorContext`: Line/column/source tracking
  - `ErrorFormatter`: Single error formatting with color support
  - `MultiErrorFormatter`: Multiple error formatting (NEW)
  - Suggestion functions: `suggest_similar()`, `levenshtein_distance()`

**Error Categories**:
1. **Lexical Errors**: Invalid characters, unterminated strings
2. **Syntax Errors**: Grammar violations, unexpected tokens
3. **Semantic Errors**: Undefined datasets, invalid references (caught at compile-time!)
4. **Type Errors**: Type mismatches, invalid operations
5. **Runtime Errors**: Execution failures

**Verbose Mode**:
The runner script provides verbose output by default, showing:
1. Generated Python code
2. Execution output

This helps with debugging and understanding the compilation process.

---

## File Structure

### Core Compilation Components
- `noeta_lexer.py` (916 lines): Defines `TokenType` enum and `Lexer` class with regex-based tokenization
- `noeta_parser.py` (3,480 lines): Contains `Parser` class with methods like `parse_statement()` for each operation type (recursive descent parser)
- `noeta_ast.py` (1,186 lines): AST node definitions (all dataclasses inheriting from `ASTNode`)
- `noeta_codegen.py` (1,795 lines): `CodeGenerator` class with visitor pattern methods (`visit_<NodeType>`)

### Execution Interfaces
- `noeta_runner.py`: CLI entry point with `compile_noeta()` and `execute_noeta()` functions
  - Supports file execution: `python noeta_runner.py script.noeta`
  - Supports inline execution: `python noeta_runner.py -c 'noeta code'`
  - Verbose mode shows generated Python code
- `noeta_kernel.py`: `NoetaKernel` class extending `ipykernel.kernelbase.Kernel`
  - Handles cell execution in Jupyter notebooks
  - Provides code completion and inspection
  - Captures matplotlib figures for inline display (PNG base64 encoding)

### Setup and Testing
- `install_kernel.py`: Registers Jupyter kernel specification with kernel.json
- `setup.py`: Package setup script for installation
- `test_noeta.py`: Basic functionality tests
- `requirements.txt`: Python package dependencies

### Build Artifacts
- `build/`: Package build directory (created by setup.py)
- `noeta.egg-info/`: Package metadata
- `output/`: Generated output files from examples

### Examples and Data
- `examples/` (20+ files): Demo `.noeta` scripts demonstrating language features
- `data/`: Sample CSV files for testing operations (sales_data.csv, etc.)

### Documentation (10,327 total lines)
- `CLAUDE.md` (444 lines): This file - development guidance and architecture overview
- `README.md` (39 lines): User-facing quick start documentation
- `FLOW_DIAGRAM.md` (829 lines): Visual flow diagrams of system architecture and execution (10 Mermaid diagrams)
- `DATA_MANIPULATION_REFERENCE.md` (3,220 lines, 92KB): Comprehensive reference for all 167 data manipulation operations
- `DATA_ANALYSIS_REFERENCE.md` (2,131 lines, 82KB): Exhaustive reference for data analysis functions (9/350 functions documented)
- `NOETA_COMMAND_REFERENCE.md` (901 lines): Command syntax reference
- `CURRENT_STATUS.md` (334 lines): Implementation status tracking (167/250 operations, 67% coverage, production ready)
- `IMPLEMENTATION_PROGRESS.md` (317 lines): Implementation tracking across all phases
- `IMPLEMENTATION_SUMMARY.md` (488 lines): Summary of implemented operations
- `PHASE11_COMPLETION_SUMMARY.md` (364 lines): Phase 11 completion details
- `PHASE12_COMPLETION_SUMMARY.md` (400 lines): Phase 12 completion details
- `PHASE11_VERIFICATION_REPORT.md`: Verification and testing results
- `REMAINING_GAPS.md` (307 lines): Analysis of 83 unimplemented operations
- `DEMO_GUIDE.md`: Interactive demo guide

---

## Adding New Operations

To add a new operation to Noeta, follow these steps:

### 1. Add Token to Lexer (`noeta_lexer.py`)
```python
# In TokenType enum
class TokenType(Enum):
    # ... existing tokens ...
    MY_OPERATION = "MY_OPERATION"

# In Lexer.__init__() keywords dictionary
self.keywords = {
    # ... existing keywords ...
    'my_operation': TokenType.MY_OPERATION,
}
```

### 2. Create AST Node (`noeta_ast.py`)
```python
@dataclass
class MyOperationNode(ASTNode):
    """Represents a my_operation statement."""
    source_alias: str  # ⚠️ Use source_alias, not source!
    column: str  # if column-based
    new_alias: str  # ⚠️ Use new_alias, not alias!
    param1: Optional[str] = None
    param2: Optional[int] = None
```

**⚠️ IMPORTANT - Naming Convention:**
- Use `source_alias` for the input dataset (NOT `source`)
- Use `new_alias` for the output dataset (NOT `alias`)
- For operations with two inputs: `alias1`, `alias2` (e.g., JoinNode)
- For merge operations: `left_alias`, `right_alias` (e.g., MergeNode)

### 3. Add Parser Method (`noeta_parser.py`)
```python
def parse_my_operation(self):
    """Parse: my_operation <source> column <column> param1=<value> as <alias>"""
    self.expect(TokenType.MY_OPERATION)

    source_token = self.expect(TokenType.IDENTIFIER)
    source = source_token.value

    self.expect(TokenType.COLUMN)
    column_token = self.expect(TokenType.IDENTIFIER)
    column = column_token.value

    # Parse optional parameters
    param1 = None
    if self.match(TokenType.PARAM1):
        self.pos += 1
        self.expect(TokenType.EQUALS)
        param1 = self.parse_value()

    # Parse alias
    alias = None
    if self.match(TokenType.AS):
        self.pos += 1
        alias_token = self.expect(TokenType.IDENTIFIER)
        alias = alias_token.value

    return MyOperationNode(source, column, param1, alias)

# Add to parse_statement() dispatcher
def parse_statement(self):
    # ... existing cases ...
    elif self.match(TokenType.MY_OPERATION):
        return self.parse_my_operation()
```

### 4. Add Code Generator Visitor (`noeta_codegen.py`)
```python
def visit_MyOperationNode(self, node):
    """Generate code for my_operation."""
    df_var = self.symbol_table.get(node.source_alias)  # Use source_alias
    if not df_var:
        raise ValueError(f"Unknown dataframe: {node.source_alias}")

    # Generate pandas code
    code = f"{df_var}['{node.column}'].my_pandas_method("

    if node.param1:
        code += f"param1={repr(node.param1)}"

    code += ")"

    # Handle alias
    if node.new_alias:  # Use new_alias
        new_var = f"df_{len(self.symbol_table)}"
        self.code.append(f"{new_var} = {code}")
        self.symbol_table[node.new_alias] = new_var
        self.code.append(f"print(f'Applied my_operation to {node.source_alias}')")
    else:
        self.code.append(f"print({code})")
```

### 5. Add Semantic Validator (`noeta_semantic.py`) ⭐ **REQUIRED!**
```python
def visit_MyOperationNode(self, node):
    """Validate my_operation operation."""
    # 1. Check source dataset exists
    source_info = self._check_dataset_exists(node.source_alias, node)

    # 2. Validate column exists (if applicable)
    if hasattr(node, 'column') and node.column:
        self._check_column_exists(source_info, node.column, node)

    # 3. Check column type (if applicable)
    # if hasattr(node, 'column') and node.column:
    #     self._check_column_type(source_info, node.column, DataType.NUMERIC, node)

    # 4. Register result dataset (if creates new dataset)
    if node.new_alias:
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"my_operation from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)
```

**Why semantic validation is critical:**
- ✅ Catches undefined dataset references at **compile-time** (not runtime)
- ✅ Provides helpful error messages with suggestions
- ✅ Maintains symbol table for tracking all datasets
- ✅ Enables better IDE support and error detection in the future

### 6. Add Tests
Create test file in `examples/test_my_operation.noeta`:
```noeta
load csv "data/sales_data.csv" as sales
my_operation sales column price param1="value" as result
describe result
```

### 7. Update Documentation
- Update `STATUS.md` coverage metrics and implementation details
- Add syntax to `NOETA_COMMAND_REFERENCE.md`

---

## Quick Checklist for New Operations

When adding a new operation, ensure ALL 5 core files are updated:

1. ✅ `noeta_lexer.py` - Token + keyword
2. ✅ `noeta_ast.py` - AST node (use `source_alias`, `new_alias`!)
3. ✅ `noeta_parser.py` - Parse method + dispatcher
4. ✅ `noeta_codegen.py` - Code generator visitor
5. ✅ `noeta_semantic.py` - **Semantic validator (REQUIRED!)** ⭐

**Common mistake:** Forgetting to add the semantic validator! This will cause undefined dataset errors to only be caught at runtime instead of compile-time.

### ⚠️ IMPORTANT: Adding New Modules

**When creating a new Python module (e.g., `noeta_errors.py`, `noeta_semantic.py`), you MUST update `setup.py`:**

Add the module name to the `py_modules` list:
```python
py_modules=[
    'noeta_lexer',
    'noeta_parser',
    'noeta_ast',
    'noeta_codegen',
    'noeta_runner',
    'noeta_kernel',
    'noeta_errors',      # <-- Add new modules here
    'install_kernel',
    'test_noeta'
],
```

Then reinstall the package:
```bash
pip install -e . --force-reinstall --no-deps
```

**Why this matters:**
- Without updating `setup.py`, the new module won't be included in the package
- This causes `ModuleNotFoundError` when using the `noeta` command
- Always update setup.py BEFORE testing with the installed package

---

## Dependencies

Required Python packages (from `requirements.txt`):
```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
jupyter
ipykernel
```

### Version Compatibility
- Python: 3.7+
- Pandas: 1.x or 2.x (code generator handles both versions)
- NumPy: Any recent version
- Matplotlib: 3.x
- Seaborn: 0.11+
- Scikit-learn: 1.0+

---

## Implementation Notes

### Lexer Details
- Uses regex patterns for token matching (keywords, identifiers, strings, numbers, operators)
- Maintains position tracking for error reporting
- Keywords are case-insensitive for user convenience
- Total token types: 150+
- Handles strings (single/double quotes), numbers (int/float), booleans, None, lists, dicts

### Parser Details
- Implements recursive descent parsing (LL(1) grammar)
- Provides helper methods: `current_token()`, `peek_token()`, `expect()`, `match()`
- Each operation has dedicated `parse_<operation>()` method
- Returns list of AST nodes representing the program
- Total parser methods: 167+
- Supports complex parameter parsing (lists, dicts, nested values)

### Code Generator Details
- Uses visitor pattern (`visit_<NodeName>()` methods)
- Maintains `self.symbol_table` dict for alias tracking
- Maintains `self.imports` set for dynamic import collection
- Tracks `self.last_plot` boolean to determine if `plt.show()` is needed
- Generates idiomatic pandas code with proper method chaining
- Total visitor methods: 167
- Handles pandas version compatibility (e.g., `applymap` vs `map`)

### Execution Details
- Generated Python code uses pandas DataFrame operations
- Visualization uses matplotlib/seaborn with `seaborn-v0_8-darkgrid` style
- File paths in Noeta code can be relative or absolute
- The system uses `exec()` to run generated Python code in a controlled namespace
- CLI execution uses `globals()` namespace
- Jupyter kernel maintains persistent `self.namespace` dict across cell executions

### Jupyter Integration
- Kernel installed via `install_kernel.py` which creates kernel spec in Jupyter data directory
- Implements `do_execute()` for code execution
- Implements `do_complete()` for tab completion
- Implements `do_inspect()` for shift+tab help
- Captures matplotlib figures before display, converts to PNG, encodes as base64
- Returns execution results as `execute_reply` with status and execution_count

---

## Debugging and Development

### Verbose Mode
The runner script can show generated Python code:
```bash
python noeta_runner.py examples/demo.noeta -v
```

Or programmatically:
```python
execute_noeta(source_code, verbose=True)
```

This is invaluable for:
- Understanding how Noeta compiles to Python
- Debugging compilation issues
- Learning pandas operations
- Verifying generated code correctness

### Testing New Operations
When adding new operations:
1. Start with simple test case in `examples/test_*.noeta`
2. Run with verbose mode to inspect generated code
3. Test in Jupyter notebook for interactive development
4. Add to comprehensive test suite
5. Update documentation files

### Common Issues and Solutions

#### Symbol not found
- **Issue**: `ValueError: Unknown dataframe: mydata`
- **Solution**: Check if alias was registered in symbol table during load/transformation
- **Debug**: Enable verbose mode to see symbol table state

#### Import missing
- **Issue**: `NameError: name 'StandardScaler' is not defined`
- **Solution**: Verify code generator adds necessary import to `self.imports`
- **Fix**: Add `self.imports.add('from sklearn.preprocessing import StandardScaler')` in visitor method

#### Syntax error in generated code
- **Issue**: Generated Python code has syntax errors
- **Solution**: Check parser expects tokens in correct order
- **Debug**: Run with verbose mode and inspect generated code

#### Visualization not showing
- **Issue**: Plot operations execute but don't display
- **Solution**: Ensure `self.last_plot` is set to True in code generator
- **Fix**: Add `self.last_plot = True` in visualization visitor methods

#### Token not recognized
- **Issue**: Lexer doesn't recognize keyword
- **Solution**: Verify keyword is added to `self.keywords` dictionary in lexer
- **Fix**: Add keyword mapping (case-insensitive)

#### Parser parameter errors
- **Issue**: Parser fails on optional parameters
- **Solution**: Use `self.match()` to check before parsing optional params
- **Fix**: Add proper `if self.match(TokenType.PARAM):` checks

---

## Architecture Patterns

### Separation of Concerns
- **Lexer**: Only responsible for character → token conversion
- **Parser**: Only responsible for token → AST conversion
- **CodeGen**: Only responsible for AST → Python code conversion
- **Runner/Kernel**: Only responsible for orchestration and execution

This clean separation makes the codebase maintainable and testable.

### Extensibility
- New operations require additions to all four components (lexer, parser, AST, codegen)
- Each component has clear extension point (keyword mapping, parse method, AST class, visitor method)
- No changes needed to runner or kernel for new operations (unless special execution handling required)

### Data Flow
```
String → Tokens → AST → Python String → Executed Code → Output
```

Each stage produces complete output before next stage begins (no streaming/incremental compilation)

### Design Patterns Used
1. **Visitor Pattern**: Code generator traverses AST using visitor methods
2. **Dataclass Pattern**: All AST nodes use Python dataclasses for clean structure
3. **Recursive Descent**: Parser uses top-down recursive descent algorithm
4. **Symbol Table**: Tracks variable bindings across compilation
5. **Template Method**: Base AST node defines interface for all nodes

---

## Current Project Status (December 12, 2025)

### Implementation Overview

**Total Operations**: 167/250 (67% coverage) - ✅ **PRODUCTION READY**
**Total Code**: 7,377 lines (core implementation)
**Total Documentation**: 10,327 lines across 14 files

**Completed Phases**:
- ✅ **Phase 1-10**: 128 foundational operations (data I/O, selection, filtering, cleaning, transformation, aggregation)
- ✅ **Phase 11**: 26 high-priority operations (cumulative, time series, date/time, string ops, binning)
- ✅ **Phase 12**: 13 medium-priority operations (scaling, encoding, validation, boolean ops)

### Coverage by Category (Detailed)

| Category | Implemented | Total | Coverage | Status |
|----------|-------------|-------|----------|--------|
| Data I/O | 10 | 10 | 100% | ✅ Complete |
| Selection & Projection | 7 | 7 | 100% | ✅ Complete |
| Filtering | 9 | 9 | 100% | ✅ Complete |
| Cleaning | 13 | 13 | 100% | ✅ Complete |
| Reshaping | 7 | 7 | 100% | ✅ Complete |
| Combining | 6 | 6 | 100% | ✅ Complete |
| Binning | 2 | 2 | 100% | ✅ Complete |
| Apply/Map | 4 | 4 | 100% | ✅ Complete |
| Cumulative | 4 | 4 | 100% | ✅ Complete |
| Time Series | 3 | 3 | 100% | ✅ Complete |
| Scaling | 4 | 4 | 100% | ✅ Complete |
| Encoding | 6 | 6 | 100% | ✅ Complete |
| Validation | 3 | 3 | 100% | ✅ Complete |
| Index Ops | 5 | 5 | 100% | ✅ Complete |
| Boolean Ops | 4 | 4 | 100% | ✅ Complete |
| String Operations | 14 | 22 | 64% | ⚠️ 8 missing |
| Date/Time | 14 | 24 | 58% | ⚠️ 10 missing |
| Math | 7 | 13 | 54% | ⚠️ 6 trig missing |
| Aggregation | 20 | 32 | 63% | ⚠️ 12 missing |
| Window Functions | 14 | 22 | 64% | ⚠️ 8 missing |
| Statistics | 9 | 19 | 47% | ⚠️ 10 missing |
| Visualization | 5 | 15 | 33% | ⚠️ 10 missing |
| **TOTAL** | **167** | **250** | **67%** | **✅ Production** |

### Production Ready For
- ✅ Standard data manipulation tasks
- ✅ Time series analysis and forecasting
- ✅ Date/time data processing
- ✅ String data cleaning and extraction
- ✅ Data aggregation and grouping
- ✅ ETL pipelines
- ✅ Business intelligence reports
- ✅ Machine learning preprocessing
- ✅ Data validation and quality checks
- ✅ Exploratory data analysis
- ✅ Feature engineering

### Current Limitations
- ⚠️ Advanced statistical testing (need Phase 13+)
- ⚠️ Complex timezone operations (need Phase 13)
- ⚠️ Advanced visualization types (need Phase 15)
- ⚠️ Trigonometric functions (need Phase 13)
- ⚠️ Memory optimization for huge datasets (need Phase 14)

### Remaining Gaps (83 low-priority operations)
See `STATUS.md` for detailed analysis of remaining gaps:
- Trigonometric functions (6 ops)
- Advanced string operations (8 ops)
- Additional date/time operations (10 ops)
- Advanced aggregations (12 ops)
- Window functions (8 ops)
- Reshaping operations (5 ops)
- Advanced merge operations (7 ops)
- Memory & performance (5 ops)
- Partitioning (2 ops)
- Statistical operations (10 ops)
- Visualization operations (10 ops)

---

## Documentation Structure

### Implementation Documentation
1. **STATUS.md** (~850 lines) - Single Source of Truth
   - Consolidated implementation status
   - Coverage analysis by category (167/250 operations, 67%)
   - Phase-wise completion overview (Phases 1-12)
   - Remaining gaps analysis (83 operations)
   - Production readiness assessment
   - Implementation roadmap (Phases 13-15)
   - Test results and code statistics

2. **DOCUMENTATION_MAP.md** (~400 lines) - Master Index
   - Complete documentation structure
   - Quick navigation for different user types
   - Document purposes and relationships
   - Update schedule and maintenance guidelines

### Phase-Specific Documentation
5. **PHASE11_COMPLETION_SUMMARY.md** (364 lines)
   - Details of 26 high-priority operations
   - Implementation statistics
   - Syntax reference and examples
   - Bug fixes and improvements

6. **PHASE12_COMPLETION_SUMMARY.md** (400 lines)
   - Details of 13 medium-priority operations
   - Coverage improvements
   - Technical implementation details
   - Test results

7. **PHASE11_VERIFICATION_REPORT.md**
   - Comprehensive verification results
   - Test execution logs
   - Code quality checks

### Reference Documentation
8. **DATA_MANIPULATION_REFERENCE.md** (3,220 lines, 92KB)
   - Comprehensive reference for all 167 operations
   - Organized in 10 parts
   - Syntax, parameters, examples for each operation
   - Pandas equivalents and best practices

9. **DATA_ANALYSIS_REFERENCE.md** (2,131 lines, 82KB)
   - Exhaustive statistical analysis reference
   - 350 planned functions across 45 parts
   - Currently 9 functions fully documented
   - Each function: ~2,000-2,500 words with 10 sections
   - Mathematical specifications and formulas
   - Statistical properties and assumptions
   - Interpretation guidelines
   - Real-world use cases

10. **NOETA_COMMAND_REFERENCE.md** (901 lines)
    - Quick syntax reference
    - Command patterns and examples
    - Parameter specifications

### Architecture Documentation
11. **FLOW_DIAGRAM.md** (829 lines)
    - 10 detailed Mermaid diagrams
    - System architecture overview
    - Compilation pipeline flow
    - CLI and Jupyter execution flows
    - Symbol table management
    - Import management
    - Error handling
    - Operation categories

12. **CLAUDE.md** (this file, 444+ lines)
    - Development guidance
    - Architecture overview
    - Implementation patterns
    - Debugging guide

### User Documentation
13. **README.md** (39 lines)
    - Quick start guide
    - Installation instructions
    - Basic usage examples

14. **DEMO_GUIDE.md**
    - Interactive demo walkthrough
    - Step-by-step tutorials

---

## Data Analysis Reference Progress

**File**: `DATA_ANALYSIS_REFERENCE.md` (82KB, 2131 lines)

**Scope**: 350 statistical functions planned across 45 parts

**Current Progress**: 9/350 functions documented (2.6%)

### Completed Parts
- ✅ **Part 1: Central Tendency Measures** (3 functions)
  - Mean, Median, Mode

- ✅ **Part 2: Dispersion Measures** (6 functions)
  - Variance, Standard Deviation, MAD, IQR, Range, CV

### Documentation Format (per function)
Each function documented with ~2,000-2,500 words covering:
1. Purpose and overview
2. Mathematical specification with formulas
3. Syntax variations (5+ examples)
4. Complete parameter specifications
5. Return value details
6. Statistical properties
7. Statistical assumptions
8. Interpretation guidelines
9. Common use cases (5+ real-world examples)
10. Related functions cross-references

### Remaining Documentation (341 functions)

**Parts 3-10: Traditional Statistical Methods**
- Shape Measures, Summary Stats, Correlation, Hypothesis Testing
- Distribution Analysis, Multiple Comparisons
- Total: 65 functions

**Parts 13-23: Regression & Time Series**
- Linear/Non-linear Regression, Time Series Analysis
- Total: 65 functions

**Parts 24-33: Advanced Statistical Methods**
- Survival Analysis, Bayesian Methods, Multivariate Analysis
- Dimensionality Reduction, Clustering, Classification
- Total: 92 functions

**Parts 34-45: Modern & Domain-Specific Methods** (NEW)
- Spatial Statistics, Network Analysis, Conformal Prediction
- Causal Inference, Model Interpretability, Functional Data
- Text Analytics, Data Quality, High-Dimensional Stats
- Total: 119 functions

### Python Packages Referenced
- **Core Statistics**: scipy.stats, statsmodels, pingouin
- **Machine Learning**: scikit-learn, lightgbm, xgboost
- **Causal Inference**: DoWhy, EconML, CausalML
- **Interpretability**: SHAP, LIME, InterpretML, Alibi
- **Spatial**: PySAL, geopandas
- **Network**: NetworkX, igraph
- **Text**: spaCy, NLTK, gensim, textstat
- **Time Series**: prophet, pmdarima, darts
- **Data Quality**: ydata-profiling, great_expectations

---

## Recent Development Activity

### December 2, 2025: Completed Phase 11 and Phase 12
- ✅ Added 39 new operations to Noeta DSL (26 + 13)
- ✅ Comprehensive testing with 10+ test files
- ✅ Documentation: Phase completion summaries and verification reports
- ✅ Bug fixes: 6 parser and code generation issues resolved
- ✅ Coverage improvement: 51% → 67% (+16%)

### December 6, 2025: DATA_ANALYSIS_REFERENCE.md Major Update
- ✅ Reorganized entire document with proper TOC structure
- ✅ Expanded scope from 182 to 350 planned functions
- ✅ Identified 11 new parts for modern statistical methods (Parts 34-45)
- ✅ Completed comprehensive documentation for Part 2 (6 dispersion functions)
- ✅ Total: 9 functions fully documented with exhaustive detail
- ✅ File size: 82KB, 2131 lines
- ✅ Updated all project documentation files

### December 12, 2025: Comprehensive CLAUDE.md Update
- ✅ Updated project status and statistics
- ✅ Added comprehensive code metrics (7,377 lines)
- ✅ Documented all 20+ example files
- ✅ Added detailed coverage breakdown
- ✅ Expanded debugging and troubleshooting sections
- ✅ Added quick reference tables
- ✅ Updated documentation structure overview

---

## Future Enhancement Opportunities

### High Priority Enhancements
1. **Complete Remaining Operations** (83 ops)
   - Phases 13-15 implementation
   - Estimated: 3-6 weeks
   - Would bring coverage to 100%

2. **Complete Statistical Documentation**
   - Document remaining 341 functions
   - Estimated: 12-16 weeks
   - Would create comprehensive statistical reference

3. **Error Messages with Line/Column Info**
   - Add position tracking throughout pipeline
   - Show helpful error messages with source location
   - Estimated: 1-2 weeks

### Medium Priority Enhancements
4. **Static Type Checking**
   - Add semantic analysis phase
   - Catch type errors before code generation
   - Verify DataFrame operations are valid
   - Estimated: 2-3 weeks

5. **Query Optimization**
   - AST transformation passes
   - Optimize pandas operations (predicate pushdown, column pruning)
   - Estimated: 2-3 weeks

6. **Language Server Protocol (LSP)**
   - Full IDE integration
   - VSCode extension with syntax highlighting
   - IntelliSense for operations and parameters
   - Estimated: 4-6 weeks

### Low Priority Enhancements
7. **Incremental Compilation**
   - Cache compiled results
   - Faster re-execution
   - Estimated: 1-2 weeks

8. **Performance Profiling**
   - Built-in timing and memory usage tracking
   - Optimization suggestions
   - Estimated: 1-2 weeks

9. **Streaming Support**
   - Handle datasets larger than memory
   - Chunk-based processing
   - Estimated: 3-4 weeks

10. **Alternative Backends**
    - Export to SQL for database execution
    - DuckDB backend for faster analytics
    - Estimated: 4-6 weeks

11. **Macro System**
    - User-defined operation compositions
    - Reusable workflow patterns
    - Estimated: 2-3 weeks

12. **Documentation Generation**
    - Auto-generate docs from Noeta scripts
    - Data lineage visualization
    - Estimated: 2-3 weeks

---

## Next Steps for Contributors

### If You Want to Add Operations
1. Choose from remaining 83 operations in `STATUS.md`
2. Follow "Adding New Operations" guide above
3. Start with Phase 13 (trigonometric, advanced strings, date/time)
4. Test thoroughly with example files
5. Update all documentation

### If You Want to Improve Documentation
1. Continue `DATA_ANALYSIS_REFERENCE.md` documentation
2. Start with Part 3: Shape Measures (Skewness, Kurtosis, Moments)
3. Follow existing format (10 sections per function)
4. Each function: 2,000-2,500 words
5. Include mathematical rigor and real-world examples

### If You Want to Enhance Architecture
1. Consider error messages with line/column info (high impact)
2. Or implement semantic analysis for type checking
3. Or build LSP for IDE integration
4. See "Future Enhancement Opportunities" above

---

## Support and Resources

### Documentation Files
- **Architecture**: `FLOW_DIAGRAM.md`, `CLAUDE.md`
- **Status**: `STATUS.md` (single source of truth), `DOCUMENTATION_MAP.md` (master index)
- **Reference**: `DATA_MANIPULATION_REFERENCE.md`, `NOETA_COMMAND_REFERENCE.md`, `SYNTAX_BLUEPRINT.md`
- **Archive**: `docs/archive/` (historical phase completion summaries)
- **Phases**: `PHASE11_COMPLETION_SUMMARY.md`, `PHASE12_COMPLETION_SUMMARY.md`
- **Analysis**: `DATA_ANALYSIS_REFERENCE.md` (statistical functions)

### Example Files
- Basic: `examples/demo_basic.noeta`, `examples/demo_advanced.noeta`
- Testing: `examples/test_*.noeta` (20+ test files)
- Phase-specific: `examples/phase11_*.noeta`, `examples/phase12_*.noeta`
- Comprehensive: `examples/test_comprehensive_all_phases.noeta`

### Quick Command Reference
```bash
# Run example
python noeta_runner.py examples/demo_basic.noeta

# Run with verbose output
python noeta_runner.py examples/demo_basic.noeta -v

# Run inline code
python noeta_runner.py -c 'load "data.csv" as d; describe d'

# Install Jupyter kernel
python install_kernel.py

# Run tests
python test_noeta.py

# Install package
pip install -e .
```

---

**Last Updated**: December 17, 2025
**Project Status**: ✅ PRODUCTION READY with Semantic Validation
**Coverage**: 167/250 operations (67%)
**Codebase**: 9,094 lines of implementation + 10,327 lines of documentation
**New Feature**: ✅ Compile-time semantic validation (138 validators)
**Maintainer**: Claude Code
