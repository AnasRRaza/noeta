# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Noeta** is a Domain-Specific Language (DSL) for data analysis that compiles to Python/Pandas code. It provides an intuitive, natural language-like syntax for data manipulation, statistical analysis, and visualization tasks.

**For comprehensive visual documentation of the entire system architecture and execution flow, see `FLOW_DIAGRAM.md` which contains 10 detailed Mermaid diagrams covering:**
- System architecture and component interactions
- Complete compilation pipeline with decision points
- CLI and Jupyter execution flows
- Symbol table and import management
- Error handling and operation categories

## Core Architecture

Noeta follows a classic compiler pipeline architecture:

1. **Lexer** (`noeta_lexer.py`): Tokenizes Noeta source code into tokens
2. **Parser** (`noeta_parser.py`): Builds an Abstract Syntax Tree (AST) from tokens
3. **AST** (`noeta_ast.py`): Defines all AST node types using dataclasses
4. **Code Generator** (`noeta_codegen.py`): Converts AST to executable Python/Pandas code
5. **Runner** (`noeta_runner.py`): Main execution script that orchestrates the compilation pipeline
6. **Kernel** (`noeta_kernel.py`): Jupyter kernel implementation for notebook integration

### Compilation Pipeline

```
Noeta source → Lexer → Tokens → Parser → AST → CodeGenerator → Python code → exec()
```

The code generator maintains a symbol table to track variable aliases and generates imports dynamically based on operations used.

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

### Testing

Basic functionality test:
```bash
python test_noeta.py
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

## Language Syntax

Noeta uses a declarative syntax with operations that follow this general pattern:
```
<operation> <source> [<parameters>] as <alias>
```

Key operation categories:
- **Data Loading**: `load`, `save`
- **Transformation**: `select`, `filter`, `sort`, `mutate`, `apply`
- **Aggregation**: `groupby`, `sample`
- **Cleaning**: `dropna`, `fillna`
- **Joining**: `join`
- **Analysis**: `describe`, `summary`, `info`, `outliers`, `quantile`, `normalize`, `binning`, `rolling`, `hypothesis`
- **Visualization**: `boxplot`, `heatmap`, `pairplot`, `timeseries`, `pie`, `export_plot`

## Key Implementation Details

### Symbol Table Management

The code generator maintains `self.symbol_table` to track DataFrame aliases. When a new alias is created (via `as <name>`), it's registered in this table. This allows Noeta code to reference DataFrames by their aliases across statements.

### Import Management

The code generator uses `self.imports` (a set) to collect necessary imports dynamically based on operations used. Standard imports (pandas, numpy, matplotlib, seaborn, scipy) are added by default in `CodeGenerator.generate()`.

### Visualization Handling

The generator tracks whether visualization operations have been used via `self.last_plot`. If any plot operations are detected, it automatically adds `plt.show()` at the end of generated code. The Jupyter kernel has special handling to capture matplotlib figures and display them as PNG images.

### Parser State

The parser maintains position state (`self.pos`) and provides helper methods:
- `current_token()`: Returns token at current position
- `peek_token(offset)`: Looks ahead without advancing
- `expect(token_type)`: Consumes and validates expected token type
- `match(*token_types)`: Checks if current token matches any of the given types

### Error Handling

The runner script provides verbose output by default, showing:
1. Generated Python code
2. Execution output

This helps with debugging and understanding the compilation process.

## File Structure

### Core Compilation Components
- `noeta_lexer.py`: Defines `TokenType` enum and `Lexer` class with regex-based tokenization
- `noeta_parser.py`: Contains `Parser` class with methods like `parse_statement()` for each operation type (recursive descent parser)
- `noeta_ast.py`: AST node definitions (all dataclasses inheriting from `ASTNode`)
- `noeta_codegen.py`: `CodeGenerator` class with visitor pattern methods (`visit_<NodeType>`)

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

### Examples and Data
- `examples/`: Demo `.noeta` scripts demonstrating language features
- `data/`: Sample CSV files for testing operations

### Documentation
- `CLAUDE.md`: This file - development guidance and architecture overview
- `FLOW_DIAGRAM.md`: Visual flow diagrams of system architecture and execution
- `README.md`: User-facing documentation

## Adding New Operations

To add a new operation to Noeta:

1. Add token type to `TokenType` enum in `noeta_lexer.py`
2. Add keyword to lexer's keyword mapping
3. Create AST node class in `noeta_ast.py`
4. Add parsing logic in `noeta_parser.py` (create `parse_<operation>()` method)
5. Add visitor method in `noeta_codegen.py` (`visit_<NodeName>()`)
6. Update kernel completion in `noeta_kernel.py` if needed

## Dependencies

Required Python packages:
```
pandas numpy matplotlib seaborn scipy scikit-learn jupyter ipykernel
```

## Implementation Notes

### Lexer Details
- Uses regex patterns for token matching (keywords, identifiers, strings, numbers, operators)
- Maintains position tracking for error reporting
- Keywords are case-insensitive for user convenience

### Parser Details
- Implements recursive descent parsing (LL(1) grammar)
- Provides helper methods: `current_token()`, `peek_token()`, `expect()`, `match()`
- Each operation has dedicated `parse_<operation>()` method
- Returns list of AST nodes representing the program

### Code Generator Details
- Uses visitor pattern (`visit_<NodeName>()` methods)
- Maintains `self.symbol_table` dict for alias tracking
- Maintains `self.imports` set for dynamic import collection
- Tracks `self.last_plot` boolean to determine if `plt.show()` is needed
- Generates idiomatic pandas code with proper method chaining

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

## Debugging and Development

### Verbose Mode
The runner script can show generated Python code:
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
1. Start with simple test case in `test_noeta.py`
2. Run with verbose mode to inspect generated code
3. Test in Jupyter notebook for interactive development
4. Add to `examples/` directory for documentation

### Common Issues
- **Symbol not found**: Check if alias was registered in symbol table during load/transformation
- **Import missing**: Verify code generator adds necessary import to `self.imports`
- **Syntax error**: Check parser expects tokens in correct order
- **Visualization not showing**: Ensure `self.last_plot` is set to True in code generator

## Architecture Patterns

### Separation of Concerns
- **Lexer**: Only responsible for character → token conversion
- **Parser**: Only responsible for token → AST conversion
- **CodeGen**: Only responsible for AST → Python code conversion
- **Runner/Kernel**: Only responsible for orchestration and execution

### Extensibility
- New operations require additions to all four components (lexer, parser, AST, codegen)
- Each component has clear extension point (keyword mapping, parse method, AST class, visitor method)
- No changes needed to runner or kernel for new operations (unless special execution handling required)

### Data Flow
```
String → Tokens → AST → Python String → Executed Code → Output
```

Each stage produces complete output before next stage begins (no streaming/incremental compilation)

## Future Enhancement Opportunities

### Potential Improvements
1. **Static Type Checking**: Add semantic analysis phase to catch errors before code generation
2. **Query Optimization**: AST transformation passes to optimize pandas operations (e.g., predicate pushdown)
3. **Incremental Compilation**: Cache compiled results for faster re-execution
4. **Better Error Messages**: Line/column information in error reports
5. **Language Server Protocol**: Full IDE integration (VSCode extension)
6. **Macro System**: User-defined operation compositions
7. **Streaming Support**: Handle datasets larger than memory
8. **Performance Profiling**: Built-in timing and memory usage tracking
9. **Export to SQL**: Alternative backend for database execution
10. **Documentation Generation**: Auto-generate docs from Noeta scripts