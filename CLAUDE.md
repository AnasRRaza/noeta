# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Noeta** is a Domain-Specific Language (DSL) for data analysis that compiles to Python/Pandas code. It provides an intuitive, natural language-like syntax for data manipulation, statistical analysis, and visualization tasks.

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

- `noeta_lexer.py`: Defines `TokenType` enum and `Lexer` class
- `noeta_parser.py`: Contains `Parser` class with methods like `parse_statement()` for each operation type
- `noeta_ast.py`: AST node definitions (all dataclasses inheriting from `ASTNode`)
- `noeta_codegen.py`: `CodeGenerator` class with visitor pattern methods (`visit_<NodeType>`)
- `noeta_runner.py`: Entry point with `compile_noeta()` and `execute_noeta()` functions
- `noeta_kernel.py`: `NoetaKernel` class extending `ipykernel.kernelbase.Kernel`
- `install_kernel.py`: Installs Jupyter kernel specification
- `examples/`: Demo `.noeta` scripts
- `data/`: Sample CSV files for testing

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

## Notes

- The lexer uses regex patterns for token matching
- Parser uses recursive descent parsing
- Code generator uses visitor pattern
- Generated Python code uses pandas DataFrame operations
- Visualization uses matplotlib/seaborn with `seaborn-v0_8-darkgrid` style
- File paths in Noeta code can be relative or absolute
- The system uses `exec()` to run generated Python code in a controlled namespace