# Noeta DSL - Data Analysis Domain-Specific Language

**Version**: 2.0
**Status**: ✅ Production Ready (67% coverage, 167/250 operations)
**Last Updated**: December 15, 2025

---

## What is Noeta?

Noeta is a production-ready Domain-Specific Language (DSL) for data analysis that compiles to Python/Pandas code. It provides an intuitive, natural language-like syntax for data manipulation, statistical analysis, and visualization tasks.

### Key Features

- ✅ **Natural Language Syntax**: Write data analysis code that reads like English
- ✅ **Production Ready**: 167 operations implemented, 67% coverage of planned features
- ✅ **Comprehensive**: Covers data I/O, transformation, cleaning, analysis, and visualization
- ✅ **Flexible Execution**: Run via command line, Jupyter notebooks, or VS Code
- ✅ **Optimized Output**: Compiles to efficient pandas/numpy/scikit-learn code
- ✅ **Type Safe**: AST-based compilation with proper type checking
- ✅ **Well Documented**: 13,400+ lines of comprehensive documentation

### Quick Example

```noeta
# Load and analyze sales data
load csv "sales.csv" as sales
filter sales where price > 100 as expensive
groupby expensive by {category} compute {sum: quantity, avg: price} as summary
describe summary
```

Compiles to optimized pandas code automatically!

---

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter ipykernel
```

### Install Noeta

```bash
# Clone or download the repository
cd /Users/anasraza/University/FALL-2025/FYP-II/Project/noeta

# Install as package (optional)
pip install -e .

# Or just install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Command Line Execution

Run a Noeta script file:
```bash
python noeta_runner.py examples/demo_basic.noeta
```

Run inline Noeta code:
```bash
python noeta_runner.py -c 'load "data/sales_data.csv" as sales
describe sales'
```

View generated Python code (verbose mode):
```bash
python noeta_runner.py examples/demo_basic.noeta -v
```

### 2. Jupyter Notebook

Install the Noeta kernel:
```bash
python install_kernel.py
```

Start Jupyter:
```bash
jupyter notebook
```

Create a new notebook and select "Noeta" kernel from the kernel menu.

### 3. VS Code Integration

1. Open the Noeta project folder in VS Code
2. Create a `.noeta` file
3. Run using Python extension (F5 or Run button)

---

## Language Overview

### Basic Operations

#### Load Data
```noeta
load csv "data/sales.csv" as sales
load json "data/customers.json" as customers
load excel "data/products.xlsx" as products
```

#### Select Columns
```noeta
select sales columns {product_id, category, price, quantity} as subset
```

#### Filter Rows
```noeta
filter sales where price > 100 as expensive
filter_between sales column price min=50 max=200 as mid_range
```

#### Transform Data
```noeta
# Math operations
round sales column price decimals=2 as rounded
sqrt sales column area as area_sqrt

# String operations
upper sales column product_name as upper_name
concat sales columns ["first_name", "last_name"] separator=" " as full_name

# Date operations
parse_datetime sales column date as dated
extract_year dated column date as yearly
```

#### Aggregate Data
```noeta
groupby sales by {category} compute {sum: quantity, avg: price} as summary
```

#### Validate Data
```noeta
assert_unique sales column customer_id
assert_no_nulls sales column price
assert_range sales column age min=0 max=120
```

#### Visualize
```noeta
boxplot sales columns {price, quantity}
heatmap sales columns {price, quantity, discount}
```

---

## Example Workflows

### 1. Data Cleaning Pipeline
```noeta
# Load messy data
load csv "raw_data.csv" as raw

# Handle missing values
fill_mean raw column price as cleaned
dropna cleaned as no_nulls

# Remove duplicates
drop_duplicates no_nulls keep="first" as unique_data

# Validate
assert_unique unique_data column id
assert_no_nulls unique_data column price

# Save cleaned data
save unique_data to "cleaned_data.csv"
```

### 2. Time Series Analysis
```noeta
# Load time series data
load csv "stock_prices.csv" as stocks
parse_datetime stocks column date as dated

# Calculate metrics
pct_change dated column close with periods=1 as returns
cumsum dated column volume as cumulative_volume

# Extract time features
extract_dayofweek dated column date as weekday
extract_quarter dated column date as quarter

# Visualize
timeseries dated x=date y=close
```

### 3. Machine Learning Preprocessing
```noeta
# Load training data
load csv "training.csv" as train

# Handle missing values
fill_median train column age as filled

# Encode categorical variables
label_encode filled column category as encoded
one_hot_encode encoded column region as onehot

# Scale features
robust_scale onehot column income as scaled

# Validate
assert_range scaled column age min=0 max=120

# Save preprocessed data
save scaled to "preprocessed.csv"
```

---

## Implementation Status

### Fully Implemented (100% Coverage)
- ✅ Data I/O (10 operations)
- ✅ Selection & Projection (7 operations)
- ✅ Filtering (9 operations)
- ✅ Cleaning (13 operations)
- ✅ Reshaping (7 operations)
- ✅ Combining (6 operations)
- ✅ Binning (2 operations)
- ✅ Apply/Map (4 operations)
- ✅ Cumulative (4 operations)
- ✅ Time Series (3 operations)
- ✅ Scaling (4 operations)
- ✅ Encoding (6 operations)
- ✅ Validation (3 operations)
- ✅ Index Operations (5 operations)
- ✅ Boolean Operations (4 operations)

### Partially Implemented
- ⚠️ String Operations (64% - 14/22)
- ⚠️ Date/Time (58% - 14/24)
- ⚠️ Math Operations (54% - 7/13)
- ⚠️ Aggregation (63% - 20/32)
- ⚠️ Window Functions (64% - 14/22)
- ⚠️ Statistics (47% - 9/19)
- ⚠️ Visualization (33% - 5/15)

**Total**: 167/250 operations (67% coverage)

**See [STATUS.md](STATUS.md) for detailed implementation status.**

---

## Documentation

### For Users
- **[README.md](README.md)** (this file) - Quick start and overview
- **[DEMO_GUIDE.md](DEMO_GUIDE.md)** - Interactive tutorials and demos
- **[NOETA_COMMAND_REFERENCE.md](NOETA_COMMAND_REFERENCE.md)** - Quick syntax reference
- **[DATA_MANIPULATION_REFERENCE.md](DATA_MANIPULATION_REFERENCE.md)** - Comprehensive operation reference (3,220 lines)

### For Developers
- **[CLAUDE.md](CLAUDE.md)** - Comprehensive developer guide (1,025 lines)
- **[FLOW_DIAGRAM.md](FLOW_DIAGRAM.md)** - Architecture diagrams (10 Mermaid diagrams)
- **[SYNTAX_BLUEPRINT.md](SYNTAX_BLUEPRINT.md)** - Design principles and patterns (1,579 lines)

### Status and Planning
- **[STATUS.md](STATUS.md)** - Implementation status and roadmap (Single source of truth)
- **[DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md)** - Master documentation index

---

## Testing

Run basic tests:
```bash
python test_noeta.py
```

Run example files:
```bash
# Phase 1-10 tests
python noeta_runner.py examples/test_phase1_io.noeta
python noeta_runner.py examples/test_phase2_selection.noeta
python noeta_runner.py examples/test_phase3_filtering.noeta

# Phase 11 & 12 tests
python noeta_runner.py examples/test_phase11_all_26_operations.noeta
python noeta_runner.py examples/test_phase12_basic.noeta

# Comprehensive test
python noeta_runner.py examples/test_comprehensive_all_phases.noeta
```

---

## Project Structure

```
noeta/
├── noeta_lexer.py          # Lexer (916 lines, 150+ tokens)
├── noeta_parser.py         # Parser (3,480 lines, 167+ methods)
├── noeta_ast.py            # AST definitions (1,186 lines, 167 nodes)
├── noeta_codegen.py        # Code generator (1,795 lines, 167 visitors)
├── noeta_runner.py         # CLI entry point
├── noeta_kernel.py         # Jupyter kernel (180 lines)
├── install_kernel.py       # Kernel installer
├── test_noeta.py           # Test suite
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
├── examples/               # 20+ example .noeta files
├── data/                   # Sample datasets
└── docs/                   # Documentation
    └── archive/            # Historical phase documentation
```

**Total Core Code**: 7,377 lines
**Total Documentation**: 13,400+ lines

---

## Production Readiness

### ✅ Noeta is Production-Ready For:

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

- ⚠️ Advanced statistical testing (planned for Phase 13+)
- ⚠️ Complex timezone operations (planned for Phase 13)
- ⚠️ Advanced visualization types (planned for Phase 15)
- ⚠️ Trigonometric functions (planned for Phase 13)
- ⚠️ Memory optimization for huge datasets (planned for Phase 14)

---

## Contributing

### To Add New Operations:

1. See [CLAUDE.md](CLAUDE.md) "Adding New Operations" section
2. Follow patterns in [SYNTAX_BLUEPRINT.md](SYNTAX_BLUEPRINT.md)
3. Modify 4 core files: lexer, ast, parser, codegen
4. Add tests and examples
5. Update documentation

### Development Workflow:

```bash
# 1. Add token to lexer
#    noeta_lexer.py: Add TokenType enum and keyword mapping

# 2. Create AST node
#    noeta_ast.py: Add dataclass for operation

# 3. Add parser method
#    noeta_parser.py: Add parse_<operation>() method

# 4. Add code generator
#    noeta_codegen.py: Add visit_<Operation>Node() method

# 5. Test
python noeta_runner.py examples/test_your_operation.noeta -v

# 6. Document
#    Update STATUS.md, DATA_MANIPULATION_REFERENCE.md
```

---

## Dependencies

### Required Packages
```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-learn>=1.0.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

### Version Compatibility
- **Python**: 3.7+
- **Pandas**: 1.x or 2.x (code generator handles both)
- **NumPy**: Any recent version
- **Matplotlib**: 3.x
- **Seaborn**: 0.11+
- **Scikit-learn**: 1.0+

---

## Architecture

Noeta uses a classic 4-stage compilation pipeline:

```
Noeta Source → Lexer → Tokens → Parser → AST → CodeGen → Python → exec()
```

**Key Components**:
- **Lexer** (916 lines): Tokenizes Noeta source into 150+ token types
- **Parser** (3,480 lines): Builds Abstract Syntax Tree from tokens
- **AST** (1,186 lines): Defines 167 operation node types
- **CodeGen** (1,795 lines): Converts AST to pandas/numpy/sklearn code

**Design Patterns**:
- Visitor pattern for code generation
- Symbol table for alias tracking
- Dynamic import management
- Dataclass-based AST nodes

**See [FLOW_DIAGRAM.md](FLOW_DIAGRAM.md) for visual architecture documentation.**

---

## Support

### For Questions:
- Check [DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md) for documentation navigation
- Review [DEMO_GUIDE.md](DEMO_GUIDE.md) for tutorials
- See [STATUS.md](STATUS.md) for implementation coverage
- Read [CLAUDE.md](CLAUDE.md) for developer guide

### For Issues:
- File issues on GitHub (if available)
- Check examples/ directory for similar use cases
- Run with `-v` flag to see generated Python code

### For Feature Requests:
- See [STATUS.md](STATUS.md) for current roadmap
- 83 operations planned for future phases
- 6-8 weeks estimated to 100% coverage

---

## What's Next?

### Planned Phases (83 operations remaining):

**Phase 13** (30 operations) - 2-3 weeks
- Trigonometric functions
- Advanced string operations
- Additional date/time operations

**Phase 14** (30 operations) - 2-3 weeks
- Advanced aggregations and window functions
- Reshaping and merge operations
- Memory & performance optimizations

**Phase 15** (23 operations) - 2 weeks
- Statistical operations
- Additional visualization types

**See [STATUS.md](STATUS.md) for detailed roadmap.**

---

## License

[Add license information]

---

## Acknowledgments

Built with Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Jupyter.

---

**Last Updated**: December 15, 2025
**Version**: 2.0
**Status**: ✅ Production Ready (167/250 operations, 67% coverage)
**Maintained By**: Noeta Development Team

**Get Started**: `python noeta_runner.py examples/demo_basic.noeta`
