# Development Checklist

Quick reference for common development tasks in Noeta.

---

## ✅ Adding a New Python Module

When creating a new `.py` file in the project:

- [ ] Create the new module (e.g., `noeta_errors.py`)
- [ ] **Add module to `setup.py`** in the `py_modules` list
- [ ] Import the module in files that need it
- [ ] Reinstall the package:
  ```bash
  pip install -e . --force-reinstall --no-deps
  ```
- [ ] Test with both `python noeta_runner.py` and `noeta` command
- [ ] Update documentation (CLAUDE.md, STATUS.md)

**Common new modules:**
- Error handling: `noeta_errors.py` ✅ (added in Phase 1)
- Semantic analysis: `noeta_semantic.py` (Phase 3)
- Operation hints: `noeta_hints.py` (Phase 2)
- Utilities: `noeta_utils.py`

---

## ✅ Adding a New Operation

**IMPORTANT**: All 5 core files must be updated for each new operation!

### 1. Lexer (`noeta_lexer.py`)
- [ ] Add token to `TokenType` enum
- [ ] Add keyword to `self.keywords` dictionary in `Lexer.__init__()`

### 2. AST (`noeta_ast.py`)
- [ ] Create AST node class using `@dataclass`
- [ ] Inherit from `ASTNode`
- [ ] Define all required attributes (e.g., `source_alias`, `new_alias`, etc.)
- [ ] ⚠️ **Check attribute naming**: Use `source_alias`, `new_alias` (not `source`, `alias`)

### 3. Parser (`noeta_parser.py`)
- [ ] Add `parse_<operation>()` method
- [ ] Add case to `parse_statement()` dispatcher
- [ ] Track start token position for error context
- [ ] Return appropriate AST node

### 4. Code Generator (`noeta_codegen.py`)
- [ ] Add `visit_<NodeName>Node()` method
- [ ] Generate pandas/Python code
- [ ] Handle imports (add to `self.imports` if needed)
- [ ] Update symbol table if operation creates new dataset

### 5. Semantic Validator (`noeta_semantic.py`) ⭐ NEW!
- [ ] **Add `visit_<NodeName>Node()` method**
- [ ] **Check source dataset(s) exist** using `_check_dataset_exists()`
- [ ] **Validate columns exist** (if applicable) using `_check_column_exists()`
- [ ] **Check types** (if applicable) using `_check_column_type()`
- [ ] **Register result dataset** if operation creates new dataset

**Semantic Validator Template:**
```python
def visit_<OperationName>Node(self, node):
    """Validate <operation> operation."""
    # 1. Check source dataset exists
    source_info = self._check_dataset_exists(node.source_alias, node)

    # 2. Validate columns (if applicable)
    # if hasattr(node, 'column'):
    #     self._check_column_exists(source_info, node.column, node)

    # 3. Register result dataset (if creates new dataset)
    if hasattr(node, 'new_alias'):
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"<operation> from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)
```

### 6. Testing & Documentation
- [ ] Create test file in `examples/test_*.noeta`
- [ ] Test the operation end-to-end
- [ ] Add unit tests in `tests/` (if test suite exists)
- [ ] Update `STATUS.md` coverage metrics
- [ ] Add to `DATA_MANIPULATION_REFERENCE.md` or `NOETA_COMMAND_REFERENCE.md`

### Quick Verification Checklist
```bash
# 1. Verify lexer recognizes token
python -c "from noeta_lexer import Lexer; print(Lexer('your_operation').tokenize())"

# 2. Verify parser creates AST
python -c "from noeta_lexer import Lexer; from noeta_parser import Parser;
code='your_operation test as result';
ast=Parser(Lexer(code).tokenize(), code).parse();
print(ast.statements[0])"

# 3. Verify semantic validation works
python -c "from noeta_runner import compile_noeta;
compile_noeta('load \"d.csv\" as d\nyour_operation d as r')"

# 4. Test full execution
noeta -c 'load "data/sales_data.csv" as d
your_operation d as result
describe result'
```

---

## ✅ Making a Commit

- [ ] Run existing tests to ensure nothing broke
- [ ] Test new functionality thoroughly
- [ ] Update documentation
- [ ] Create meaningful commit message
- [ ] Follow the commit message format used in the repo

---

## ✅ Package Installation Issues

If you see `ModuleNotFoundError`:

1. Check if the module is listed in `setup.py` → `py_modules`
2. Reinstall package: `pip install -e . --force-reinstall --no-deps`
3. Verify installation: `pip show noeta`
4. Check import: `python -c "import noeta_errors"`

---

## ✅ Testing Workflow

**Quick test with inline code:**
```bash
noeta -c 'load "data/sales_data.csv" as d
describe d'
```

**Test with file:**
```bash
noeta examples/test_basic.noeta
```

**Verbose mode (see generated Python):**
```bash
noeta examples/test_basic.noeta -v
```

**Test all examples:**
```bash
for f in examples/test_*.noeta; do
    echo "Testing $f..."
    noeta "$f" || echo "FAILED: $f"
done
```

---

## ✅ Error Message Testing

Create intentional errors to test error infrastructure:

```bash
# Syntax error
noeta -c 'load "data.csv" sales'  # Missing AS

# Lexer error
noeta -c 'load "data.csv" as sales @ test'  # Invalid character

# EOF error
noeta -c 'load "data.csv" as'  # Incomplete
```

---

## Common Gotchas

⚠️ **Dataclass field ordering**: Fields without defaults must come before fields with defaults
⚠️ **setup.py**: Always update when adding new modules
⚠️ **Package reinstall**: Required after modifying setup.py
⚠️ **Import order**: Circular imports can break everything
⚠️ **Parser position**: Track start token before parsing for error context
⚠️ **Attribute naming**: Use `source_alias` and `new_alias` (not `source` and `alias`)
⚠️ **Semantic validator**: ALWAYS add when adding new operations (catches errors at compile-time)

---

Last updated: December 17, 2025 (Semantic Validation System)
