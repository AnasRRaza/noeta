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

- [ ] Add token to `noeta_lexer.py` (TokenType enum + keywords dict)
- [ ] Create AST node in `noeta_ast.py`
- [ ] Add parse method in `noeta_parser.py`
- [ ] Add visitor method in `noeta_codegen.py`
- [ ] Create test file in `examples/test_*.noeta`
- [ ] Test the operation
- [ ] Update `STATUS.md` coverage metrics
- [ ] Add to `DATA_MANIPULATION_REFERENCE.md` or `NOETA_COMMAND_REFERENCE.md`

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

---

Last updated: Phase 1 completion (Error Infrastructure)
