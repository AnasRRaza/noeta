# Noeta DSL - Data Analysis Domain-Specific Language

## Quick Start

### Installation
```bash
pip install pandas numpy matplotlib seaborn jupyter ipykernel
```

### Running Noeta Scripts

#### Command Line
```bash
python noeta_runner.py examples/demo.noeta
```

#### VS Code
1. Open any `.noeta` file
2. Run using Python extension (F5 or Run button)

#### Jupyter Notebook
```bash
# Install the kernel
python -m noeta_kernel.install

# Start Jupyter
jupyter notebook
# Select "Noeta" kernel when creating new notebook
```

## Language Examples

```noeta
load "data.csv" as mydata
select mydata {col1, col2} as subset
filter subset [col1 > 100] as filtered
describe filtered
```
