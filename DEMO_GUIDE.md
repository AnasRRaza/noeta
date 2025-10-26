# Noeta DSL - Setup and Demo Guide

## Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter ipykernel
```

### 2. Install Noeta Jupyter Kernel
```bash
cd /home/claude/noeta
python install_kernel.py
```

## Demo 1: Command Line Execution

### Basic Demo
```bash
# Navigate to the Noeta directory
cd /home/claude/noeta

# Run a simple inline example
python noeta_runner.py -c 'load "data/sales_data.csv" as sales
describe sales'

# Run the basic demo file
python noeta_runner.py examples/demo_basic.noeta
```

## Demo 2: VS Code Integration

### Setup for VS Code
1. Open VS Code
2. Open the folder `/home/claude/noeta`
3. Install Python extension if not already installed
4. Create a new file with `.noeta` extension

### Running in VS Code
1. Create a new file `test_demo.noeta` with this content:
```noeta
# Live demo in VS Code
load "data/sales_data.csv" as sales
select sales {product_id, category, price} as products
filter products [price > 100] as expensive
describe expensive
```

2. Create a Python script `run_noeta.py`:
```python
import sys
sys.path.insert(0, '/home/claude/noeta')
from noeta_runner import execute_noeta

# Read the .noeta file
with open('test_demo.noeta', 'r') as f:
    code = f.read()

# Execute it
execute_noeta(code)
```

3. Run using VS Code's Run button or F5

## Demo 3: Jupyter Notebook

### Setup Jupyter
```bash
# Start Jupyter
jupyter notebook
```

### In Jupyter Notebook:
1. Create a new notebook
2. Select "Noeta" kernel from the kernel menu
3. Start writing Noeta code in cells

### Example Notebook Cells:

**Cell 1: Load and explore data**
```noeta
load "data/sales_data.csv" as sales
info sales
```

**Cell 2: Data manipulation**
```noeta
select sales {category, price, quantity} as products
filter products [price > 50] as expensive
describe expensive
```

**Cell 3: Aggregation**
```noeta
groupby sales by: {category} agg: {count:product_id, sum:quantity, avg:price} as summary
describe summary
```

**Cell 4: Visualization**
```noeta
boxplot sales columns: {price, quantity}
```

## Live Demonstration Script

### Part 1: Basic Data Loading and Exploration (2 minutes)
```noeta
# Load the sales dataset
load "data/sales_data.csv" as sales

# Get basic information
info sales
summary sales
```

### Part 2: Data Transformation (3 minutes)
```noeta
# Select specific columns
select sales {product_id, category, price, quantity} as products

# Filter for high-value items
filter products [price > 100] as expensive_products

# Sort by price
sort expensive_products by: price desc as sorted_products

# Show statistics
describe sorted_products
```

### Part 3: Advanced Analysis (3 minutes)
```noeta
# Group by category
groupby sales by: {category} agg: {count:product_id, sum:quantity, avg:price} as category_analysis

# Handle missing data
load "data/customers.csv" as customers
fillna customers value: "Unknown" columns: {city} as complete_customers

# Join datasets
join sales with: complete_customers on: customer_id as full_data
```

### Part 4: Statistical Analysis (2 minutes)
```noeta
# Detect outliers
outliers sales method: iqr columns: {price}

# Calculate quantiles
quantile sales column: price q: 0.95

# Create derived columns
mutate sales {total: "price * quantity", discounted: "price * (1 - discount/100)"} as enhanced
```

### Part 5: Visualization (2 minutes)
```noeta
# Box plot for price distribution
boxplot sales columns: {price}

# Correlation heatmap
select sales {price, quantity, discount} as numeric_cols
heatmap numeric_cols columns: {price, quantity, discount}

# Time series analysis
timeseries sales x: date y: price
```

### Part 6: Export Results (1 minute)
```noeta
# Save processed data
save sorted_products to: "output/processed_sales.csv" format: csv

# Export visualizations
export_plot filename: "output/analysis.png" width: 1200 height: 800
```

## Key Features to Highlight

1. **Intuitive Syntax**: Natural language-like commands
2. **Comprehensive Operations**: 25+ data operations supported
3. **Seamless Integration**: Works in CLI, VS Code, and Jupyter
4. **Automatic Python Generation**: Compiles to optimized Pandas code
5. **Built-in Visualizations**: Direct support for common plots
6. **Error Handling**: Clear error messages in DSL terms

## Troubleshooting

### If Jupyter kernel doesn't appear:
```bash
jupyter kernelspec list  # Check if noeta is listed
python install_kernel.py  # Reinstall
jupyter notebook --debug  # Run with debug for more info
```

### If imports fail:
```bash
pip install --upgrade pandas numpy matplotlib seaborn scipy scikit-learn
```

### If file paths don't work:
- Use absolute paths: `/home/claude/noeta/data/sales_data.csv`
- Or ensure you're in the correct directory

## Advanced Features Demo (If Time Permits)

### Rolling Window Analysis
```noeta
rolling sales column: price window: 3 function: mean as rolling_avg
```

### Binning
```noeta
binning sales column: price bins: 5 as price_bins
```

### Normalization
```noeta
normalize sales columns: {price, quantity} method: zscore as normalized
```

### Hypothesis Testing
```noeta
# Load two datasets for comparison
sample sales n: 50 random as group1
sample sales n: 50 random as group2
hypothesis group1 vs: group2 columns: {price} test: ttest_ind
```

## Presentation Tips

1. Start with simple examples and build complexity
2. Show the generated Python code to demonstrate compilation
3. Emphasize the reduction in code complexity (Noeta vs raw Python)
4. Have backup screenshots in case of technical issues
5. Keep sample data small for quick execution
6. Focus on the DSL design and ease of use

## Success Metrics to Mention

- Successfully implemented 25+ operations from the formal grammar
- Full lexer, parser, and code generator pipeline
- Integration with industry-standard tools (Jupyter, VS Code)
- Clean separation of concerns in architecture
- Extensible design for future features

Good luck with your presentation!
