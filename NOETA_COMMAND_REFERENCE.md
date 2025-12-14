# Noeta Command Reference Guide

**Purpose**: Quick reference for Noeta command syntax
**Audience**: Users who need fast syntax lookup
**Scope**: Syntax patterns and examples for all 167 operations
**Length**: 901 lines
**Last Updated**: December 15, 2025

**Use this document when**: You need to quickly look up the syntax for a specific operation

**Related Documents**:
- [DATA_MANIPULATION_REFERENCE.md](DATA_MANIPULATION_REFERENCE.md) - Comprehensive operation details
- [SYNTAX_BLUEPRINT.md](SYNTAX_BLUEPRINT.md) - Design principles and patterns
- [STATUS.md](STATUS.md) - Implementation coverage

---

## Table of Contents
1. [Data Manipulation Commands](#data-manipulation-commands)
2. [Analysis Commands](#analysis-commands)
3. [Visualization Commands](#visualization-commands)
4. [File Operations](#file-operations)

---

## Data Manipulation Commands

### 1. LOAD
**Description:** Loads data from a file into memory and assigns it an alias.

**Syntax:**
```noeta
load "<file_path>" as <alias>
```

**Supported Formats:** CSV, JSON, Excel (.xlsx, .xls)

**Example:**
```noeta
load "data/sales_data.csv" as sales
load "data/customers.json" as customers
load "data/products.xlsx" as products
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns
Loaded data/customers.json as customers: 500 rows, 5 columns
Loaded data/products.xlsx as products: 200 rows, 4 columns
```

---

### 2. SELECT
**Description:** Selects specific columns from a dataset and creates a new dataset.

**Syntax:**
```noeta
select <source_alias> {column1, column2, ...} as <new_alias>
```

**Example:**
```noeta
load "data/sales_data.csv" as sales
select sales {product_id, category, price, quantity} as products
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns
Selected 4 columns from sales
```

**Use Case:** Extract only relevant columns for analysis to reduce memory usage and focus on specific attributes.

---

### 3. FILTER
**Description:** Filters rows based on a condition.

**Syntax:**
```noeta
filter <source_alias> [column operator value] as <new_alias>
```

**Supported Operators:** `>`, `<`, `>=`, `<=`, `==`, `!=`

**Example:**
```noeta
load "data/sales_data.csv" as sales
filter sales [price > 50] as expensive_products
filter sales [category == Electronics] as electronics
filter sales [quantity >= 10] as bulk_orders
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns
Filtered sales: 342 rows match condition
Filtered sales: 156 rows match condition
Filtered sales: 234 rows match condition
```

**Use Case:** Extract subsets of data that meet specific criteria for targeted analysis.

---

### 4. SORT
**Description:** Sorts a dataset by one or more columns in ascending or descending order.

**Syntax:**
```noeta
sort <source_alias> by: {column1 ASC/DESC, column2 ASC/DESC} as <new_alias>
```

**Example:**
```noeta
load "data/sales_data.csv" as sales
sort sales by: {price DESC} as sorted_by_price
sort sales by: {category ASC, price DESC} as sorted_multi
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns
Sorted sales by ['price']
Sorted sales by ['category', 'price']
```

**Use Case:** Order data for identifying top/bottom performers, ranking, or preparing for sequential analysis.

---

### 5. JOIN
**Description:** Joins two datasets on a common column using an inner join.

**Syntax:**
```noeta
join <alias1> with: <alias2> on: <column_name> as <new_alias>
```

**Example:**
```noeta
load "data/sales_data.csv" as sales
load "data/customers.csv" as customers
join sales with: customers on: customer_id as sales_customers
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns
Loaded data/customers.csv as customers: 500 rows, 5 columns
Joined sales and customers: 987 rows
```

**Use Case:** Combine related data from multiple sources to enrich analysis with additional context.

---

### 6. GROUPBY
**Description:** Groups data by specified columns and applies aggregation functions.

**Syntax:**
```noeta
groupby <source_alias> by: {column1, column2} agg: {function:column, ...} as <new_alias>
```

**Supported Functions:** `sum`, `avg`/`mean`, `count`, `min`, `max`, `std`

**Example:**
```noeta
load "data/sales_data.csv" as sales
groupby sales by: {category} agg: {sum:quantity, avg:price} as category_summary
groupby sales by: {category, region} agg: {count:order_id, sum:revenue} as detailed_summary
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns
Grouped by ['category']: 5 groups
Grouped by ['category', 'region']: 20 groups
```

**Result Columns:** `category`, `quantity_sum`, `price_mean` (column names are auto-generated)

**Use Case:** Aggregate data for summary statistics, reporting, and high-level insights.

---

### 7. SAMPLE
**Description:** Extracts a sample of rows from a dataset, either randomly or from the top.

**Syntax:**
```noeta
sample <source_alias> n: <size> [random] as <new_alias>
```

**Example:**
```noeta
load "data/sales_data.csv" as sales
sample sales n: 100 as top_100_sales
sample sales n: 50 random as random_50_sales
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns
Sampled 100 rows from sales
Sampled 50 rows from sales
```

**Use Case:** Create smaller datasets for testing, quick analysis, or when working with large datasets.

---

### 8. DROPNA
**Description:** Removes rows containing missing values (NA/NaN).

**Syntax:**
```noeta
dropna <source_alias> [columns: {column1, column2}] as <new_alias>
```

**Note:** If columns are not specified, drops rows with ANY missing values.

**Example:**
```noeta
load "data/sales_data.csv" as sales
dropna sales as clean_sales
dropna sales columns: {price, quantity} as sales_with_valid_amounts
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns
Dropped NA values: 45 rows removed
Dropped NA values: 23 rows removed
```

**Use Case:** Clean datasets by removing incomplete records before analysis.

---

### 9. FILLNA
**Description:** Fills missing values with a specified value.

**Syntax:**
```noeta
fillna <source_alias> value: <fill_value> [columns: {column1, column2}] as <new_alias>
```

**Example:**
```noeta
load "data/customers.csv" as customers
fillna customers value: "Unknown" columns: {city} as complete_customers
fillna customers value: 0 columns: {age, income} as customers_with_defaults
fillna customers value: "N/A" as all_filled
```

**Expected Output:**
```
Loaded data/customers.csv as customers: 500 rows, 5 columns
Filled NA values in customers
Filled NA values in customers
Filled NA values in customers
```

**Use Case:** Impute missing data with default values to maintain dataset completeness.

---

### 10. MUTATE
**Description:** Creates new columns or modifies existing ones using expressions.

**Syntax:**
```noeta
mutate <source_alias> {new_column: "expression", ...} as <new_alias>
```

**Example:**
```noeta
load "data/sales_data.csv" as sales
mutate sales {total_amount: "price * quantity"} as sales_with_total
mutate sales {total_amount: "price * quantity", discount_rate: "discount / 100"} as enhanced_sales
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns
Added/modified 1 columns
Added/modified 2 columns
```

**Use Case:** Derive new features from existing columns for calculations and enriched analysis.

---

### 11. APPLY
**Description:** Applies a transformation function to specified columns.

**Syntax:**
```noeta
apply <source_alias> columns: {column1, column2} function: "<expression>" as <new_alias>
```

**Note:** Use `x` in the expression to reference the column value.

**Example:**
```noeta
load "data/sales_data.csv" as sales
apply sales columns: {price} function: "x * 1.1" as sales_with_markup
apply sales columns: {quantity} function: "x ** 2" as sales_squared
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns
Applied transformation to ['price']
Applied transformation to ['quantity']
```

**Result:** Creates new columns with `_transformed` suffix (e.g., `price_transformed`)

**Use Case:** Apply mathematical transformations to columns for feature engineering.

---

## Analysis Commands

### 12. DESCRIBE
**Description:** Generates descriptive statistics for numeric columns.

**Syntax:**
```noeta
describe <source_alias> [columns: {column1, column2}]
```

**Example:**
```noeta
load "data/sales_data.csv" as sales
describe sales
describe sales columns: {price, quantity}
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns

Descriptive Statistics for sales:
       price    quantity
count  1000.0    1000.0
mean    75.32      15.4
std     25.67       8.2
min     10.00       1.0
25%     55.00       9.0
50%     73.00      14.0
75%     92.00      21.0
max    150.00      50.0

Descriptive Statistics for ['price', 'quantity']:
[Similar output for specified columns only]
```

**Use Case:** Quickly understand the distribution and central tendencies of numeric data.

---

### 13. SUMMARY
**Description:** Provides a comprehensive overview of a dataset including shape, columns, data types, and missing values.

**Syntax:**
```noeta
summary <source_alias>
```

**Example:**
```noeta
load "data/sales_data.csv" as sales
summary sales
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns

Dataset Summary for sales:
Shape: (1000, 8)
Columns: ['order_id', 'customer_id', 'product_id', 'category', 'price', 'quantity', 'date', 'region']

Data types:
order_id        int64
customer_id     int64
product_id      int64
category       object
price         float64
quantity        int64
date           object
region         object
dtype: object

Missing values:
order_id        0
customer_id     0
product_id      5
category        0
price          12
quantity        8
date            0
region          3
dtype: int64
```

**Use Case:** Get a complete overview of your dataset's structure and quality before analysis.

---

### 14. INFO
**Description:** Displays concise information about a dataset including index dtype, column dtypes, non-null counts, and memory usage.

**Syntax:**
```noeta
info <source_alias>
```

**Example:**
```noeta
load "data/sales_data.csv" as sales
info sales
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns

Dataset Info for sales:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 8 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   order_id     1000 non-null   int64
 1   customer_id  1000 non-null   int64
 2   product_id   995 non-null    float64
 3   category     1000 non-null   object
 4   price        988 non-null    float64
 5   quantity     992 non-null    float64
 6   date         1000 non-null   object
 7   region       997 non-null    object
dtypes: float64(3), int64(2), object(3)
memory usage: 62.6+ KB
```

**Use Case:** Quickly assess data completeness and memory footprint.

---

### 15. OUTLIERS
**Description:** Detects outliers in numeric columns using statistical methods.

**Syntax:**
```noeta
outliers <source_alias> method: <iqr|zscore> columns: {column1, column2}
```

**Methods:**
- `iqr`: Interquartile Range method (values beyond 1.5 * IQR from Q1/Q3)
- `zscore`: Z-score method (values with |z| > 3)

**Example:**
```noeta
load "data/sales_data.csv" as sales
outliers sales method: iqr columns: {price, quantity}
outliers sales method: zscore columns: {price}
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns
Outliers in price: 23 rows
Outliers in quantity: 15 rows
Outliers in price (|z| > 3): 8 values
```

**Use Case:** Identify unusual data points that may need investigation or special handling.

---

### 16. QUANTILE
**Description:** Calculates a specific quantile (percentile) for a column.

**Syntax:**
```noeta
quantile <source_alias> column: <column_name> q: <quantile_value>
```

**Note:** Quantile value should be between 0.0 and 1.0 (e.g., 0.95 = 95th percentile)

**Example:**
```noeta
load "data/sales_data.csv" as sales
quantile sales column: price q: 0.95
quantile sales column: quantity q: 0.5
quantile sales column: price q: 0.25
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns
0.95 quantile of price: 135.4500
0.5 quantile of quantity: 14.0000
0.25 quantile of price: 55.0000
```

**Use Case:** Understand data distribution and identify threshold values at specific percentiles.

---

### 17. NORMALIZE
**Description:** Normalizes numeric columns using standardization methods.

**Syntax:**
```noeta
normalize <source_alias> columns: {column1, column2} method: <zscore|minmax> as <new_alias>
```

**Methods:**
- `zscore`: Standard scaling (mean=0, std=1)
- `minmax`: Min-max scaling (range 0-1)

**Example:**
```noeta
load "data/sales_data.csv" as sales
normalize sales columns: {price, quantity} method: zscore as normalized_sales
normalize sales columns: {price} method: minmax as scaled_sales
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns
Normalized columns ['price', 'quantity'] using zscore
Normalized columns ['price'] using minmax
```

**Use Case:** Scale features for machine learning algorithms or comparing variables on different scales.

---

### 18. BINNING
**Description:** Discretizes continuous data into bins (intervals).

**Syntax:**
```noeta
binning <source_alias> column: <column_name> bins: <number_of_bins> as <new_alias>
```

**Example:**
```noeta
load "data/sales_data.csv" as sales
binning sales column: price bins: 5 as sales_price_categories
binning sales column: quantity bins: 3 as sales_quantity_groups
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns
Created 5 bins for column price
Created 3 bins for column quantity
```

**Result:** Creates a new column `<column>_binned` with interval labels

**Use Case:** Convert continuous variables into categorical groups for classification or segmentation.

---

### 19. ROLLING
**Description:** Applies rolling window calculations to a column.

**Syntax:**
```noeta
rolling <source_alias> column: <column_name> window: <window_size> function: <function_name> as <new_alias>
```

**Supported Functions:** `mean`, `sum`, `min`, `max`, `std`, `count`

**Example:**
```noeta
load "data/timeseries.csv" as ts_data
rolling ts_data column: sales window: 7 function: mean as weekly_avg
rolling ts_data column: temperature window: 3 function: max as three_day_max
```

**Expected Output:**
```
Loaded data/timeseries.csv as ts_data: 365 rows, 4 columns
Applied rolling mean with window 7
Applied rolling max with window 3
```

**Result:** Creates a new column `<column>_rolling_<function>`

**Use Case:** Analyze trends and patterns in time-series data using moving averages or aggregations.

---

### 20. HYPOTHESIS
**Description:** Performs hypothesis testing between two datasets.

**Syntax:**
```noeta
hypothesis test: <ttest|ttest_ind|chi2> between: <alias1> and: <alias2> columns: {column1, column2}
```

**Supported Tests:**
- `ttest` / `ttest_ind`: Independent t-test for comparing means
- `chi2`: Chi-square test (placeholder for future implementation)

**Example:**
```noeta
load "data/group_a.csv" as group_a
load "data/group_b.csv" as group_b
hypothesis test: ttest between: group_a and: group_b columns: {score, performance}
```

**Expected Output:**
```
Loaded data/group_a.csv as group_a: 100 rows, 3 columns
Loaded data/group_b.csv as group_b: 100 rows, 3 columns
T-test for score: t-statistic=2.3456, p-value=0.0201
T-test for performance: t-statistic=1.2345, p-value=0.2187
```

**Use Case:** Test statistical significance of differences between groups or conditions.

---

## Visualization Commands

### 21. BOXPLOT
**Description:** Creates box plots to visualize the distribution of numeric columns.

**Syntax:**
```noeta
boxplot <source_alias> columns: {column1, column2}
```

**Example:**
```noeta
load "data/sales_data.csv" as sales
boxplot sales columns: {price, quantity}
boxplot sales columns: {price}
```

**Expected Output:**
- **Jupyter Notebook:** Box plot displayed inline below the cell
- **VS Code:** Box plot opens in a separate window

**Visual Output:** A box plot showing median, quartiles, and outliers for each specified column

**Use Case:** Visualize data distribution, identify outliers, and compare distributions across columns.

---

### 22. HEATMAP
**Description:** Creates a correlation heatmap for numeric columns.

**Syntax:**
```noeta
heatmap <source_alias> columns: {column1, column2, column3}
```

**Example:**
```noeta
load "data/sales_data.csv" as sales
heatmap sales columns: {price, quantity, discount}
```

**Expected Output:**
- **Jupyter Notebook:** Heatmap displayed inline below the cell
- **VS Code:** Heatmap opens in a separate window

**Visual Output:** A color-coded matrix showing correlation coefficients between columns (values from -1 to 1)

**Use Case:** Identify relationships and correlations between multiple numeric variables.

---

### 23. PAIRPLOT
**Description:** Creates pairwise scatter plots for exploring relationships between multiple variables.

**Syntax:**
```noeta
pairplot <source_alias> columns: {column1, column2, column3}
```

**Example:**
```noeta
load "data/sales_data.csv" as sales
pairplot sales columns: {price, quantity, discount}
```

**Expected Output:**
- **Jupyter Notebook:** Pair plot grid displayed inline below the cell
- **VS Code:** Pair plot opens in a separate window

**Visual Output:** A grid of scatter plots showing relationships between all pairs of specified columns, with histograms on the diagonal

**Use Case:** Explore multivariate relationships and distributions simultaneously.

---

### 24. TIMESERIES
**Description:** Creates a time series line plot.

**Syntax:**
```noeta
timeseries <source_alias> x: <time_column> y: <value_column>
```

**Example:**
```noeta
load "data/daily_sales.csv" as daily_data
timeseries daily_data x: date y: revenue
timeseries daily_data x: date y: customer_count
```

**Expected Output:**
- **Jupyter Notebook:** Time series plot displayed inline below the cell
- **VS Code:** Time series plot opens in a separate window

**Visual Output:** A line plot showing how values change over time

**Use Case:** Visualize trends, patterns, and changes over time in sequential data.

---

### 25. PIE
**Description:** Creates a pie chart showing proportions of categorical data.

**Syntax:**
```noeta
pie <source_alias> values: <values_column> labels: <labels_column>
```

**Example:**
```noeta
load "data/sales_data.csv" as sales
groupby sales by: {category} agg: {sum:quantity} as category_totals
pie category_totals values: quantity_sum labels: category
```

**Expected Output:**
- **Jupyter Notebook:** Pie chart displayed inline below the cell
- **VS Code:** Pie chart opens in a separate window

**Visual Output:** A circular chart divided into slices representing proportions, with percentages displayed

**Use Case:** Show the composition of a whole and relative sizes of different categories.

---

## File Operations

### 26. SAVE
**Description:** Saves a dataset to a file in various formats.

**Syntax:**
```noeta
save <source_alias> to: "<file_path>" format: <csv|json|parquet>
```

**Supported Formats:**
- `csv`: Comma-separated values
- `json`: JSON format (records orientation)
- `parquet`: Apache Parquet format

**Example:**
```noeta
load "data/sales_data.csv" as sales
filter sales [price > 100] as expensive_sales
save expensive_sales to: "output/expensive_sales.csv" format: csv
save expensive_sales to: "output/expensive_sales.json" format: json
save expensive_sales to: "output/expensive_sales.parquet" format: parquet
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns
Filtered sales: 342 rows match condition
Saved expensive_sales to output/expensive_sales.csv
Saved expensive_sales to output/expensive_sales.json
Saved expensive_sales to output/expensive_sales.parquet
```

**Use Case:** Export processed data for use in other tools or for sharing results.

---

### 27. EXPORT_PLOT
**Description:** Exports the most recent plot to an image file.

**Syntax:**
```noeta
export_plot filename: "<file_name>" [width: <pixels>] [height: <pixels>]
```

**Supported Formats:** PNG, JPG, PDF, SVG (determined by file extension)

**Example:**
```noeta
load "data/sales_data.csv" as sales
boxplot sales columns: {price, quantity}
export_plot filename: "output/sales_boxplot.png" width: 1200 height: 800

heatmap sales columns: {price, quantity, discount}
export_plot filename: "output/correlation_heatmap.pdf"
```

**Expected Output:**
```
Loaded data/sales_data.csv as sales: 1000 rows, 8 columns
[Box plot displayed]
Exported plot to output/sales_boxplot.png
[Heatmap displayed]
Exported plot to output/correlation_heatmap.pdf
```

**Use Case:** Save visualizations for reports, presentations, or documentation.

---

## Complete Example Workflow

Here's a complete example demonstrating multiple commands in a data analysis workflow:

```noeta
# Load data
load "data/sales_data.csv" as sales
load "data/customers.csv" as customers

# Data cleaning
dropna sales as clean_sales
fillna customers value: "Unknown" columns: {city} as complete_customers

# Join datasets
join clean_sales with: complete_customers on: customer_id as sales_customers

# Feature engineering
mutate sales_customers {total_amount: "price * quantity", discount_rate: "discount / 100"} as enhanced_data

# Filtering and sampling
filter enhanced_data [total_amount > 100] as high_value_sales
sample high_value_sales n: 15 random as sample_data

# Statistical analysis
describe high_value_sales columns: {price, quantity, total_amount}
normalize high_value_sales columns: {price, quantity} method: zscore as normalized_sales
outliers normalized_sales method: iqr columns: {price}
quantile high_value_sales column: total_amount q: 0.95

# Aggregation
groupby enhanced_data by: {category} agg: {sum:quantity, avg:price, count:order_id} as category_summary
sort category_summary by: {quantity_sum DESC} as top_categories

# Visualizations
boxplot high_value_sales columns: {price, quantity}
heatmap normalized_sales columns: {price, quantity, total_amount}
pairplot high_value_sales columns: {price, quantity, total_amount}

# Export results
save high_value_sales to: "output/high_value_sales.csv" format: csv
save category_summary to: "output/category_summary.json" format: json
export_plot filename: "output/analysis_plots.png" width: 1200 height: 800
```

---

## Tips and Best Practices

1. **Always use aliases:** Every command that creates or modifies data requires an alias with the `as` keyword.

2. **Column lists use curly braces:** When specifying multiple columns, use `{column1, column2}` syntax.

3. **Conditions use square brackets:** Filter conditions are enclosed in `[condition]`.

4. **String values in quotes:** File paths and string literals should be in double quotes.

5. **Chain operations:** Each operation creates a new dataset, allowing you to build complex analysis pipelines.

6. **Check data quality first:** Use `summary`, `info`, and `describe` before performing analysis.

7. **Visualize before and after:** Create visualizations to verify transformations and understand data distributions.

8. **Export important results:** Use `save` to preserve processed datasets and `export_plot` for visualizations.

9. **Environment-aware plotting:** Plots automatically display inline in Jupyter notebooks and in separate windows in VS Code.

10. **Comments are supported:** Use `#` for single-line comments to document your analysis.

---

## Error Handling

If you encounter errors:
- Check file paths are correct and files exist
- Verify column names match those in your dataset
- Ensure aliases are defined before use
- Confirm syntax matches the examples above
- Check that numeric operations are applied to numeric columns

---

## Version Information

**Noeta Version:** 1.0
**Last Updated:** 2025-10-26
