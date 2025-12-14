# Noeta DSL Syntax Blueprint

**Version**: 2.0
**Last Updated**: December 12, 2025
**Status**: Reference Document for Unified Syntax System

---

## Purpose

This document serves as the **authoritative style guide** for the Noeta DSL. It defines:

1. **Design principles** that guide all syntax decisions
2. **Core syntax patterns** that cover all operation types
3. **Grammar specifications** for the language
4. **Guidelines for adding new operations** while maintaining consistency
5. **Expression language** for DSL-native transformations

**Use this document when**:
- Adding new operations to Noeta
- Reviewing syntax for consistency
- Designing new language features
- Teaching others about Noeta design

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [Core Syntax Patterns](#core-syntax-patterns)
3. [Grammar Reference](#grammar-reference)
4. [Adding New Operations](#adding-new-operations)
5. [Expression Language](#expression-language)
6. [Style Guidelines](#style-guidelines)
7. [Examples Gallery](#examples-gallery)
8. [Common Mistakes](#common-mistakes)

---

## Design Principles

### Principle 1: Consistency Above All
**Rule**: Every operation of the same type should follow the same syntax pattern.

**Good**:
```noeta
round data column price with decimals=2 as rounded
sqrt data column area as area_sqrt
log data column value with base=10 as logged
```
All math operations follow: `<op> <source> column <col> [with params] as <alias>`

**Bad**:
```noeta
round data column price with decimals=2 as rounded
sqrt data with column=area as area_sqrt        # Different pattern!
log data on value base: 10 as logged           # Different keywords!
```

---

### Principle 2: Natural Language Keywords
**Rule**: Prefer readable English words over symbols and abbreviations.

**Good**:
```noeta
filter sales where price > 100 as expensive
groupby sales by {category} compute {sum: quantity} as stats
join orders with customers on customer_id as enriched
```

**Bad**:
```noeta
filter sales [price > 100] as expensive        # Brackets less clear
groupby sales by: {category} agg: {sum:qty}   # Abbreviation "agg"
join orders -> customers : customer_id         # Symbols
```

---

### Principle 3: Minimal Punctuation
**Rule**: Use punctuation only when necessary for clarity, not decoration.

**Good**:
```noeta
select sales columns {product_id, price, quantity} as subset
filter sales where price > 100 as expensive
```

**Bad**:
```noeta
select sales columns: {product_id, price, quantity} as subset   # Unnecessary colon
filter sales where: price > 100 as expensive                     # Unnecessary colon
```

**Acceptable punctuation**:
- Braces `{}` for column lists: `{col1, col2, col3}`
- Brackets `[]` for value lists: `[1, 2, 3]`
- Colon `:` for key-value in aggregation: `{sum: quantity}`
- Equals `=` for parameters: `with n=10`
- Quotes `""` for string literals: `"Electronics"`

---

### Principle 4: Pure DSL (No Python Leakage)
**Rule**: The syntax should not require Python knowledge. No lambda, no isinstance, no Python-specific constructs.

**Good**:
```noeta
apply sales with transform value * 1.1 where is_numeric(value) else value as inflated
map data column price with transform if(value > 100, 100, value) as capped
```

**Bad**:
```noeta
applymap sales function="lambda x: x * 1.1 if isinstance(x, (int, float)) else x"
apply data function="lambda row: row['price'] * row['quantity']"
```

---

### Principle 5: Explicit is Better Than Implicit
**Rule**: Make the operation's intent clear through explicit keywords.

**Good**:
```noeta
select sales columns {product_id, price} as subset
groupby sales by {category} compute {sum: quantity} as stats
filter sales where price > 100 as expensive
```

**Bad**:
```noeta
select sales {product_id, price} as subset              # Missing "columns" keyword
groupby sales {category} {sum: quantity} as stats       # Missing "by" and "compute"
filter sales price > 100 as expensive                   # Missing "where"
```

---

### Principle 6: One Canonical Syntax Per Operation
**Rule**: Each operation has exactly one correct syntax. No dual syntax, no alternatives.

**Good**: One way to filter
```noeta
filter sales where price > 100 as expensive
```

**Bad**: Multiple ways (confusing)
```noeta
# Option 1
filter sales where price > 100 as expensive
# Option 2
filter sales [price > 100] as expensive        # DEPRECATED
# Option 3
filter sales with column=price value>100       # DEPRECATED
```

---

### Principle 7: Scalable Pattern Design
**Rule**: Syntax patterns should work for simple cases and scale to complex cases.

**Good**: Same pattern works for simple and complex
```noeta
# Simple
groupby sales by {category} as grouped

# Complex
groupby sales by {region, category} compute {sum: quantity, mean: price, count: order_id} as stats
```

**Bad**: Different patterns for different complexity
```noeta
# Simple
groupby sales by category as grouped

# Complex
groupby sales by: {region, category} agg: {sum:quantity, mean:price, count:order_id} as stats
```

---

## Core Syntax Patterns

The Noeta DSL is built on **7 core syntax patterns** that cover all operation types.

---

### Pattern 1: Simple Column Transformation

**Purpose**: Transform a single column with optional parameters

**Syntax**:
```
<operation> <source> column <column_name> [with <param>=<value> ...] as <alias>
```

**Grammar**:
```ebnf
simple_transformation = operation source "column" column_id [with_clause] as_clause ;
operation = IDENTIFIER ;
source = IDENTIFIER ;
column_id = IDENTIFIER ;
with_clause = "with" parameter {"with" parameter} ;
parameter = IDENTIFIER "=" value ;
as_clause = "as" IDENTIFIER ;
```

**Applies to** (54 operations after consolidation):
- Math operations: round, abs, sqrt, power, log, ceil, floor
- String operations: upper, lower, strip, replace, split, concat, substring, length, title, capitalize, lstrip, rstrip, extract_regex, find
- Date/time operations: parse_datetime, extract (consolidated), date_add, date_subtract, format_datetime, date_diff
- Cumulative operations: cumsum, cummax, cummin, cumprod
- Time series operations: pct_change, diff, shift
- Type operations: astype, to_numeric, one_hot_encode, label_encode, ordinal_encode, target_encode
- Cleaning operations: fillna (consolidated), interpolate
- Index operations: set_index
- Binning operations: binning, cut

**Examples**:

```noeta
# Math operations
round sales column price with decimals=2 as rounded_prices
sqrt products column area as area_sqrt
log data column value with base=10 as log_values
power features column distance with exponent=2 as distance_squared
abs temps column reading as abs_temp

# String operations
upper customers column name as uppercase_names
replace text column description with old="bad" new="good" as cleaned
substring emails column address with start=0 end=10 as prefixes
concat data columns {first_name, last_name} with separator=" " as full_name
length products column sku as sku_length

# Date/time operations
extract timestamps column created_at with part="hour" as hour_of_day
extract orders column order_date with part="month" as month
parse_datetime data column date_str with format="%Y-%m-%d" as parsed_dates
date_add events column event_date with value=7 unit="days" as week_later

# Cumulative operations
cumsum sales column revenue as running_total
cummax prices column value as max_so_far
cumprod growth column rate as compound_growth

# Time series operations
pct_change sales column price with periods=1 as price_change
diff metrics column value with periods=7 as week_over_week
shift data column sales with periods=1 fill_value=0 as prev_sales

# Type operations
astype data column age with dtype="int64" as age_int
to_numeric data column price with errors="coerce" as price_numeric
label_encode data column category as category_encoded
```

**When to use**:
- Operation targets a single column
- Operation creates a new column or modifies one column
- Optional parameters are simple scalars (numbers, strings, booleans)

**Rules**:
1. Always use `column` keyword before column name
2. Column name is an IDENTIFIER (no quotes)
3. Optional parameters use `with param=value` syntax
4. Multiple parameters: `with param1=value1 with param2=value2`
5. Always end with `as <alias>`

---

### Pattern 2: Multi-Column Operations

**Purpose**: Operations that target multiple columns simultaneously

**Syntax**:
```
<operation> <source> columns {<col1>, <col2>, ...} [with <param>=<value>] as <alias>
```

**Grammar**:
```ebnf
multi_column_operation = operation source "columns" column_list [with_clause] as_clause ;
column_list = "{" column_id ("," column_id)* "}" ;
```

**Applies to** (30 operations):
- Selection: select, select_by_type, rename, reorder
- Scaling: standard_scale, minmax_scale, robust_scale, maxabs_scale
- Statistics: describe, corr, cov
- Visualization: boxplot, heatmap, pairplot
- Cleaning: dropna (with column subset), drop_duplicates
- Others: set_multiindex

**Examples**:

```noeta
# Selection
select sales columns {product_id, price, quantity} as core_fields
select_by_type data with type="numeric" as numeric_cols
rename customers columns {first_name: fname, last_name: lname} as renamed
reorder sales columns {date, product, quantity, price} as reordered

# Scaling
standard_scale features columns {age, income, score} as scaled_features
minmax_scale data columns {price, rating} as normalized
robust_scale metrics columns {metric1, metric2, metric3} as robust_scaled

# Statistics
describe data columns {price, quantity, discount}
corr sales columns {price, quantity, revenue}
cov returns columns {stock_a, stock_b, stock_c}

# Visualization
boxplot sales columns {price, quantity}
heatmap correlations columns {var1, var2, var3, var4}
pairplot features columns {age, income, score}

# Cleaning
dropna data columns {price, quantity} as clean_data
drop_duplicates customers columns {email, phone} with keep="first" as unique_customers

# Index
set_multiindex data columns {region, category} as hierarchical
```

**When to use**:
- Operation needs to target multiple columns
- All targeted columns receive the same treatment
- Result affects multiple columns

**Rules**:
1. Use `columns` (plural) keyword
2. Column list in braces: `{col1, col2, col3}`
3. For rename: use `{old_name: new_name}` mapping syntax
4. Optional parameters still use `with param=value`
5. End with `as <alias>`

**Special Case - Rename**:
```noeta
rename data columns {old_col1: new_col1, old_col2: new_col2} as renamed
```
Uses colon `:` to map old → new names

---

### Pattern 3: Filtering & Conditional Operations

**Purpose**: Filter rows based on conditions

**Syntax**:
```
<operation> <source> where <condition> as <alias>
```

**Condition Syntax**:
```
<column> <operator> <value>
<column> between <min> and <max>
<column> in [<val1>, <val2>, ...]
<column> contains "<pattern>"
<column> starts_with "<prefix>"
<column> ends_with "<suffix>"
<column> matches "<regex>"
<column> is null
<column> is not null
```

**Grammar**:
```ebnf
filtering_operation = operation source "where" condition as_clause ;

condition = simple_condition | compound_condition ;

simple_condition = column_id operator value
                 | column_id "between" value "and" value
                 | column_id "in" list_value
                 | column_id ("contains" | "starts_with" | "ends_with" | "matches") string
                 | column_id "is" ["not"] "null"
                 ;

compound_condition = condition logical_op condition
                   | "not" condition
                   | "(" condition ")"
                   ;

operator = "==" | "!=" | "<" | ">" | "<=" | ">=" ;
logical_op = "and" | "or" ;
```

**Applies to** (1 consolidated operation):
- filter (handles all filtering cases through rich where clause syntax)

**Examples**:

```noeta
# Basic comparison
filter sales where price > 100 as expensive_items
filter customers where age >= 18 as adults
filter products where category == "Electronics" as electronics
filter orders where quantity != 0 as valid_orders

# Between
filter sales where price between 50 and 200 as mid_range
filter events where date between "2023-01-01" and "2023-12-31" as year_2023

# In list
filter data where category in ["A", "B", "C"] as selected_categories
filter customers where country in ["US", "CA", "MX"] as north_america

# String matching
filter text where description contains "premium" as premium_items
filter products where code starts_with "PRO" as pro_products
filter emails where address ends_with "@gmail.com" as gmail_users
filter data where phone matches "\\d{3}-\\d{3}-\\d{4}" as valid_phones

# Null checking
filter data where discount is null as no_discount
filter data where email is not null as has_email

# Duplicates (special case - uses "with" for parameters)
filter orders where duplicated with subset={customer_id, date} keep="first" as unique_orders

# Complex conditions (future enhancement)
filter sales where price > 100 and quantity < 10 as edge_cases
filter products where category == "Electronics" or category == "Computers" as tech
filter data where not (status == "inactive") as active
```

**When to use**:
- Filtering rows based on column values
- Selecting subset of data based on criteria

**Rules**:
1. Always use `where` keyword
2. Column name is IDENTIFIER (no quotes)
3. String values in quotes: `"Electronics"`
4. Numeric values: no quotes: `100`
5. Lists use brackets: `["A", "B", "C"]`
6. Special keywords: `between`, `in`, `contains`, `starts_with`, `ends_with`, `matches`, `is null`, `is not null`

---

### Pattern 4: Aggregation & Grouping

**Purpose**: Group data by columns and compute aggregations

**Syntax**:
```
groupby <source> by {<col1>, <col2>} compute {<agg1>: <target_col1>, <agg2>: <target_col2>} as <alias>
```

**Grammar**:
```ebnf
groupby_operation = "groupby" source "by" column_list ["compute" aggregation_list] as_clause ;

aggregation_list = "{" aggregation ("," aggregation)* "}" ;

aggregation = function_name ":" column_id ;

function_name = "sum" | "mean" | "median" | "min" | "max" | "count" | "std" | "var"
              | "first" | "last" | "nunique" | "quantile" | ... ;
```

**Applies to** (20 operations):
- groupby (with compute)
- agg
- Window functions: rank, dense_rank, row_number, percent_rank, ntile, lag, lead
- Rolling/expanding: rolling, expanding

**Examples**:

```noeta
# Single groupby column
groupby sales by {category} compute {sum: revenue, mean: price, count: order_id} as category_stats

# Multiple groupby columns
groupby orders by {region, category} compute {sum: quantity, mean: price} as regional_summary

# Simple groupby (no aggregation)
groupby customers by {country} as by_country

# Complex aggregations
groupby transactions by {customer_id, date} compute {
  sum: amount,
  mean: amount,
  count: transaction_id,
  nunique: product_id
} as daily_customer_stats

# Window functions
rank sales by {price} with method="dense" as price_ranks
row_number data by {date} with partition={category} as row_nums
ntile revenue by {amount} with n=4 as quartiles

# Lag/Lead
lag sales column price with periods=1 as prev_price
lead sales column price with periods=1 as next_price

# Rolling/Expanding
rolling sales column price with window=7 function="mean" as rolling_avg
expanding sales column revenue with function="sum" as expanding_total
```

**When to use**:
- Grouping data by one or more columns
- Computing aggregate statistics per group
- Window functions (rank, lag, lead, etc.)

**Rules**:
1. Use `by {columns}` for grouping columns (no `with` keyword)
2. Use `compute {func: col}` for aggregations (replaces old `agg:`)
3. Aggregation list uses colon to map function to column
4. Window functions use `with` for additional parameters

**Key Change**: `agg:` → `compute:` for clarity and natural language

---

### Pattern 5: Combining Operations

**Purpose**: Join, merge, or concatenate multiple datasets

**Syntax**:
```
<operation> <source1> with <source2> [on <column>] [with <param>=<value>] as <alias>
```

**Grammar**:
```ebnf
combining_operation = operation source1 "with" source2 [on_clause] [with_clause] as_clause ;

on_clause = "on" column_id ;

source1 = IDENTIFIER ;
source2 = IDENTIFIER ;
```

**Applies to** (6 operations):
- join
- merge
- concat_vertical
- concat_horizontal
- append
- cross_join

**Examples**:

```noeta
# Join operations
join sales with customers on customer_id as enriched_sales
join orders with products on product_id as complete_orders
cross_join categories with regions as all_combinations

# Merge with options
merge orders with products on product_id with how="left" as complete_orders
merge customers with addresses on customer_id with how="inner" as customer_addresses

# Concatenation
concat_vertical sales_2023 with sales_2024 as all_sales
concat_horizontal features with labels as training_data

# Append
append new_rows with existing_data as updated_data
```

**When to use**:
- Combining two or more datasets
- Joining on common keys
- Concatenating rows or columns

**Rules**:
1. Use `with` to specify second dataset
2. Use `on <column>` for join key
3. Additional parameters use `with param=value`
4. Both source datasets are IDENTIFIERS (aliases)

---

### Pattern 6: Reshaping Operations

**Purpose**: Change the structure/shape of data (pivot, melt, transpose, etc.)

**Syntax**:
```
<operation> <source> with <param>=<value> [with <param>=<value>] as <alias>
```

**Grammar**:
```ebnf
reshaping_operation = operation source with_clause as_clause ;
```

**Applies to** (7 operations):
- pivot
- melt
- stack
- unstack
- transpose
- explode
- normalize

**Examples**:

```noeta
# Pivot
pivot sales with index=category columns=region values=revenue as pivoted

# Melt
melt wide_data with id_vars={id, name} value_vars={q1, q2, q3, q4} as long_data

# Explode
explode data with column=tags as exploded_tags

# Stack/Unstack
stack pivoted_data as stacked
unstack long_data as unstacked

# Transpose
transpose data as transposed

# Normalize (nested JSON/dict to flat table)
normalize json_data as flat_table
```

**When to use**:
- Converting wide to long format or vice versa
- Restructuring data layout
- Flattening nested structures

**Rules**:
1. Parameters specify structure (index, columns, values, etc.)
2. Use `with param=value` for all parameters
3. Column lists use braces: `id_vars={col1, col2}`

---

### Pattern 7: Function Application (DSL Expressions)

**Purpose**: Apply custom transformations using DSL expression language

**Syntax**:
```
apply <source> with transform <expression> as <alias>
map <source> column <column> with transform <expression> as <alias>
map <source> column <column> with mapping {<key1>: <val1>, <key2>: <val2>} as <alias>
```

**Grammar**:
```ebnf
apply_operation = "apply" source "with" "transform" expression as_clause ;

map_operation = "map" source "column" column_id "with" map_spec as_clause ;

map_spec = "transform" expression
         | "mapping" dict_value
         ;

expression = term (("+"|"-") term)* ;

term = factor (("*"|"/"|"%") factor)* ;

factor = value
       | column_id
       | function_call
       | "(" expression ")"
       | conditional_expr
       ;

function_call = function_name "(" [expression ("," expression)*] ")" ;

conditional_expr = "if" "(" condition "," expression "," expression ")"
                 | expression "where" condition "else" expression
                 ;
```

**Applies to** (4 operations):
- apply
- map
- (Replaces: applymap, map_values)

**DSL Expression Features**:

**Variables**:
- `value` - current cell value (in map operations)
- `row` - current row (in apply operations - future)
- `col` - current column name (in apply operations - future)

**Operators**:
- Arithmetic: `+`, `-`, `*`, `/`, `%`, `**`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `and`, `or`, `not`

**DSL Functions**:
- Type checking: `is_numeric(x)`, `is_string(x)`, `is_null(x)`
- Math: `abs(x)`, `sqrt(x)`, `round(x, decimals)`, `power(x, exp)`, `log(x, base)`
- String: `upper(x)`, `lower(x)`, `length(x)`, `strip(x)`
- Conditionals: `if(condition, true_val, false_val)`, `coalesce(val1, val2, default)`
- Aggregates: `mean(column)`, `sum(column)`, `max(column)`, `min(column)` (for apply)

**Examples**:

```noeta
# Element-wise transformation
apply sales with transform value * 1.1 as inflated
map data column price with transform value * 1.1 as price_markup
map salaries column amount with transform value * 1.05 as adjusted

# Conditional transformation
map sales column discount with transform if(value > 0.5, 0.5, value) as capped_discount
apply data with transform value * 1.1 where is_numeric(value) else value as inflated

# Type-safe operations (replaces isinstance checks)
map mixed column data with transform value ** 2 where is_numeric(value) else 0 as squared
apply data with transform round(value, 2) where is_numeric(value) else value as rounded

# String transformations
map customers column name with transform upper(value) as upper_names
map products column sku with transform length(value) as sku_length

# Null handling
map data column price with transform coalesce(value, 0) as price_filled
map orders column discount with transform if(is_null(value), 0, value) as discount_filled

# Map with dictionary (simpler than current map_values)
map sales column status with mapping {"active": 1, "inactive": 0, "pending": 2} as status_code
map products column size with mapping {"S": 1, "M": 2, "L": 3, "XL": 4} as size_code
```

**When to use**:
- Custom element-wise transformations
- Conditional transformations
- Type-safe operations
- Value mapping

**Rules**:
1. `apply` works on entire dataframe (future: row-wise operations)
2. `map` works on single column (element-wise)
3. Expressions use DSL functions (no Python lambda)
4. Use `where condition else value` for conditionals
5. Use `mapping {dict}` for value replacement

**Replaces**:
- Old `applymap` operation (merged into `apply`)
- Old `map_values` operation (now `map ... with mapping`)
- All lambda function strings
- All isinstance checks

---

## Grammar Reference

### Complete EBNF Grammar

```ebnf
(* Noeta DSL Unified Grammar v2.0 *)

program = {statement} ;

statement = load_statement
          | save_statement
          | operation_statement
          | groupby_statement
          | combining_statement
          | apply_statement
          | analysis_statement
          | visualization_statement
          | validation_statement
          ;

(* I/O Operations *)

load_statement = "load" format string_literal [with_clause] as_clause ;

save_statement = "save" source "to" string_literal "as" format [with_clause] ;

format = "csv" | "json" | "excel" | "parquet" | "sql" ;

(* Operation Statements *)

operation_statement = operation source [column_spec | columns_spec] [where_clause] [with_clause] as_clause ;

operation = IDENTIFIER ;

source = IDENTIFIER ;

column_spec = "column" IDENTIFIER ;

columns_spec = "columns" column_list ;

column_list = "{" IDENTIFIER ("," IDENTIFIER)* "}" ;

where_clause = "where" condition ;

condition = simple_condition | compound_condition ;

simple_condition = IDENTIFIER operator value
                 | IDENTIFIER "between" value "and" value
                 | IDENTIFIER "in" list_value
                 | IDENTIFIER ("contains"|"starts_with"|"ends_with"|"matches") string_literal
                 | IDENTIFIER "is" ["not"] "null"
                 ;

compound_condition = condition logical_op condition
                   | "not" condition
                   | "(" condition ")"
                   ;

operator = "==" | "!=" | "<" | ">" | "<=" | ">=" ;

logical_op = "and" | "or" ;

with_clause = "with" parameter {"with" parameter} ;

parameter = IDENTIFIER "=" value ;

as_clause = "as" IDENTIFIER ;

value = string_literal | numeric_literal | boolean_literal | list_value | dict_value | IDENTIFIER ;

list_value = "[" [value ("," value)*] "]" ;

dict_value = "{" [key_value_pair ("," key_value_pair)*] "}" ;

key_value_pair = (string_literal | IDENTIFIER) ":" value ;

(* Groupby Statement *)

groupby_statement = "groupby" source "by" column_list ["compute" aggregation_list] as_clause ;

aggregation_list = "{" aggregation ("," aggregation)* "}" ;

aggregation = IDENTIFIER ":" IDENTIFIER ;

(* Combining Statement *)

combining_statement = operation source "with" source ["on" IDENTIFIER] [with_clause] as_clause ;

(* Apply/Map Statements *)

apply_statement = apply_operation | map_operation ;

apply_operation = "apply" source "with" "transform" expression as_clause ;

map_operation = "map" source "column" IDENTIFIER "with" map_spec as_clause ;

map_spec = "transform" expression | "mapping" dict_value ;

(* Expression Language *)

expression = term (("+" | "-") term)* ;

term = factor (("*" | "/" | "%") factor)* ;

factor = value
       | IDENTIFIER
       | function_call
       | "(" expression ")"
       | conditional_expression
       ;

function_call = IDENTIFIER "(" [expression ("," expression)*] ")" ;

conditional_expression = "if" "(" condition "," expression "," expression ")"
                       | expression "where" condition "else" expression
                       ;

(* Analysis Statements *)

analysis_statement = ("describe" | "summary" | "info" | "count_na") source [columns_spec] ;

(* Visualization Statements *)

visualization_statement = plot_type source [plot_params] ;

plot_type = "boxplot" | "heatmap" | "pairplot" | "timeseries" | "pie" ;

plot_params = columns_spec | with_clause ;

(* Validation Statements *)

validation_statement = assert_statement ;

assert_statement = "assert_unique" source column_spec
                 | "assert_no_nulls" source column_spec
                 | "assert_range" source column_spec "with" "min" "=" value "max" "=" value
                 ;

(* Terminals *)

IDENTIFIER = letter {letter | digit | "_"} ;

string_literal = '"' {character} '"' ;

numeric_literal = ["-"] digit {digit} ["." digit {digit}] ;

boolean_literal = "true" | "false" ;

letter = "a".."z" | "A".."Z" ;

digit = "0".."9" ;

character = (* any printable character except " *) ;
```

### Token Types

All tokens defined in `noeta_lexer.py`:

**Keywords**:
- Operations: load, save, select, filter, groupby, join, merge, etc.
- Clause keywords: as, column, columns, where, with, by, compute, on, transform
- Conditional: if, where, else, and, or, not
- Type keywords: between, in, contains, starts_with, ends_with, matches, is, null
- Formats: csv, json, excel, parquet, sql

**Operators**:
- Comparison: ==, !=, <, >, <=, >=
- Arithmetic: +, -, *, /, %, **
- Logical: and, or, not

**Delimiters**:
- Braces: {, }
- Brackets: [, ]
- Parentheses: (, )
- Punctuation: :, ,, =

**Literals**:
- STRING: Double-quoted strings
- NUMERIC: Integer or float
- BOOLEAN: true, false
- IDENTIFIER: Variable names, column names

---

## Adding New Operations

### Step-by-Step Guide

#### Step 1: Choose the Appropriate Pattern

Determine which of the 7 core patterns your operation belongs to:

1. **Simple Column Transformation**: Operates on single column, optional parameters
2. **Multi-Column**: Operates on multiple columns simultaneously
3. **Filtering**: Filters rows based on conditions
4. **Aggregation**: Groups and aggregates data
5. **Combining**: Joins/merges multiple datasets
6. **Reshaping**: Changes data structure
7. **Function Application**: Custom transformations using expressions

#### Step 2: Design the Syntax

Follow the chosen pattern exactly. Example for a new math operation `sigmoid`:

**Pattern**: Simple Column Transformation

**Syntax**:
```noeta
sigmoid <source> column <column> as <alias>
```

#### Step 3: Add Token to Lexer

File: `noeta_lexer.py`

```python
class TokenType(Enum):
    # ... existing tokens ...
    SIGMOID = "SIGMOID"

# In Lexer.__init__()
self.keywords = {
    # ... existing keywords ...
    'sigmoid': TokenType.SIGMOID,
}
```

#### Step 4: Create AST Node

File: `noeta_ast.py`

```python
@dataclass
class SigmoidNode(ASTNode):
    """Represents a sigmoid transformation: sigmoid <source> column <col> as <alias>"""
    source_alias: str
    column: str
    new_alias: str
```

#### Step 5: Add Parser Method

File: `noeta_parser.py`

```python
def parse_sigmoid(self):
    """Parse: sigmoid <source> column <column> as <alias>"""
    self.expect(TokenType.SIGMOID)

    source_token = self.expect(TokenType.IDENTIFIER)
    source = source_token.value

    self.expect(TokenType.COLUMN)
    column_token = self.expect(TokenType.IDENTIFIER)
    column = column_token.value

    # Parse alias
    self.expect(TokenType.AS)
    alias_token = self.expect(TokenType.IDENTIFIER)
    alias = alias_token.value

    return SigmoidNode(source, column, alias)

# Add to parse_statement() dispatcher
def parse_statement(self):
    # ... existing cases ...
    elif self.match(TokenType.SIGMOID):
        return self.parse_sigmoid()
    # ...
```

#### Step 6: Add Code Generator

File: `noeta_codegen.py`

```python
def visit_SigmoidNode(self, node):
    """Generate code for sigmoid transformation."""
    df_var = self.symbol_table.get(node.source_alias)
    if not df_var:
        raise ValueError(f"Unknown dataframe: {node.source_alias}")

    # Add import
    self.imports.add("import numpy as np")

    # Generate code
    code = f"{node.new_alias} = {df_var}.copy()\n"
    code += f"{node.new_alias}['{node.column}'] = 1 / (1 + np.exp(-{df_var}['{node.column}']))\n"
    code += f"print(f'Applied sigmoid to {node.column}')"

    self.code_lines.append(code)
    self.symbol_table[node.new_alias] = True
```

#### Step 7: Add Tests and Examples

Create test file: `examples/test_sigmoid.noeta`

```noeta
load csv "data/test.csv" as data
sigmoid data column score as transformed
describe transformed
```

#### Step 8: Update Documentation

1. Add to `NOETA_COMMAND_REFERENCE.md`
2. Add to `DATA_MANIPULATION_REFERENCE.md`
3. Update operation count in `CURRENT_STATUS.md`

### Naming Conventions

**Operation Names**:
- Use descriptive verb or verb_noun format
- Lowercase with underscores: `extract_year`, `filter_between`
- Avoid abbreviations: `compute` not `agg`, `standard_scale` not `stdscale`

**Column Names**:
- Always identifiers (no quotes): `price` not `"price"`
- Use snake_case: `order_date` not `orderDate`

**Alias Names**:
- Descriptive of transformation: `rounded_prices`, `scaled_features`
- Use snake_case
- Avoid generic names: `result`, `output`, `data2`

**Parameter Names**:
- Clear and explicit: `decimals=2`, `window=7`, `method="linear"`
- Use full words: `delimiter` not `delim`, `separator` not `sep`
- Match pandas parameter names when applicable

---

## Expression Language

The Noeta DSL includes a first-class expression language for transformations, replacing Python lambda functions.

### Expression Syntax

```ebnf
expression = term (("+" | "-") term)* ;

term = factor (("*" | "/" | "%") factor)* ;

factor = value
       | IDENTIFIER
       | function_call
       | "(" expression ")"
       | conditional_expression
       ;
```

### Variables

- **value**: Current cell value (in element-wise operations)
- **row** (future): Current row as dict
- **col** (future): Current column name

### Operators

**Arithmetic**:
- `+` Addition
- `-` Subtraction
- `*` Multiplication
- `/` Division
- `%` Modulo
- `**` Exponentiation

**Comparison** (in conditionals):
- `==` Equal
- `!=` Not equal
- `<` Less than
- `>` Greater than
- `<=` Less than or equal
- `>=` Greater than or equal

**Logical** (in conditionals):
- `and` Logical AND
- `or` Logical OR
- `not` Logical NOT

### DSL Functions

**Type Checking**:
```noeta
is_numeric(value)   # Returns true if value is int/float
is_string(value)    # Returns true if value is string
is_null(value)      # Returns true if value is null/NaN
is_boolean(value)   # Returns true if value is boolean
```

**Math Functions**:
```noeta
abs(value)              # Absolute value
sqrt(value)             # Square root
round(value, decimals)  # Round to decimals places
power(value, exponent)  # Raise to power
log(value, base)        # Logarithm with base
ceil(value)             # Round up
floor(value)            # Round down
```

**String Functions**:
```noeta
upper(value)        # Convert to uppercase
lower(value)        # Convert to lowercase
length(value)       # String length
strip(value)        # Remove whitespace
```

**Conditional Functions**:
```noeta
if(condition, true_value, false_value)      # Ternary conditional
coalesce(val1, val2, ..., default)          # Return first non-null value
```

**Aggregate Functions** (in apply, for future use):
```noeta
mean(column)    # Mean of column
sum(column)     # Sum of column
max(column)     # Maximum of column
min(column)     # Minimum of column
count(column)   # Count of non-null values
```

### Expression Examples

**Simple arithmetic**:
```noeta
map sales column price with transform value * 1.1 as increased
map data column x with transform value ** 2 as squared
map temps column celsius with transform value * 9 / 5 + 32 as fahrenheit
```

**Conditionals**:
```noeta
map sales column discount with transform if(value > 0.5, 0.5, value) as capped
map data column age with transform if(value < 0, 0, value) as cleaned
```

**Type-safe operations**:
```noeta
apply mixed with transform value * 2 where is_numeric(value) else value as doubled
map data column price with transform round(value, 2) where is_numeric(value) else 0 as rounded
```

**Null handling**:
```noeta
map data column price with transform coalesce(value, 0) as filled
map orders column discount with transform if(is_null(value), 0, value) as discount_filled
```

**Combined operations**:
```noeta
map sales column price with transform round(value * 1.1, 2) where is_numeric(value) else value as markup
map data column score with transform if(value > 100, 100, if(value < 0, 0, value)) as clamped
```

---

## Style Guidelines

### Keyword Casing
- All keywords are **lowercase**: `load`, `filter`, `groupby`, `select`
- Never use uppercase: ~~`LOAD`~~, ~~`Filter`~~

### Identifier Guidelines
- **Column names**: snake_case, no quotes: `product_id`, `order_date`
- **Alias names**: snake_case, descriptive: `rounded_prices`, `filtered_data`
- **Source names**: snake_case: `sales_data`, `customers`

### Spacing
- Space around `=`: `decimals=2` ✓ not `decimals =2` ✗
- Space around operators in expressions: `value * 1.1` ✓ not `value*1.1` ✗
- No space before comma: `{col1, col2}` ✓ not `{col1 , col2}` ✗
- Space after comma: `{col1, col2}` ✓ not `{col1,col2}` ✗

### Line Length
- Keep statements under 100 characters when possible
- Break long parameter lists:
  ```noeta
  # Good
  pivot sales
    with index=category
    with columns=region
    with values=revenue
    as pivoted

  # Also acceptable
  pivot sales with index=category columns=region values=revenue as pivoted
  ```

### Comments
- Use `#` for comments
- Comment before complex operations:
  ```noeta
  # Calculate 7-day rolling average of sales
  rolling sales column revenue with window=7 function="mean" as rolling_avg
  ```

### Alias Naming Best Practices

**Good alias names**:
- `rounded_prices` (describes transformation)
- `high_value_customers` (describes subset)
- `monthly_revenue` (describes aggregation level)
- `scaled_features` (describes processing)

**Bad alias names**:
- `result`, `output`, `data2` (too generic)
- `temp`, `tmp`, `x` (unclear purpose)
- `df1`, `df2` (generic numbering)

### String Literals
- Always use double quotes: `"Electronics"` not `'Electronics'`
- Escape internal quotes: `"He said \"hello\""`

### Numeric Literals
- No quotes: `100`, `3.14`, `-5`
- Use decimal point for floats: `1.0` not `1`

### Boolean Literals
- Lowercase: `true`, `false`
- Never use: ~~`True`~~, ~~`False`~~ (Python style)

---

## Examples Gallery

### Complete Data Pipeline

**Before (inconsistent syntax)**:
```noeta
load csv "data/sales.csv" as sales
select sales {product_id, category, price, quantity} as products
filter products [price > 50] as expensive
groupby expensive by: {category} agg: {sum:quantity, avg:price} as summary
dropna summary columns: {quantity_sum} as clean
standard_scale clean column "price_mean" as normalized
describe normalized
save normalized to: "output/results.csv" format: csv
```

**After (unified syntax)**:
```noeta
load csv "data/sales.csv" as sales
select sales columns {product_id, category, price, quantity} as products
filter products where price > 50 as expensive
groupby expensive by {category} compute {sum: quantity, mean: price} as summary
dropna summary columns {quantity_sum} as clean
standard_scale clean column price_mean as normalized
describe normalized
save normalized to "output/results.csv" as csv
```

### Time Series Analysis

**Unified syntax**:
```noeta
# Load time series data
load csv "data/timeseries.csv" as ts

# Parse dates
parse_datetime ts column date with format="%Y-%m-%d" as parsed

# Calculate cumulative sum
cumsum parsed column sales as running_total

# Calculate percentage change
pct_change running_total column sales with periods=1 as growth_rate

# Lag sales by one period
shift growth_rate column sales with periods=1 fill_value=0 as lagged

# Apply 10% markup using DSL expressions
apply lagged with transform value * 1.1 where is_numeric(value) else value as adjusted

# Filter high growth periods
filter adjusted where growth_rate > 0.05 as high_growth

# Calculate 7-day rolling average
rolling high_growth column sales with window=7 function="mean" as smoothed

# Summary statistics
describe smoothed columns {sales, growth_rate}
```

### Customer Segmentation

**Unified syntax**:
```noeta
# Load customer data
load csv "data/customers.csv" as customers

# Select relevant columns
select customers columns {id, name, age, city, purchases, revenue} as subset

# Filter adults in target age range
filter_between subset where age between 25 and 65 as adults

# Standardize name formatting
upper adults column name as upper_names

# Clean city names
replace upper_names column city with old="NYC" new="New York" as cleaned

# Map status to numeric codes
map cleaned column status with mapping {"active": 1, "inactive": 0} as coded

# Create customer segments by city
groupby coded by {city} compute {
  count: id,
  mean: age,
  sum: revenue,
  mean: purchases
} as city_stats

# Sort by average revenue
sort city_stats by revenue_mean desc as sorted

# Save results
save sorted to "output/customer_segments.json" as json
```

### Feature Engineering

**Unified syntax**:
```noeta
# Load training data
load csv "data/training.csv" as raw

# Fill missing ages with median
fill_median raw column age as filled_age

# Fill missing fares with 0
fillna filled_age column fare with value=0 as filled

# Create family size feature
mutate filled with family_size = sibsp + parch as with_family

# Create is_alone indicator
map with_family column family_size with transform if(value == 0, 1, 0) as with_alone

# Bin age into categories
cut with_alone column age
  with bins=[0, 12, 18, 35, 60, 100]
  with labels=["child", "teen", "young_adult", "adult", "senior"]
  as with_age_group

# One-hot encode categorical features
one_hot_encode with_age_group column age_group as encoded

# Standard scale numeric features
standard_scale encoded columns {age, fare, family_size} as scaled

# Save feature set
save scaled to "output/features.csv" as csv
```

---

## Common Mistakes

### Mistake 1: Quoting Column Names

**Wrong**:
```noeta
round sales column "price" with decimals=2 as rounded
filter data where "category" == "Electronics" as electronics
```

**Right**:
```noeta
round sales column price with decimals=2 as rounded
filter data where category == "Electronics" as electronics
```

**Rule**: Column names are identifiers (no quotes). Only string *values* need quotes.

---

### Mistake 2: Using Old Syntax Patterns

**Wrong**:
```noeta
groupby sales by: {category} agg: {sum:quantity} as stats
filter data [price > 100] as expensive
sample data n: 10 as sampled
```

**Right**:
```noeta
groupby sales by {category} compute {sum: quantity} as stats
filter data where price > 100 as expensive
sample data with n=10 as sampled
```

**Rule**: Use unified syntax patterns (no colons after keywords, use `with`, use `where`, use `compute`).

---

### Mistake 3: Inconsistent Parameter Syntax

**Wrong**:
```noeta
head data n=10 as first_ten              # Missing "with"
filter data where price > 100 as exp     # Correct
groupby data category {sum:qty} as stats # Missing "by" keyword
```

**Right**:
```noeta
head data with n=10 as first_ten
filter data where price > 100 as exp
groupby data by {category} compute {sum: qty} as stats
```

**Rule**: Follow the exact pattern for each operation type. Parameters use `with`, filtering uses `where`, groupby uses `by`.

---

### Mistake 4: Using Python Constructs

**Wrong**:
```noeta
applymap sales function="lambda x: x * 1.1" as inflated
apply data function="lambda row: row['price'] * row['qty']" as total
map data column status mapping={"active": 1, "inactive": 0} as coded  # Old syntax
```

**Right**:
```noeta
apply sales with transform value * 1.1 as inflated
apply data with transform price * quantity as total  # (future: row access)
map data column status with mapping {"active": 1, "inactive": 0} as coded
```

**Rule**: Use DSL expressions, not Python lambda. Use `with transform` or `with mapping`.

---

### Mistake 5: Missing Keywords

**Wrong**:
```noeta
select sales {product_id, price} as subset        # Missing "columns"
groupby sales {category} {sum: qty} as stats      # Missing "by" and "compute"
filter sales price > 100 as expensive              # Missing "where"
```

**Right**:
```noeta
select sales columns {product_id, price} as subset
groupby sales by {category} compute {sum: qty} as stats
filter sales where price > 100 as expensive
```

**Rule**: Don't omit required keywords. Explicit is better than implicit.

---

### Mistake 6: Incorrect Column List Syntax

**Wrong**:
```noeta
select sales columns [product_id, price] as subset         # Wrong brackets
select sales columns product_id, price as subset           # Missing braces
select sales columns {"product_id", "price"} as subset     # Quoted identifiers
```

**Right**:
```noeta
select sales columns {product_id, price} as subset
```

**Rule**: Column lists use braces `{}`, identifiers are not quoted.

---

### Mistake 7: Wrong Aggregation Syntax

**Wrong**:
```noeta
groupby sales by: {category} agg: {sum:quantity} as stats  # Old syntax with colons
groupby sales by {category} agg {sum: quantity} as stats   # Still using "agg"
```

**Right**:
```noeta
groupby sales by {category} compute {sum: quantity} as stats
```

**Rule**: Use `compute` (not `agg`), no colons after keywords.

---

## Version History

- **v2.0** (December 12, 2025): Unified syntax system
  - Standardized to 7 core patterns
  - Eliminated Python constructs
  - Replaced `agg` with `compute`
  - Added DSL expression language
  - Single canonical syntax per operation

- **v1.0** (Pre-unification): Organic growth phase
  - Multiple syntax patterns
  - Dual syntax support
  - Python lambda functions
  - Inconsistent parameter syntax

---

## Conclusion

This blueprint defines the **authoritative syntax standard** for Noeta DSL v2.0. By following these patterns and principles, all operations maintain consistency, readability, and user-friendliness.

**Core Takeaways**:
1. **7 core patterns** cover all operation types
2. **Pure DSL** - no Python constructs
3. **Consistent keywords** - `with`, `where`, `by`, `compute`, `columns`, `column`
4. **Identifiers not strings** for column names
5. **DSL expressions** replace lambda functions
6. **One canonical syntax** per operation

When in doubt, refer to this blueprint. Consistency is the foundation of good language design.
