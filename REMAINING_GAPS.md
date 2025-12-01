# Remaining Implementation Gaps

**Last Updated**: December 2, 2025
**Current Coverage**: 154/250 operations (61%)
**Remaining**: 96 operations

---

## Summary by Priority

| Priority | Operations | Impact | Timeline Estimate |
|----------|-----------|---------|-------------------|
| 🟡 Medium | 13 operations | Enhances functionality | 1-2 weeks |
| 🟢 Low | 83 operations | Nice-to-have | 2-4 weeks |
| **TOTAL** | **96 operations** | - | **3-6 weeks** |

---

## 🟡 MEDIUM PRIORITY (13 Operations)

### Scaling & Normalization (2 operations)
1. **Robust Scale** - Scale using median and IQR (robust to outliers)
   - Syntax: `robust_scale data column price as price_robust`
   - Pandas: `from sklearn.preprocessing import RobustScaler`

2. **Max Abs Scale** - Scale by maximum absolute value
   - Syntax: `maxabs_scale data column value as value_scaled`
   - Pandas: `from sklearn.preprocessing import MaxAbsScaler`

### Advanced Encoding (2 operations)
3. **Ordinal Encode** - Encode with specified order
   - Syntax: `ordinal_encode data column size order=["S", "M", "L", "XL"] as size_encoded`
   - Pandas: `df['col'].map({'S': 1, 'M': 2, 'L': 3, 'XL': 4})`

4. **Target Encode** - Encode based on target variable (for ML)
   - Syntax: `target_encode data column category target="sales" as category_encoded`
   - Pandas: `df.groupby('category')['sales'].mean()`

### Data Validation (3 operations)
5. **Assert Unique** - Validate uniqueness constraint
   - Syntax: `assert_unique data column id`
   - Pandas: `assert df['id'].is_unique, "Duplicate IDs found"`

6. **Assert No Nulls** - Validate no missing values
   - Syntax: `assert_no_nulls data column required_field`
   - Pandas: `assert not df['col'].isnull().any()`

7. **Assert Range** - Validate values within range
   - Syntax: `assert_range data column age min=0 max=120`
   - Pandas: `assert df['age'].between(0, 120).all()`

### Advanced Index Operations (2 operations)
8. **Reindex** - Conform DataFrame to new index
   - Syntax: `reindex data with index=[0, 1, 2, 3] as reindexed`
   - Pandas: `df.reindex([0, 1, 2, 3])`

9. **MultiIndex Operations** - Hierarchical indexing
   - Syntax: `set_multiindex data columns ["category", "subcategory"] as hierarchical`
   - Pandas: `df.set_index(['category', 'subcategory'])`

### Boolean Operations (4 operations)
10. **Any** - Check if any value is True
    - Syntax: `any data column flag`
    - Pandas: `df['flag'].any()`

11. **All** - Check if all values are True
    - Syntax: `all data column flag`
    - Pandas: `df['flag'].all()`

12. **Boolean Aggregations** - Count True values
    - Syntax: `count_true data column flag`
    - Pandas: `df['flag'].sum()`

13. **DataFrame Comparison** - Compare two DataFrames
    - Syntax: `compare df1 with df2`
    - Pandas: `df1.compare(df2)`

---

## 🟢 LOW PRIORITY (83 Operations)

### Trigonometric Operations (6 operations)
14. **Sin** - Sine function
15. **Cos** - Cosine function
16. **Tan** - Tangent function
17. **ArcSin** - Inverse sine
18. **ArcCos** - Inverse cosine
19. **ArcTan** - Inverse tangent

**Use Case**: Scientific computing, signal processing, physics simulations

### Advanced String Operations (8 operations)
20. **Count Substring** - Count occurrences of substring
21. **Repeat String** - Repeat string N times
22. **Pad String** - Add padding characters
23. **String Slice with Step** - Advanced slicing
24. **String Contains (case-sensitive)**
25. **String Replace (regex)**
26. **String Join** - Join list elements
27. **String Wrap** - Wrap text to width

**Use Case**: Text preprocessing, NLP, report formatting

### Additional Date/Time Operations (10 operations)
28. **Is Month Start** - Check if date is first day of month
29. **Is Month End** - Check if date is last day of month
30. **Is Quarter Start** - Check if date is first day of quarter
31. **Is Quarter End** - Check if date is last day of quarter
32. **Is Year Start** - Check if date is January 1st
33. **Is Year End** - Check if date is December 31st
34. **Timezone Localize** - Add timezone information
35. **Timezone Convert** - Convert between timezones
36. **Business Days** - Calculate business days between dates
37. **Add Business Days** - Add N business days to date

**Use Case**: Financial calendars, business date logic, international operations

### Additional Aggregation Operations (12 operations)
38. **Weighted Mean** - Average with weights
39. **Weighted Sum** - Sum with weights
40. **Mode** - Most frequent value
41. **Variance** - Statistical variance
42. **Skewness** - Distribution skewness
43. **Kurtosis** - Distribution kurtosis
44. **Correlation** - Pearson correlation
45. **Covariance** - Statistical covariance
46. **Named Aggregations** - Multiple aggs with custom names
47. **Conditional Aggregation** - Aggregate with conditions
48. **First Valid** - First non-null value
49. **Last Valid** - Last non-null value

**Use Case**: Statistical analysis, data quality metrics, advanced analytics

### Window Functions (8 operations)
50. **Window Cumulative Sum within Groups**
51. **Window Dense Rank**
52. **Window Percent Rank**
53. **Window Row Number**
54. **Window First Value**
55. **Window Last Value**
56. **Window Nth Value**
57. **Custom Window Functions**

**Use Case**: Advanced SQL-like analytics, ranking within groups

### Reshaping Operations (5 operations)
58. **Wide to Long** - Specialized melt
59. **Long to Wide** - Specialized pivot
60. **Explode** - Transform list-like column into rows
61. **Implode** - Combine rows into lists
62. **Normalize Nested JSON** - Flatten nested structures

**Use Case**: Data format conversion, API data processing

### Advanced Merge Operations (7 operations)
63. **Merge with Indicator** - Show merge source
64. **Merge Asof** - Nearest key merge
65. **Merge Ordered** - Merge with sorting
66. **Cross Join** - Cartesian product
67. **Anti Join** - Rows in left not in right
68. **Semi Join** - Rows in left that have match in right
69. **Merge with Validation** - Validate merge types

**Use Case**: Complex data integration, time series joins

### Memory & Performance (5 operations)
70. **Memory Usage** - Report memory consumption
71. **Optimize Types** - Downcast to smaller types
72. **To Sparse** - Convert to sparse format
73. **To Dense** - Convert from sparse format
74. **Categorize** - Convert to categorical dtype

**Use Case**: Large dataset optimization, memory-constrained environments

### Partition & Chunking (2 operations)
75. **Chunk by Size** - Split DataFrame into chunks
76. **Partition by Column** - Partition based on column values

**Use Case**: Parallel processing, distributed computing

### Statistical Operations (10 operations)
77. **Z-Score** - Standardized score
78. **T-Test** - Student's t-test
79. **Chi-Square Test** - Chi-square test
80. **ANOVA** - Analysis of variance
81. **Linear Regression** - Fit linear model
82. **Polynomial Regression** - Fit polynomial model
83. **Moving Average (various types)**
84. **Exponential Smoothing**
85. **Seasonal Decomposition**
86. **Autocorrelation**

**Use Case**: Statistical modeling, hypothesis testing, forecasting

### Visualization Operations (10 operations)
87. **Scatter Plot** - X-Y scatter plot
88. **Line Plot** - Line chart
89. **Bar Chart** - Bar chart
90. **Histogram** - Distribution histogram
91. **Area Chart** - Filled area plot
92. **Violin Plot** - Distribution violin plot
93. **Swarm Plot** - Categorical scatter
94. **Joint Plot** - Bivariate distribution
95. **Facet Grid** - Multi-plot grid
96. **3D Plot** - Three-dimensional visualization

**Use Case**: Data exploration, presentation, reporting

---

## Implementation Roadmap

### Phase 12: Medium Priority Operations (13 ops) - 1-2 weeks
**Focus**: Scaling, encoding, validation, boolean operations
**Impact**: Enhanced ML pipelines, data quality checks
**Estimated LOC**: ~800 lines
**Files to modify**: 4 (lexer, ast, parser, codegen)

### Phase 13: Low Priority Batch 1 (30 ops) - 2 weeks
**Focus**: Trigonometric, advanced strings, additional dates
**Impact**: Specialized use cases
**Estimated LOC**: ~2000 lines

### Phase 14: Low Priority Batch 2 (30 ops) - 2 weeks
**Focus**: Advanced aggregations, window functions, reshaping
**Impact**: Advanced analytics capabilities
**Estimated LOC**: ~2000 lines

### Phase 15: Low Priority Batch 3 (23 ops) - 1-2 weeks
**Focus**: Merge operations, memory optimization, statistics, visualization
**Impact**: Performance and specialized analytics
**Estimated LOC**: ~1500 lines

---

## Current vs Target Coverage

```
Current: ████████████████░░░░░░░░░░░░░░░░ 61% (154/250)
Target:  ████████████████████████████████ 100% (250/250)
Gap:     ░░░░░░░░░░░░░░░░                 39% (96 operations)
```

### Coverage by Category (Current State)

| Category | Ops | Coverage | Bar |
|----------|-----|----------|-----|
| Data I/O | 10/10 | 100% | ████████████████████ |
| Selection | 7/7 | 100% | ████████████████████ |
| Filtering | 9/9 | 100% | ████████████████████ |
| Cleaning | 13/13 | 100% | ████████████████████ |
| Reshaping | 7/7 | 100% | ████████████████████ |
| Combining | 6/6 | 100% | ████████████████████ |
| Binning | 2/2 | 100% | ████████████████████ |
| Apply/Map | 4/4 | 100% | ████████████████████ |
| Date/Time | 14/24 | 58% | ███████████░░░░░░░░░ |
| String | 14/22 | 64% | ████████████░░░░░░░░ |
| Math | 7/13 | 54% | ██████████░░░░░░░░░░ |
| Aggregation | 20/32 | 63% | ████████████░░░░░░░░ |
| Window | 14/22 | 64% | ████████████░░░░░░░░ |
| Stats | 9/19 | 47% | █████████░░░░░░░░░░░ |
| Validation | 0/3 | 0% | ░░░░░░░░░░░░░░░░░░░░ |
| Viz | 5/15 | 33% | ██████░░░░░░░░░░░░░░ |

---

## Recommendations

### For Production Use (Current State)
✅ **READY FOR**:
- Standard data manipulation tasks
- Time series analysis
- Data cleaning and preprocessing
- Basic to intermediate analytics
- ETL pipelines
- Business intelligence reports

⚠️ **LIMITATIONS**:
- Advanced statistical testing (need Phase 12+)
- Complex timezone operations (need Phase 13)
- Advanced visualization (need Phase 15)
- Memory optimization for huge datasets (need Phase 14)

### Quick Wins (If Time is Limited)
If you can only implement a few more operations, prioritize these 5:

1. **Assert Operations** (3 ops) - Data quality validation
2. **Robust Scale** (1 op) - ML preprocessing
3. **Ordinal Encode** (1 op) - Categorical encoding

These would bring coverage to 63% with minimal effort.

---

## Conclusion

**Current Status**: 61% coverage (154/250 operations)
**Phase 11 Achievement**: +26 operations, +10% coverage
**Remaining Work**: 96 operations across 4 phases

The Noeta DSL is now **production-ready** for most common data manipulation tasks. The remaining gaps are primarily:
- Specialized operations (trigonometry, advanced stats)
- Optimization features (memory management)
- Additional convenience functions (more viz types)

**Next Priority**: If continuing, implement Phase 12 (13 medium-priority operations) to reach 67% coverage.
