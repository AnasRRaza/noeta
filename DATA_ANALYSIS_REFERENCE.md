# Comprehensive Data Analysis Reference for DSL Development

**Version**: 1.0
**Last Updated**: 2025-12-02
**Purpose**: Complete reference for all data analysis operations a DSL should support

---

## Table of Contents

### Part 1: Central Tendency Measures
1. [Mean](#11-mean)
2. [Median](#12-median)
3. [Mode](#13-mode)

### Part 2: Dispersion Measures
4. [Variance](#21-variance)
5. [Standard Deviation](#22-standard-deviation)
6. [Median Absolute Deviation](#23-median-absolute-deviation)
7. [Interquartile Range](#24-interquartile-range)
8. [Range](#25-range)
9. [Coefficient of Variation](#26-coefficient-of-variation)

### Part 3: Shape Measures
10. [Skewness](#31-skewness)
11. [Kurtosis](#32-kurtosis)
12. [Moments](#33-moments)

### Part 4: Summary Statistics
13. [Descriptive Summary](#41-descriptive-summary)
14. [Five Number Summary](#42-five-number-summary)
15. [Percentiles & Quantiles](#43-percentiles-quantiles)

### Part 5: Correlation & Association Analysis
16. [Pearson Correlation](#51-pearson-correlation)
17. [Spearman Correlation](#52-spearman-correlation)
18. [Kendall's Tau](#53-kendalls-tau)
19. [Partial Correlation](#54-partial-correlation)
20. [Canonical Correlation](#55-canonical-correlation)
21. [Distance Correlation](#56-distance-correlation)
22. [Point-Biserial Correlation](#57-point-biserial-correlation)
23. [Phi Coefficient](#58-phi-coefficient)
24. [Cramér's V](#59-cramers-v)
25. [Mutual Information](#510-mutual-information)

### Part 6: Hypothesis Testing - Parametric Tests
26. [One-Sample T-Test](#61-one-sample-t-test)
27. [Two-Sample T-Test](#62-two-sample-t-test)
28. [Paired T-Test](#63-paired-t-test)
29. [One-Way ANOVA](#64-one-way-anova)
30. [Two-Way ANOVA](#65-two-way-anova)
31. [Repeated Measures ANOVA](#66-repeated-measures-anova)
32. [ANCOVA](#67-ancova)
33. [F-Tests for Variance](#68-f-tests-for-variance)
34. [Z-Tests](#69-z-tests)

### Part 7: Hypothesis Testing - Non-Parametric Tests
35. [Mann-Whitney U Test](#71-mann-whitney-u-test)
36. [Wilcoxon Signed-Rank Test](#72-wilcoxon-signed-rank-test)
37. [Kruskal-Wallis Test](#73-kruskal-wallis-test)
38. [Friedman Test](#74-friedman-test)
39. [Sign Test](#75-sign-test)
40. [Runs Test](#76-runs-test)
41. [Kolmogorov-Smirnov Test](#77-kolmogorov-smirnov-test)

### Part 8: Hypothesis Testing - Normality & Distribution Tests
42. [Shapiro-Wilk Test](#81-shapiro-wilk-test)
43. [Anderson-Darling Test](#82-anderson-darling-test)
44. [Jarque-Bera Test](#83-jarque-bera-test)
45. [D'Agostino-Pearson Test](#84-dagostino-pearson-test)
46. [Lilliefors Test](#85-lilliefors-test)
47. [Chi-Square Goodness-of-Fit](#86-chi-square-goodness-of-fit)

### Part 9: Hypothesis Testing - Variance & Homogeneity Tests
48. [Levene's Test](#91-levenes-test)
49. [Bartlett's Test](#92-bartletts-test)
50. [Brown-Forsythe Test](#93-brown-forsythe-test)
51. [Fligner-Killeen Test](#94-fligner-killeen-test)

### Part 10: Hypothesis Testing - Independence & Categorical Tests
52. [Chi-Square Test of Independence](#101-chi-square-test-of-independence)
53. [Fisher's Exact Test](#102-fishers-exact-test)
54. [McNemar's Test](#103-mcnemars-test)
55. [Cochran's Q Test](#104-cochrans-q-test)
56. [Mantel-Haenszel Test](#105-mantel-haenszel-test)

### Part 11: Multiple Comparison Corrections
57. [Bonferroni Correction](#111-bonferroni-correction)
58. [Holm-Bonferroni Method](#112-holm-bonferroni-method)
59. [Benjamini-Hochberg Procedure](#113-benjamini-hochberg-procedure)
60. [Tukey's HSD](#114-tukeys-hsd)
61. [Dunnett's Test](#115-dunnetts-test)
62. [Scheffe's Method](#116-scheffes-method)

### Part 12: Distribution Analysis & Fitting
63. [Distribution Fitting](#121-distribution-fitting)
64. [Maximum Likelihood Estimation](#122-maximum-likelihood-estimation)
65. [Method of Moments](#123-method-of-moments)
66. [Probability Density Function](#124-probability-density-function)
67. [Cumulative Distribution Function](#125-cumulative-distribution-function)
68. [Quantile Function](#126-quantile-function)
69. [Empirical Distribution Function](#127-empirical-distribution-function)
70. [Kernel Density Estimation](#128-kernel-density-estimation)

### Part 13: Regression Analysis - Linear Models
71. [Simple Linear Regression](#131-simple-linear-regression)
72. [Multiple Linear Regression](#132-multiple-linear-regression)
73. [Polynomial Regression](#133-polynomial-regression)
74. [Weighted Least Squares](#134-weighted-least-squares)
75. [Generalized Least Squares](#135-generalized-least-squares)

### Part 14: Regression Analysis - Advanced Linear Models
76. [Ridge Regression](#141-ridge-regression)
77. [Lasso Regression](#142-lasso-regression)
78. [Elastic Net Regression](#143-elastic-net-regression)
79. [Principal Component Regression](#144-principal-component-regression)
80. [Partial Least Squares Regression](#145-partial-least-squares-regression)

### Part 15: Regression Analysis - Non-Linear & Robust
81. [Non-Linear Regression](#151-non-linear-regression)
82. [Robust Regression](#152-robust-regression)
83. [Quantile Regression](#153-quantile-regression)
84. [Isotonic Regression](#154-isotonic-regression)

### Part 16: Regression Analysis - Generalized Models
85. [Logistic Regression](#161-logistic-regression)
86. [Poisson Regression](#162-poisson-regression)
87. [Negative Binomial Regression](#163-negative-binomial-regression)
88. [Gamma Regression](#164-gamma-regression)
89. [Beta Regression](#165-beta-regression)

### Part 17: Regression Diagnostics
90. [Residual Analysis](#171-residual-analysis)
91. [Influence Diagnostics](#172-influence-diagnostics)
92. [Multicollinearity Detection](#173-multicollinearity-detection)
93. [Heteroscedasticity Tests](#174-heteroscedasticity-tests)
94. [Autocorrelation Tests](#175-autocorrelation-tests)
95. [Specification Tests](#176-specification-tests)

### Part 18: Time Series Analysis - Decomposition
96. [Time Series Decomposition](#181-time-series-decomposition)
97. [Seasonal Decomposition](#182-seasonal-decomposition)
98. [Trend Extraction](#183-trend-extraction)
99. [Deseasonalization](#184-deseasonalization)

### Part 19: Time Series Analysis - Stationarity
100. [Augmented Dickey-Fuller Test](#191-augmented-dickey-fuller-test)
101. [KPSS Test](#192-kpss-test)
102. [Phillips-Perron Test](#193-phillips-perron-test)
103. [Differencing Operations](#194-differencing-operations)

### Part 20: Time Series Analysis - Autocorrelation
104. [Autocorrelation Function](#201-autocorrelation-function)
105. [Partial Autocorrelation Function](#202-partial-autocorrelation-function)
106. [Cross-Correlation Function](#203-cross-correlation-function)
107. [Ljung-Box Test](#204-ljung-box-test)
108. [Durbin-Watson Test](#205-durbin-watson-test)

### Part 21: Time Series Analysis - Modeling
109. [ARIMA Modeling](#211-arima-modeling)
110. [SARIMA Modeling](#212-sarima-modeling)
111. [ARIMAX Modeling](#213-arimax-modeling)
112. [VAR Models](#214-var-models)
113. [GARCH Models](#215-garch-models)

### Part 22: Time Series Analysis - Smoothing & Forecasting
114. [Exponential Smoothing](#221-exponential-smoothing)
115. [Holt-Winters Method](#222-holt-winters-method)
116. [Moving Average Smoothing](#223-moving-average-smoothing)
117. [Forecasting Methods](#224-forecasting-methods)
118. [Forecast Accuracy Metrics](#225-forecast-accuracy-metrics)

### Part 23: Time Series Analysis - Advanced Topics
119. [Change Point Detection](#231-change-point-detection)
120. [Structural Break Tests](#232-structural-break-tests)
121. [Cointegration Tests](#233-cointegration-tests)
122. [Granger Causality](#234-granger-causality)
123. [Spectral Analysis](#235-spectral-analysis)

### Part 24: Multivariate Analysis
124. [MANOVA](#241-manova)
125. [Linear Discriminant Analysis](#242-linear-discriminant-analysis)
126. [Quadratic Discriminant Analysis](#243-quadratic-discriminant-analysis)
127. [Canonical Correlation Analysis](#244-canonical-correlation-analysis)
128. [Multivariate Regression](#245-multivariate-regression)
129. [Hotelling's T-Square](#246-hotellings-t-square)

### Part 25: Dimensionality Reduction
130. [Principal Component Analysis](#251-principal-component-analysis)
131. [Factor Analysis](#252-factor-analysis)
132. [Independent Component Analysis](#253-independent-component-analysis)
133. [Multidimensional Scaling](#254-multidimensional-scaling)
134. [Correspondence Analysis](#255-correspondence-analysis)

### Part 26: Clustering Analysis
135. [K-Means Clustering](#261-k-means-clustering)
136. [Hierarchical Clustering](#262-hierarchical-clustering)
137. [DBSCAN](#263-dbscan)
138. [Gaussian Mixture Models](#264-gaussian-mixture-models)
139. [Cluster Validation](#265-cluster-validation)
140. [Optimal Cluster Number](#266-optimal-cluster-number)

### Part 27: Survival Analysis
141. [Kaplan-Meier Estimation](#271-kaplan-meier-estimation)
142. [Cox Proportional Hazards](#272-cox-proportional-hazards)
143. [Log-Rank Test](#273-log-rank-test)
144. [Accelerated Failure Time Models](#274-accelerated-failure-time-models)
145. [Competing Risks Analysis](#275-competing-risks-analysis)

### Part 28: Bayesian Analysis
146. [Bayesian Inference](#281-bayesian-inference)
147. [Prior Distribution Specification](#282-prior-distribution-specification)
148. [Posterior Distribution](#283-posterior-distribution)
149. [MCMC Methods](#284-mcmc-methods)
150. [Gibbs Sampling](#285-gibbs-sampling)
151. [Metropolis-Hastings](#286-metropolis-hastings)
152. [Bayesian Model Comparison](#287-bayesian-model-comparison)

### Part 29: Experimental Design & Power Analysis
153. [Sample Size Determination](#291-sample-size-determination)
154. [Power Analysis](#292-power-analysis)
155. [Effect Size Calculation](#293-effect-size-calculation)
156. [Randomization Methods](#294-randomization-methods)
157. [Blocking Designs](#295-blocking-designs)
158. [Factorial Designs](#296-factorial-designs)

### Part 30: Categorical Data Analysis
159. [Contingency Tables](#301-contingency-tables)
160. [Odds Ratio](#302-odds-ratio)
161. [Relative Risk](#303-relative-risk)
162. [Log-Linear Models](#304-log-linear-models)
163. [Multinomial Models](#305-multinomial-models)

### Part 31: Outlier & Anomaly Detection
164. [Z-Score Method](#311-z-score-method)
165. [Modified Z-Score](#312-modified-z-score)
166. [IQR Method](#313-iqr-method)
167. [Grubbs' Test](#314-grubbs-test)
168. [Dixon's Q Test](#315-dixons-q-test)
169. [Isolation Forest](#316-isolation-forest)
170. [Local Outlier Factor](#317-local-outlier-factor)

### Part 32: Robust Statistics
171. [Robust Location Estimates](#321-robust-location-estimates)
172. [Robust Scale Estimates](#322-robust-scale-estimates)
173. [Winsorization](#323-winsorization)
174. [Trimming](#324-trimming)
175. [M-Estimators](#325-m-estimators)

### Part 33: Feature Selection & Engineering
176. [Univariate Feature Selection](#331-univariate-feature-selection)
177. [Recursive Feature Elimination](#332-recursive-feature-elimination)
178. [Feature Importance](#333-feature-importance)
179. [Interaction Detection](#334-interaction-detection)
180. [Polynomial Features](#335-polynomial-features)

### Part 34: Model Validation & Selection
181. [Cross-Validation](#341-cross-validation)
182. [Bootstrap Methods](#342-bootstrap-methods)
183. [Information Criteria](#343-information-criteria)
184. [Model Comparison](#344-model-comparison)
185. [Goodness-of-Fit Measures](#345-goodness-of-fit-measures)

### Part 35: Resampling Methods
186. [Permutation Tests](#351-permutation-tests)
187. [Bootstrap Confidence Intervals](#352-bootstrap-confidence-intervals)
188. [Jackknife Methods](#353-jackknife-methods)
189. [Monte Carlo Simulation](#354-monte-carlo-simulation)

### Part 36: Advanced Statistical Methods
190. [Mixed Effects Models](#361-mixed-effects-models)
191. [Hierarchical Models](#362-hierarchical-models)
192. [Structural Equation Modeling](#363-structural-equation-modeling)
193. [Propensity Score Methods](#364-propensity-score-methods)
194. [Causal Inference](#365-causal-inference)
195. [Meta-Analysis](#366-meta-analysis)

### Appendices
- [Appendix A: Statistical Distributions Reference](#appendix-a-statistical-distributions-reference)
- [Appendix B: Test Statistics Reference](#appendix-b-test-statistics-reference)
- [Appendix C: Effect Size Measures Reference](#appendix-c-effect-size-measures-reference)
- [Appendix D: Model Evaluation Metrics Reference](#appendix-d-model-evaluation-metrics-reference)
- [Appendix E: Alphabetical Function Index](#appendix-e-alphabetical-function-index)

---

# Part 1: Central Tendency Measures

## 1.1 MEAN

#### Purpose
Calculate the arithmetic mean (average) of a dataset, representing the central location of the data distribution.

#### Mathematical Specification
For a sample of size n with observations x₁, x₂, ..., xₙ:

**Sample Mean:**
```
x̄ = (1/n) Σᵢ₌₁ⁿ xᵢ
```

**Population Mean:**
```
μ = (1/N) Σᵢ₌₁ᴺ xᵢ
```

**Weighted Mean:**
```
x̄w = Σᵢ₌₁ⁿ (wᵢ × xᵢ) / Σᵢ₌₁ⁿ wᵢ
```

**Trimmed Mean:**
```
x̄trim = (1/(n-2k)) Σᵢ₌ₖ₊₁ⁿ⁻ᵏ x₍ᵢ₎
where x₍ᵢ₎ are order statistics and k is number of trimmed values from each tail
```

#### Syntax Variations
```
mean data column="price" as avg_price
mean data columns=["price","quantity"] type="arithmetic" as result
mean data column="salary" type="trimmed" trim_percent=0.1 as trimmed_mean
mean data column="score" weighted_by="frequency" as weighted_avg
mean data columns=all numeric=true skipna=true as means
```

#### Parameters

##### column / columns
- **Type**: `string | list[string] | int | list[int] | pattern | "all"`
- **Required**: Yes
- **Valid Values**:
  - Single column: `"price"`, `0`
  - Multiple columns: `["price", "quantity"]`, `[0, 1, 2]`
  - Pattern: `pattern("num_*")`, `pattern("*_score")`
  - All numeric: `"all"`
- **Behavior**: Specifies which column(s) to calculate mean for
- **Edge Cases**:
  - Non-numeric columns raise TypeError
  - Empty selection raises ValueError
  - Single value returns that value
  - All NaN values return NaN (if skipna=True) or error (if skipna=False)

##### type
- **Type**: `string`
- **Required**: No
- **Default**: `"arithmetic"`
- **Valid Values**:
  - `"arithmetic"`: Standard arithmetic mean
  - `"geometric"`: Geometric mean (nth root of product)
  - `"harmonic"`: Harmonic mean (reciprocal of mean of reciprocals)
  - `"trimmed"`: Trimmed mean (remove outliers from tails)
  - `"winsorized"`: Winsorized mean (replace tail values with percentile values)
  - `"weighted"`: Weighted mean (requires weights parameter)
- **Behavior**: Determines the type of mean calculation
- **Mathematical Specifications**:
  - **Geometric**: `(∏ᵢ₌₁ⁿ xᵢ)^(1/n)` - appropriate for rates of change, ratios
  - **Harmonic**: `n / Σᵢ₌₁ⁿ (1/xᵢ)` - appropriate for rates and speeds
  - **Trimmed**: Remove k% from each tail before calculating
  - **Winsorized**: Replace tail values beyond kth percentile
- **Statistical Assumptions**:
  - Arithmetic: No special assumptions
  - Geometric: Requires all positive values
  - Harmonic: Requires all non-zero values
  - Trimmed/Winsorized: Assumes symmetric distribution for unbiased estimation
- **Edge Cases**:
  - Geometric with zero or negative values raises error
  - Harmonic with zero values raises error
  - Trimmed with trim_percent >= 0.5 raises error

##### skipna
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to exclude NaN values from calculation
- **Edge Cases**:
  - If false and NaN present, result is NaN
  - If true and all values NaN, result is NaN
  - Empty result after removing NaN raises error if skipna=true

##### weighted_by
- **Type**: `string | list[number]`
- **Required**: No (required if type="weighted")
- **Default**: `null`
- **Valid Values**:
  - Column name: `"frequency"`, `"weight"`
  - List of weights: `[0.5, 0.3, 0.2]`
- **Behavior**: Specifies weights for weighted mean calculation
- **Edge Cases**:
  - Weight length mismatch with data raises error
  - Negative weights raise warning
  - Zero sum of weights raises error
  - NaN in weights handled according to skipna parameter

##### trim_percent
- **Type**: `float`
- **Required**: No (required if type="trimmed" or "winsorized")
- **Default**: `0.1` (10%)
- **Valid Values**: `0.0 < trim_percent < 0.5`
- **Behavior**: Proportion of data to trim/winsorize from each tail
- **Edge Cases**:
  - Value >= 0.5 raises error (would remove all data)
  - Value = 0 equivalent to arithmetic mean
  - With small sample size, may trim more than intended due to rounding

##### axis
- **Type**: `int | string`
- **Required**: No
- **Default**: `0` (by rows)
- **Valid Values**: `0` (rows), `1` (columns), `"index"`, `"columns"`
- **Behavior**: Direction along which to calculate mean
- **Edge Cases**:
  - Invalid axis value raises error
  - For Series input, axis is ignored

##### numeric_only
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to include only numeric columns
- **Edge Cases**:
  - If false, attempts to compute mean on non-numeric data (may fail)
  - If true, silently excludes non-numeric columns

##### ddof
- **Type**: `int`
- **Required**: No
- **Default**: `0`
- **Valid Values**: Non-negative integers
- **Behavior**: Delta degrees of freedom (used in variance calculations for certain mean types)
- **Edge Cases**:
  - Only applicable for certain advanced mean calculations
  - Value >= sample size raises error

#### Return Values
- **Type**: `float | Series | DataFrame`
- **Single Column**: Returns scalar float
- **Multiple Columns**: Returns Series with column names as index
- **2D Calculation**: Returns Series or DataFrame depending on axis

#### Statistical Properties
- **Expectation**: E[x̄] = μ (unbiased estimator of population mean)
- **Variance**: Var(x̄) = σ²/n (decreases with sample size)
- **Distribution**: By Central Limit Theorem, x̄ ~ N(μ, σ²/n) for large n
- **Efficiency**: Most efficient estimator for normal distributions
- **Robustness**:
  - Arithmetic mean: NOT robust to outliers
  - Trimmed mean: More robust, efficiency ≈ 95% of arithmetic mean for normal data
  - Winsorized mean: Robust to outliers while retaining more information than trimmed

#### Statistical Assumptions
- **Arithmetic Mean**:
  - No distribution assumptions required
  - Optimal for symmetric, normal distributions
- **Geometric Mean**:
  - All values must be positive
  - Appropriate for log-normal distributions
- **Harmonic Mean**:
  - All values must be non-zero
  - Appropriate for rates and ratios
- **Trimmed/Winsorized**:
  - Symmetric distribution for unbiased estimation
  - Heavy-tailed distributions benefit most

#### Interpretation Guidelines
- **Arithmetic Mean**: Center of mass of distribution, affected by all values equally
- **Geometric Mean**: Multiplicative center, useful for growth rates and ratios
- **Harmonic Mean**: Appropriate for averaging rates (e.g., speed, productivity)
- **Trimmed Mean**: Robust estimate of central tendency in presence of outliers
- **Effect of Outliers**: Arithmetic mean is highly sensitive; consider trimmed/winsorized for skewed data

#### Common Use Cases
1. **Arithmetic Mean**: General purpose central tendency, financial averaging, scientific measurements
2. **Geometric Mean**: Investment returns, growth rates, index calculations
3. **Harmonic Mean**: Average speed/velocity, rates of work, harmonic oscillation
4. **Trimmed Mean**: Economic indicators (e.g., CPI calculations), robust averaging
5. **Weighted Mean**: Survey data with sampling weights, portfolio returns

#### Related Functions
- [MEDIAN](#12-median) - More robust alternative to mean
- [MODE](#13-mode) - Most frequent value
- [TRIMMED_MEAN](#159-robust-location-estimates) - Specialized robust mean
- [STANDARD_DEVIATION](#21-standard-deviation) - Measure of spread around mean

---

## 1.2 MEDIAN

#### Purpose
Calculate the median (50th percentile) of a dataset, representing the middle value that separates the higher half from the lower half of the data distribution.

#### Mathematical Specification
For a sorted sample of size n:

**Odd sample size (n = 2k+1):**
```
median = x₍ₖ₊₁₎
```

**Even sample size (n = 2k):**
```
median = (x₍ₖ₎ + x₍ₖ₊₁₎) / 2
```

Where x₍ᵢ₎ represents the ith order statistic (sorted values).

**Weighted Median:**
For weighted observations (xᵢ, wᵢ), the weighted median is the value xₖ where:
```
Σᵢ₌₁ᵏ⁻¹ wᵢ < 0.5 × Σᵢ₌₁ⁿ wᵢ ≤ Σᵢ₌₁ᵏ wᵢ
```

#### Syntax Variations
```
median data column="price" as med_price
median data columns=["price","quantity"] as result
median data column="salary" interpolation="linear" as med_salary
median data column="score" weighted_by="frequency" as weighted_median
median data columns=all numeric=true skipna=true as medians
```

#### Parameters

##### column / columns
- **Type**: `string | list[string] | int | list[int] | pattern | "all"`
- **Required**: Yes
- **Valid Values**:
  - Single column: `"price"`, `0`
  - Multiple columns: `["price", "quantity"]`, `[0, 1, 2]`
  - Pattern: `pattern("num_*")`
  - All numeric: `"all"`
- **Behavior**: Specifies which column(s) to calculate median for
- **Edge Cases**:
  - Non-numeric columns raise TypeError
  - Empty selection raises ValueError
  - Single value returns that value
  - All NaN values return NaN

##### interpolation
- **Type**: `string`
- **Required**: No
- **Default**: `"linear"`
- **Valid Values**:
  - `"linear"`: Linear interpolation between two middle values (i + (j-i) × 0.5)
  - `"lower"`: Always use lower middle value (i)
  - `"higher"`: Always use higher middle value (j)
  - `"midpoint"`: (i + j) / 2 (same as linear for median)
  - `"nearest"`: Use nearest middle value
- **Behavior**: Method for calculating median when n is even
- **Mathematical Specification**:
  - For sorted values x₍ₖ₎ and x₍ₖ₊₁₎ when n = 2k:
    - linear: x₍ₖ₎ + 0.5 × (x₍ₖ₊₁₎ - x₍ₖ₎)
    - lower: x₍ₖ₎
    - higher: x₍ₖ₊₁₎
    - midpoint: (x₍ₖ₎ + x₍ₖ₊₁₎) / 2
    - nearest: x₍ₖ₎ if |x - x₍ₖ₎| < |x - x₍ₖ₊₁₎|, else x₍ₖ₊₁₎
- **Edge Cases**:
  - For odd n, interpolation method has no effect
  - "nearest" may arbitrarily choose one value when equidistant

##### skipna
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to exclude NaN values from calculation
- **Edge Cases**:
  - If false and NaN present, result is NaN
  - If true and all values NaN, result is NaN
  - Order of NaN values in sorting is undefined

##### weighted_by
- **Type**: `string | list[number]`
- **Required**: No
- **Default**: `null`
- **Valid Values**:
  - Column name: `"frequency"`, `"weight"`
  - List of weights: `[0.5, 0.3, 0.2]`
- **Behavior**: Specifies weights for weighted median calculation
- **Mathematical Note**: Weighted median is the value that minimizes sum of absolute weighted deviations
- **Edge Cases**:
  - Weight length mismatch raises error
  - Negative weights raise warning (absolute values used)
  - Zero weights are ignored
  - All zero weights raise error

##### axis
- **Type**: `int | string`
- **Required**: No
- **Default**: `0` (by rows)
- **Valid Values**: `0` (rows), `1` (columns), `"index"`, `"columns"`
- **Behavior**: Direction along which to calculate median
- **Edge Cases**:
  - Invalid axis value raises error
  - For Series input, axis is ignored

##### numeric_only
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to include only numeric columns
- **Edge Cases**:
  - If false, attempts median on non-numeric data (may fail)
  - If true, silently excludes non-numeric columns
  - DateTime columns can be included if numeric_only=false

##### method
- **Type**: `string`
- **Required**: No
- **Default**: `"quickselect"`
- **Valid Values**:
  - `"quickselect"`: O(n) average time selection algorithm
  - `"sort"`: O(n log n) full sort then select middle
  - `"heap"`: O(n log n) partial heap sort
- **Behavior**: Algorithm used for median calculation
- **Performance Characteristics**:
  - quickselect: Fastest for large datasets, O(n) average, O(n²) worst case
  - sort: Stable, useful if sorted data needed elsewhere, O(n log n) guaranteed
  - heap: Memory efficient, O(n log n) guaranteed
- **Edge Cases**:
  - All methods produce identical results (different performance only)
  - For small n (<100), differences negligible

#### Return Values
- **Type**: `float | Series | DataFrame`
- **Single Column**: Returns scalar float
- **Multiple Columns**: Returns Series with column names as index
- **2D Calculation**: Returns Series or DataFrame depending on axis

#### Statistical Properties
- **Expectation**: For symmetric distributions, E[median] = μ
- **Variance**: Var(median) ≈ (π/2) × (σ²/n) for normal distribution
  - Asymptotic variance: 1/(4n × f(m)²) where f is PDF at median m
- **Distribution**: Asymptotically normal by CLT
- **Efficiency**:
  - Relative efficiency vs mean: ≈ 64% for normal distribution
  - Efficiency increases for heavy-tailed distributions
- **Robustness**:
  - Breakdown point: 50% (can tolerate up to 50% outliers)
  - Highly robust to extreme values
  - Does not depend on extreme observations

#### Statistical Assumptions
- **Distribution**: No assumptions required (distribution-free)
- **Scale**: Requires ordinal or higher level of measurement
- **Independence**: Assumes independent observations (for inference)
- **Continuity**: For continuous data, probability of exact ties is zero
- **Ties**: For discrete data with ties, may not uniquely partition data

#### Interpretation Guidelines
- **Location**: Median divides distribution into two equal halves
- **Comparison with Mean**:
  - Mean > Median: Right-skewed distribution (positive skew)
  - Mean < Median: Left-skewed distribution (negative skew)
  - Mean ≈ Median: Symmetric distribution
- **Robustness**: Preferred over mean for skewed distributions or presence of outliers
- **Percentile Interpretation**: Median is the 50th percentile (Q2, second quartile)

#### Common Use Cases
1. **Income/Salary Data**: Median income more representative than mean due to skewness
2. **Housing Prices**: Median price less affected by luxury properties
3. **Robust Central Tendency**: When outliers present or distribution unknown
4. **Ordinal Data**: Appropriate for ordered categorical data
5. **Skewed Distributions**: More representative of typical value than mean

#### Related Functions
- [MEAN](#11-mean) - Alternative measure of central tendency
- [MODE](#13-mode) - Most frequent value
- [QUANTILE](#55-quantile-function) - Generalization to other percentiles
- [MAD](#23-median-absolute-deviation) - Robust dispersion measure based on median
- [QUARTILES](#45-quartiles) - 25th, 50th (median), and 75th percentiles

---

## 1.3 MODE

#### Purpose
Calculate the mode of a dataset, representing the most frequently occurring value(s) in the distribution.

#### Mathematical Specification
For a dataset with values {x₁, x₂, ..., xₙ}:

**Mode:**
```
mode = argmax(fᵢ) where fᵢ is the frequency of value xᵢ
```

**Modal Frequency:**
```
f_mode = max(f₁, f₂, ..., fₖ) where k is number of unique values
```

**Relative Frequency:**
```
p_mode = f_mode / n
```

**Multimodality Detection:**
- **Unimodal**: One mode
- **Bimodal**: Two modes with equal maximum frequency
- **Multimodal**: More than two modes with equal maximum frequency
- **Uniform**: All values equally frequent (no distinct mode)

#### Syntax Variations
```
mode data column="category" as most_common
mode data columns=["product","region"] return="all" as modes
mode data column="score" keep="first" as single_mode
mode data column="value" min_frequency=5 as filtered_mode
mode data columns=all dropna=true as all_modes
```

#### Parameters

##### column / columns
- **Type**: `string | list[string] | int | list[int] | pattern | "all"`
- **Required**: Yes
- **Valid Values**:
  - Single column: `"category"`, `0`
  - Multiple columns: `["product", "region"]`, `[0, 1]`
  - Pattern: `pattern("cat_*")`
  - All: `"all"`
- **Behavior**: Specifies which column(s) to calculate mode for
- **Edge Cases**:
  - Can be applied to both numeric and categorical data
  - Empty selection raises ValueError
  - Single value is the mode (frequency = 1)

##### keep
- **Type**: `string`
- **Required**: No
- **Default**: `"all"`
- **Valid Values**:
  - `"all"`: Return all modes if multiple exist
  - `"first"`: Return first mode encountered (in sorted order)
  - `"last"`: Return last mode encountered (in sorted order)
  - `"smallest"`: Return smallest mode value
  - `"largest"`: Return largest mode value
- **Behavior**: How to handle multiple modes (multimodal distribution)
- **Edge Cases**:
  - For multimodal data with keep="first", depends on data ordering
  - "smallest"/"largest" only applicable to numeric/ordinal data
  - For categorical data, "smallest"/"largest" use lexicographic ordering

##### return_frequency
- **Type**: `bool`
- **Required**: No
- **Default**: `false`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to return frequency count along with mode value
- **Return Format**:
  - If true: Returns tuple/dict with (value, frequency)
  - If false: Returns only mode value(s)
- **Edge Cases**:
  - Useful for assessing modality strength
  - Can identify uniform distributions (all frequencies equal)

##### min_frequency
- **Type**: `int`
- **Required**: No
- **Default**: `1`
- **Valid Values**: Positive integers
- **Behavior**: Minimum frequency required for a value to be considered a mode
- **Edge Cases**:
  - If no value meets threshold, returns NaN or empty result
  - Useful for filtering spurious modes in large datasets
  - Value > max frequency returns no mode

##### dropna
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to exclude NaN values from mode calculation
- **Edge Cases**:
  - If false, NaN treated as a distinct value
  - If false and NaN is most frequent, NaN is returned as mode
  - For categorical data with missing category, behavior depends on this parameter

##### numeric_only
- **Type**: `bool`
- **Required**: No
- **Default**: `false`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to include only numeric columns
- **Note**: Mode is one of few statistics applicable to categorical data
- **Edge Cases**:
  - If true, excludes all non-numeric columns
  - If false, mode calculated for all data types

##### bin_numeric
- **Type**: `bool | int | list[float]`
- **Required**: No
- **Default**: `false`
- **Valid Values**:
  - `false`: No binning (exact values)
  - `true`: Auto-bin numeric data (uses Freedman-Diaconis rule)
  - Integer: Number of bins
  - List: Bin edges
- **Behavior**: Whether to bin continuous numeric data before finding mode
- **Mathematical Note**: For continuous data, modal class more meaningful than exact mode
- **Edge Cases**:
  - Only applicable to numeric data
  - Binning can create/eliminate modes
  - Auto-binning may not work well for multimodal distributions

##### relative_frequency
- **Type**: `bool`
- **Required**: No
- **Default**: `false`
- **Valid Values**: `true`, `false`
- **Behavior**: Return relative frequency (proportion) instead of count
- **Mathematical Specification**: p = f_mode / n
- **Edge Cases**:
  - Only applicable if return_frequency=true
  - Value range: (0, 1]

##### detect_multimodality
- **Type**: `bool`
- **Required**: No
- **Default**: `false`
- **Valid Values**: `true`, `false`
- **Behavior**: Detect and report distribution modality type
- **Return Values**:
  - "unimodal": Single mode
  - "bimodal": Two modes
  - "multimodal": >2 modes
  - "uniform": No distinct mode
- **Edge Cases**:
  - Requires frequency threshold to avoid spurious multimodality
  - Sensitive to sample size and data discretization

#### Return Values
- **Type**: `value | list[value] | Series | DataFrame | dict`
- **Single Column, Single Mode**: Scalar value
- **Single Column, Multiple Modes**: List of values
- **Multiple Columns**: Series with column names as index
- **With Frequency**: Dictionary or DataFrame with value and frequency

#### Statistical Properties
- **Existence**:
  - May not exist (uniform distribution)
  - May not be unique (multimodal distribution)
- **Uniqueness**: Not guaranteed for any distribution
- **Efficiency**:
  - 0% efficiency for continuous distributions (theoretically)
  - Can be quite efficient for highly discrete distributions
- **Robustness**:
  - Highly robust to outliers (only depends on frequencies)
  - Breakdown point: Near 50% for unimodal data
- **Sample Variability**:
  - High variance in small samples
  - May jump between values as sample size changes

#### Statistical Assumptions
- **Data Type**: Can be applied to any measurement level (nominal, ordinal, interval, ratio)
- **Distribution**: No distributional assumptions
- **Independence**: Assumes independent observations for interpretation
- **Discreteness**: Most meaningful for discrete or discretized data
- **Sample Size**: Requires sufficient sample size for stable estimation

#### Interpretation Guidelines
- **Categorical Data**: Mode is the only appropriate measure of central tendency
- **Numerical Data**:
  - Mode may not be representative for continuous distributions
  - Consider binning continuous data
  - Mode location relative to mean/median indicates skewness
- **Multimodality**:
  - May indicate mixture of populations
  - Suggests need for stratified analysis
  - Common in clustered data
- **Mode vs Mean/Median**:
  - For symmetric unimodal: Mode ≈ Mean ≈ Median
  - For right-skewed: Mode < Median < Mean
  - For left-skewed: Mean < Median < Mode

#### Common Use Cases
1. **Categorical Data**: Most common category (e.g., most frequent product, most common diagnosis)
2. **Discrete Numerical Data**: Most common value (e.g., most common household size)
3. **Quality Control**: Most frequent defect type
4. **Marketing**: Most popular product/size/color
5. **Survey Data**: Most common response category
6. **Mixture Detection**: Identifying subpopulations through multimodality

#### Related Functions
- [MEAN](#11-mean) - Arithmetic average
- [MEDIAN](#12-median) - Middle value
- [FREQUENCY_TABLE](#146-contingency-tables) - Full frequency distribution
- [UNIQUE_COUNTS](#4-summary-statistics) - Count of each unique value
- [DISTRIBUTION_SHAPE](#3-shape-measures) - Modality detection and characterization

---

# Part 2: Dispersion Measures

## 2.1 VARIANCE

#### Purpose
Calculate the variance of a dataset, measuring the average squared deviation from the mean and quantifying the spread or dispersion of the data distribution.

#### Mathematical Specification
For a dataset with observations x₁, x₂, ..., xₙ:

**Population Variance:**
```
σ² = (1/N) Σᵢ₌₁ᴺ (xᵢ - μ)²
where μ is the population mean and N is the population size
```

**Sample Variance (with Bessel's Correction):**
```
s² = (1/(n-1)) Σᵢ₌₁ⁿ (xᵢ - x̄)²
where x̄ is the sample mean and n is the sample size
```

**Weighted Variance:**
```
σ²w = Σᵢ₌₁ⁿ wᵢ(xᵢ - μw)² / Σᵢ₌₁ⁿ wᵢ
where μw is the weighted mean
```

**Computational Formula (numerically stable):**
```
s² = (Σᵢ₌₁ⁿ xᵢ²  - (Σᵢ₌₁ⁿ xᵢ)²/n) / (n-1)
```

#### Syntax Variations
```
variance data column="price" as price_var
variance data columns=["price","quantity"] type="sample" as result
variance data column="salary" ddof=1 as sample_variance
variance data column="score" weighted_by="frequency" as weighted_var
variance data columns=all numeric=true skipna=true as variances
variance data column="values" type="population" ddof=0 as pop_var
```

#### Parameters

##### column / columns
- **Type**: `string | list[string] | int | list[int] | pattern | "all"`
- **Required**: Yes
- **Valid Values**:
  - Single column: `"price"`, `0`
  - Multiple columns: `["price", "quantity"]`, `[0, 1, 2]`
  - Pattern: `pattern("num_*")`, `pattern("*_score")`
  - All numeric: `"all"`
- **Behavior**: Specifies which column(s) to calculate variance for
- **Edge Cases**:
  - Non-numeric columns raise TypeError
  - Empty selection raises ValueError
  - Single value returns 0.0 (or NaN depending on ddof)
  - All NaN values return NaN (if skipna=True) or error (if skipna=False)

##### type
- **Type**: `string`
- **Required**: No
- **Default**: `"sample"`
- **Valid Values**:
  - `"sample"`: Sample variance with Bessel's correction (ddof=1)
  - `"population"`: Population variance (ddof=0)
  - `"weighted"`: Weighted variance (requires weights parameter)
- **Behavior**: Determines the type of variance calculation
- **Mathematical Specifications**:
  - **Sample**: Unbiased estimator of population variance, divides by (n-1)
  - **Population**: Assumes data represents entire population, divides by n
  - **Weighted**: Uses weights to account for different observation importance
- **Statistical Note**: Sample variance is unbiased: E[s²] = σ²
- **Edge Cases**:
  - For n=1, sample variance is undefined (division by zero)
  - Population variance with n=1 returns 0

##### ddof
- **Type**: `int`
- **Required**: No
- **Default**: `1` (for sample variance)
- **Valid Values**: Non-negative integers, typically 0 or 1
- **Behavior**: Delta Degrees of Freedom - divisor is (n - ddof)
- **Mathematical Specification**: Variance = Σ(xᵢ - x̄)² / (n - ddof)
- **Common Values**:
  - `ddof=0`: Population variance (biased estimator)
  - `ddof=1`: Sample variance (unbiased estimator, Bessel's correction)
  - `ddof=2`: Used in some specialized applications
- **Edge Cases**:
  - If ddof >= n, returns inf or raises error
  - If ddof < 0, raises ValueError
  - For weighted variance, ddof interpretation differs

##### skipna
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to exclude NaN values from calculation
- **Edge Cases**:
  - If false and NaN present, result is NaN
  - If true and all values NaN, result is NaN
  - Empty result after removing NaN raises error if count < ddof + 1

##### weighted_by
- **Type**: `string | list[number]`
- **Required**: No (required if type="weighted")
- **Default**: `null`
- **Valid Values**:
  - Column name: `"frequency"`, `"weight"`, `"importance"`
  - List of weights: `[0.5, 0.3, 0.2]`
- **Behavior**: Specifies weights for weighted variance calculation
- **Mathematical Note**: Weighted variance accounts for heterogeneous observation reliability
- **Edge Cases**:
  - Weight length mismatch with data raises error
  - Negative weights raise warning or error
  - Zero sum of weights raises error
  - NaN in weights handled according to skipna parameter

##### axis
- **Type**: `int | string`
- **Required**: No
- **Default**: `0` (by rows)
- **Valid Values**: `0` (rows), `1` (columns), `"index"`, `"columns"`
- **Behavior**: Direction along which to calculate variance
- **Edge Cases**:
  - Invalid axis value raises error
  - For Series input, axis is ignored

##### numeric_only
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to include only numeric columns
- **Edge Cases**:
  - If false, attempts to compute variance on non-numeric data (will fail)
  - If true, silently excludes non-numeric columns

##### method
- **Type**: `string`
- **Required**: No
- **Default**: `"two_pass"`
- **Valid Values**:
  - `"two_pass"`: Standard two-pass algorithm (calculate mean first)
  - `"welford"`: Welford's online algorithm (numerically stable, single pass)
  - `"naive"`: Direct computation (can have numerical instability)
- **Behavior**: Algorithm used for variance calculation
- **Numerical Stability**:
  - two_pass: Good stability, requires two passes through data
  - welford: Excellent stability, online algorithm, single pass
  - naive: Poor stability for large values with small variance (catastrophic cancellation)
- **Performance**:
  - two_pass: O(n) time, O(1) space
  - welford: O(n) time, O(1) space, slightly slower than two_pass
  - naive: O(n) time, O(1) space, fastest but numerically unstable
- **Edge Cases**:
  - For small datasets (<1000), differences negligible
  - For large values with small variance, naive can return negative variance!

#### Return Values
- **Type**: `float | Series | DataFrame`
- **Single Column**: Returns scalar float
- **Multiple Columns**: Returns Series with column names as index
- **2D Calculation**: Returns Series or DataFrame depending on axis
- **Units**: Square of original data units (e.g., dollars² for dollar data)

#### Statistical Properties
- **Expectation**: E[s²] = σ² (sample variance is unbiased estimator)
- **Bias**: Sample variance is unbiased; population variance is biased (underestimates by factor of (n-1)/n)
- **Variance of Variance**:
  - For normal distribution: Var(s²) = 2σ⁴/(n-1)
  - For general distribution: Var(s²) = (μ₄ - (n-3)σ⁴/(n-1))/n where μ₄ is fourth central moment
- **Distribution**:
  - (n-1)s²/σ² ~ χ²(n-1) for normal data (chi-square with n-1 degrees of freedom)
  - Approximately normal for large n by CLT
- **Efficiency**: Maximum likelihood estimator for normal distributions
- **Robustness**:
  - NOT robust to outliers (squared deviations amplify extreme values)
  - Single outlier can dominate variance calculation
  - Consider MAD or IQR for robust dispersion estimation

#### Statistical Assumptions
- **Distribution**: No assumptions required for calculation
- **For Inference**:
  - Normality: Chi-square distribution of (n-1)s²/σ² requires normality
  - Independence: Assumes independent observations
  - Identically Distributed: Assumes same distribution for all observations
- **For Unbiased Estimation**: Random sampling from population

#### Interpretation Guidelines
- **Scale**: Variance has squared units, making direct interpretation difficult
- **Comparison**:
  - Variance = 0: All values identical (no variability)
  - Small variance: Data clustered tightly around mean
  - Large variance: Data widely dispersed from mean
- **Relative Interpretation**: Use coefficient of variation (CV = SD/mean) for comparing variability across different scales
- **Outlier Sensitivity**: A few extreme values can greatly inflate variance
- **Negative Variance**: Impossible mathematically; if computed, indicates numerical error

#### Common Use Cases
1. **Portfolio Risk**: Measuring investment volatility in finance (variance of returns)
2. **Quality Control**: Assessing manufacturing process consistency (variance in measurements)
3. **ANOVA**: Partitioning variance to test group differences
4. **Regression Analysis**: Measuring unexplained variance (residual variance)
5. **Hypothesis Testing**: Variance tests (F-test, Levene's test, Bartlett's test)
6. **Signal Processing**: Quantifying signal variability and noise
7. **Experimental Design**: Sample size calculations and power analysis

#### Related Functions
- [STANDARD_DEVIATION](#22-standard-deviation) - Square root of variance, same units as data
- [MEAN](#11-mean) - Center around which variance is calculated
- [MAD](#23-median-absolute-deviation) - Robust alternative to variance
- [IQR](#24-interquartile-range) - Robust measure of spread
- [COEFFICIENT_OF_VARIATION](#25-coefficient-of-variation) - Normalized variance measure
- [COVARIANCE](#6-correlation-analysis) - Variance between two variables

---

## 2.2 STANDARD DEVIATION

#### Purpose
Calculate the standard deviation of a dataset, measuring the average distance of observations from the mean and providing a measure of dispersion in the same units as the original data.

#### Mathematical Specification
Standard deviation is the square root of variance:

**Population Standard Deviation:**
```
σ = √(σ²) = √[(1/N) Σᵢ₌₁ᴺ (xᵢ - μ)²]
```

**Sample Standard Deviation:**
```
s = √(s²) = √[(1/(n-1)) Σᵢ₌₁ⁿ (xᵢ - x̄)²]
```

**Weighted Standard Deviation:**
```
σw = √[Σᵢ₌₁ⁿ wᵢ(xᵢ - μw)² / Σᵢ₌₁ⁿ wᵢ]
```

**Relationship to Variance:**
```
SD = √Variance
Variance = SD²
```

#### Syntax Variations
```
std data column="price" as price_std
stdev data columns=["price","quantity"] type="sample" as result
std data column="salary" ddof=1 as sample_std
stdev data column="score" weighted_by="frequency" as weighted_std
std data columns=all numeric=true skipna=true as std_devs
standard_deviation data column="values" type="population" as pop_std
```

#### Parameters

##### column / columns
- **Type**: `string | list[string] | int | list[int] | pattern | "all"`
- **Required**: Yes
- **Valid Values**:
  - Single column: `"price"`, `0`
  - Multiple columns: `["price", "quantity"]`, `[0, 1, 2]`
  - Pattern: `pattern("num_*")`
  - All numeric: `"all"`
- **Behavior**: Specifies which column(s) to calculate standard deviation for
- **Edge Cases**: Same as variance function

##### type
- **Type**: `string`
- **Required**: No
- **Default**: `"sample"`
- **Valid Values**:
  - `"sample"`: Sample standard deviation (ddof=1)
  - `"population"`: Population standard deviation (ddof=0)
  - `"weighted"`: Weighted standard deviation
- **Behavior**: Determines the type of standard deviation calculation
- **Mathematical Note**: Sample SD is biased but consistent estimator of population SD

##### ddof
- **Type**: `int`
- **Required**: No
- **Default**: `1`
- **Valid Values**: Non-negative integers
- **Behavior**: Delta Degrees of Freedom - applied to variance before taking square root
- **Mathematical Specification**: SD = √[Σ(xᵢ - x̄)² / (n - ddof)]

##### skipna
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to exclude NaN values from calculation

##### weighted_by
- **Type**: `string | list[number]`
- **Required**: No (required if type="weighted")
- **Default**: `null`
- **Valid Values**: Column name or list of weights

##### axis
- **Type**: `int | string`
- **Required**: No
- **Default**: `0`
- **Valid Values**: `0`, `1`, `"index"`, `"columns"`

##### numeric_only
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Valid Values**: `true`, `false`

##### method
- **Type**: `string`
- **Required**: No
- **Default**: `"two_pass"`
- **Valid Values**: `"two_pass"`, `"welford"`, `"naive"`
- **Behavior**: Algorithm for underlying variance calculation

#### Return Values
- **Type**: `float | Series | DataFrame`
- **Units**: Same units as original data (unlike variance)
- **Range**: Always non-negative (SD ≥ 0)

#### Statistical Properties
- **Expectation**: E[s] ≠ σ (sample SD is biased estimator, but bias is small)
- **Bias**: Sample SD slightly underestimates population SD
  - Bias factor: E[s]/σ ≈ 1 - 1/(4n) for normal distribution
- **Variance**: Var(s) ≈ σ²/(2n) for large n and normal distribution
- **Distribution**:
  - For normal data: s follows scaled chi distribution
  - Approximately normal for large n
- **Efficiency**: Highly efficient for normal distributions
- **Robustness**: NOT robust to outliers (inherits sensitivity from variance)

#### Statistical Assumptions
- **For Calculation**: No distributional assumptions required
- **For Inference**:
  - Normality for exact distributional properties
  - Independence of observations
  - Identically distributed observations
- **For Confidence Intervals**: Assumes approximate normality (or large sample)

#### Interpretation Guidelines
- **Empirical Rule (68-95-99.7 Rule)** for approximately normal data:
  - ≈68% of data falls within μ ± 1σ
  - ≈95% of data falls within μ ± 2σ
  - ≈99.7% of data falls within μ ± 3σ
- **Comparative Interpretation**:
  - SD = 0: No variability (all values identical)
  - Small SD: Data clustered near mean
  - Large SD: Data widely spread
- **Coefficient of Variation**: CV = (SD/mean) × 100% for relative comparison
- **Outlier Detection**: Values beyond μ ± 3σ often considered outliers (for normal data)
- **Units**: Unlike variance, SD has same units as data (more interpretable)

#### Common Use Cases
1. **Risk Assessment**: Portfolio volatility, VaR calculations in finance
2. **Quality Control**: Process capability analysis (Six Sigma uses ±6σ limits)
3. **Grading/Scoring**: Standardizing test scores (z-scores)
4. **Confidence Intervals**: Mean ± (critical value × SD/√n)
5. **Control Charts**: Process monitoring (mean ± 3SD control limits)
6. **Sampling**: Determining required sample size for desired precision
7. **Data Normalization**: Standardization (subtract mean, divide by SD)
8. **Uncertainty Quantification**: Reporting measurement precision

#### Related Functions
- [VARIANCE](#21-variance) - Squared standard deviation
- [MAD](#23-median-absolute-deviation) - Robust alternative to SD
- [IQR](#24-interquartile-range) - Robust spread measure
- [MEAN](#11-mean) - Center of distribution
- [COEFFICIENT_OF_VARIATION](#25-coefficient-of-variation) - Normalized SD
- [Z_SCORE](#151-z-score-method) - Standardized deviations
- [STANDARD_ERROR](#4-summary-statistics) - SD of sampling distribution

---

## 2.3 MEDIAN ABSOLUTE DEVIATION

#### Purpose
Calculate the Median Absolute Deviation (MAD), a robust measure of statistical dispersion that quantifies variability using median-based deviations rather than mean-based deviations, making it resistant to outliers.

#### Mathematical Specification
For a univariate dataset X = {x₁, x₂, ..., xₙ}:

**Basic MAD:**
```
MAD = median(|xᵢ - median(X)|)
```

**Scaled MAD (consistent estimator of SD for normal distribution):**
```
MAD* = c × median(|xᵢ - median(X)|)
where c = 1.4826 (consistency constant for normal distribution)
```

**Modified MAD (using mean absolute deviation from median):**
```
MADmean = mean(|xᵢ - median(X)|)
```

**Consistency Constant Derivation:**
```
For normal distribution N(μ, σ²):
c = 1 / Φ⁻¹(0.75) ≈ 1.4826
where Φ⁻¹ is the inverse CDF of standard normal distribution
```

#### Syntax Variations
```
mad data column="price" as price_mad
mad data columns=["price","quantity"] scale=1.4826 as result
mad data column="salary" center="median" scaled=true as robust_spread
mad data column="score" center="mean" as mean_mad
mad data columns=all numeric=true skipna=true as mad_values
median_absolute_deviation data column="values" constant=1.4826 as scaled_mad
```

#### Parameters

##### column / columns
- **Type**: `string | list[string] | int | list[int] | pattern | "all"`
- **Required**: Yes
- **Valid Values**:
  - Single column: `"price"`, `0`
  - Multiple columns: `["price", "quantity"]`
  - Pattern: `pattern("num_*")`
  - All numeric: `"all"`
- **Behavior**: Specifies which column(s) to calculate MAD for
- **Edge Cases**:
  - Non-numeric columns raise TypeError
  - Empty selection raises ValueError
  - Single value returns 0.0
  - Two values returns half the absolute difference

##### center
- **Type**: `string | float`
- **Required**: No
- **Default**: `"median"`
- **Valid Values**:
  - `"median"`: Use median as center (robust, standard MAD)
  - `"mean"`: Use mean as center (less robust)
  - Numeric value: Use specified value as center
- **Behavior**: Determines the center point for absolute deviations
- **Statistical Note**:
  - median: Provides robust MAD with 50% breakdown point
  - mean: Reduces to mean absolute deviation (not robust)
  - custom: Useful for hypothesis testing against specific value
- **Edge Cases**:
  - For symmetric distributions, median ≈ mean (results similar)
  - For skewed distributions, choice significantly impacts result

##### scale / constant
- **Type**: `float | bool`
- **Required**: No
- **Default**: `1.4826` (or `false` for unscaled)
- **Valid Values**:
  - `false` or `1.0`: Unscaled MAD
  - `1.4826`: Scale for consistency with SD under normality
  - `1.253314`: Alternative for unbiased estimation
  - Custom float: User-specified scaling
- **Behavior**: Scaling constant to make MAD comparable to standard deviation
- **Mathematical Basis**:
  - 1.4826 ≈ 1/Φ⁻¹(3/4) makes E[MAD*] = σ for normal data
  - Without scaling, E[MAD] ≈ 0.6745σ for normal data
- **Use Cases**:
  - scaled=true: When comparing to SD or using for outlier detection
  - scaled=false: When interested in absolute median deviation
- **Edge Cases**:
  - Scale = 0 would give MAD = 0 (invalid, raises error)
  - Negative scale raises error

##### skipna
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to exclude NaN values from calculation
- **Edge Cases**:
  - If false and NaN present, result is NaN
  - If true and all values NaN, result is NaN
  - Median computation handles NaN according to this parameter

##### axis
- **Type**: `int | string`
- **Required**: No
- **Default**: `0`
- **Valid Values**: `0` (rows), `1` (columns)
- **Behavior**: Direction along which to calculate MAD

##### numeric_only
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Valid Values**: `true`, `false`

##### method
- **Type**: `string`
- **Required**: No
- **Default**: `"quickselect"`
- **Valid Values**:
  - `"quickselect"`: Fast O(n) median algorithm
  - `"sort"`: O(n log n) full sort
  - `"heap"`: O(n log n) partial sort
- **Behavior**: Algorithm for computing medians (both center and MAD)
- **Performance**: quickselect is fastest for large datasets

##### return_components
- **Type**: `bool`
- **Required**: No
- **Default**: `false`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to return (MAD, center, absolute_deviations) tuple
- **Use Case**: Useful for further analysis of deviation distribution
- **Edge Cases**: If true, returns dict with keys: 'mad', 'center', 'deviations'

#### Return Values
- **Type**: `float | Series | DataFrame | dict`
- **Single Column**: Returns scalar float (or dict if return_components=true)
- **Multiple Columns**: Returns Series with column names as index
- **Units**: Same as original data (when unscaled) or comparable to SD (when scaled)
- **Range**: Always non-negative (MAD ≥ 0)

#### Statistical Properties
- **Expectation**: For normal N(μ, σ²), E[MAD*] = σ when scale=1.4826
- **Efficiency**:
  - Relative efficiency vs SD: ≈ 37% for normal distribution
  - Efficiency increases for heavy-tailed distributions (can exceed 100%)
- **Breakdown Point**: 50% (can tolerate up to 50% outliers before arbitrary results)
- **Asymptotic Distribution**:
  - √n(MAD - σ/c) → N(0, τ²) where τ depends on underlying distribution
  - For normal: τ² ≈ 1.4 × σ²
- **Bias**: Approximately unbiased for large samples from normal distribution
- **Robustness**:
  - Highly robust to outliers (breakdown point = 50%)
  - Does not depend on extreme observations
  - Resistant to heavy-tailed distributions
  - Finite even when variance is infinite (e.g., Cauchy distribution)

#### Statistical Assumptions
- **Distribution**: No assumptions required (distribution-free)
- **Scale**: Requires at least ordinal measurement level
- **Independence**: Assumes independent observations (for inference)
- **For Normal Consistency**: Scaling constant 1.4826 assumes normality for interpretation
- **Finite Median**: Requires population median to exist (always true for finite samples)

#### Interpretation Guidelines
- **Robustness Comparison**:
  - MAD more robust than SD for outlier-contaminated data
  - MAD and SD similar for clean normal data
  - MAD preferred when distribution unknown or non-normal
- **Outlier Detection**:
  - Modified Z-score: (xᵢ - median) / MAD*
  - Threshold: |modified Z| > 3.5 suggests outlier
  - More robust than standard Z-score (based on mean ± 3SD)
- **Scale Interpretation**:
  - Unscaled MAD: Median distance from median
  - Scaled MAD (×1.4826): Approximately equal to SD for normal data
- **Comparison to IQR**:
  - MAD uses all data (after computing median)
  - IQR uses only quartiles (Q3 - Q1)
  - MAD typically smaller: MAD ≈ 0.74 × IQR for normal data

#### Common Use Cases
1. **Outlier Detection**: Robust alternative to Z-score method in contaminated data
   - Finance: Detecting unusual trading volumes or price movements
   - Quality Control: Identifying defective products resistant to outlier contamination
2. **Heavy-Tailed Distributions**: When data has fat tails (finance, insurance)
   - Returns analysis in volatile markets
   - Loss distributions in insurance
3. **Non-Normal Data**: When normality assumption violated
   - Income/salary distributions (right-skewed)
   - Environmental measurements with contamination
4. **Robust Regression**: Used in robust regression diagnostics and weighting
   - M-estimators use MAD for scale estimation
   - Iteratively reweighted least squares (IRLS)
5. **Healthcare & Clinical Trials**: Handling measurement errors and rare events
   - Identifying anomalous patient measurements
   - Robust analysis in presence of protocol deviations
6. **Robust Standardization**: Creating robust z-scores
   - Formula: (x - median) / MAD*
   - Used in robust PCA and other multivariate methods

#### Related Functions
- [STANDARD_DEVIATION](#22-standard-deviation) - Non-robust alternative
- [VARIANCE](#21-variance) - Squared dispersion measure
- [IQR](#24-interquartile-range) - Another robust spread measure
- [MEDIAN](#12-median) - Robust central tendency measure
- [MODIFIED_Z_SCORE](#152-modified-z-score) - Outlier detection using MAD
- [ROBUST_SCALE](#robust-statistics) - Family of robust scale estimators
- [WINSORIZED_SD](#160-winsorization) - Another robust dispersion measure

---

## 2.4 INTERQUARTILE RANGE

#### Purpose
Calculate the Interquartile Range (IQR), a robust measure of statistical dispersion representing the range of the middle 50% of the data, calculated as the difference between the third quartile (Q3, 75th percentile) and first quartile (Q1, 25th percentile).

#### Mathematical Specification
For a dataset with observations x₁, x₂, ..., xₙ:

**Basic IQR:**
```
IQR = Q3 - Q1
where Q1 is the 25th percentile and Q3 is the 75th percentile
```

**Quartile Definitions:**
```
Q1 (First Quartile): 25th percentile - value below which 25% of data falls
Q2 (Second Quartile): 50th percentile - median
Q3 (Third Quartile): 75th percentile - value below which 75% of data falls
```

**Outlier Detection Bounds (Tukey's Fences):**
```
Lower Fence = Q1 - 1.5 × IQR
Upper Fence = Q3 + 1.5 × IQR

Extreme Outlier Bounds:
Lower Extreme = Q1 - 3.0 × IQR
Upper Extreme = Q3 + 3.0 × IQR
```

**Relationship to Standard Deviation:**
```
For normal distribution: IQR ≈ 1.349σ
Therefore: σ ≈ IQR / 1.349 ≈ 0.7413 × IQR
```

#### Syntax Variations
```
iqr data column="price" as price_iqr
iqr data columns=["price","quantity"] as result
iqr data column="salary" interpolation="linear" as salary_iqr
iqr data column="score" return_quartiles=true as quartile_info
iqr data columns=all numeric=true skipna=true as iqr_values
interquartile_range data column="values" method="exclusive" as iqr_result
```

#### Parameters

##### column / columns
- **Type**: `string | list[string] | int | list[int] | pattern | "all"`
- **Required**: Yes
- **Valid Values**:
  - Single column: `"price"`, `0`
  - Multiple columns: `["price", "quantity"]`
  - Pattern: `pattern("num_*")`
  - All numeric: `"all"`
- **Behavior**: Specifies which column(s) to calculate IQR for
- **Edge Cases**:
  - Non-numeric columns raise TypeError
  - Empty selection raises ValueError
  - Fewer than 4 values may give unreliable IQR
  - Single value returns 0.0

##### interpolation
- **Type**: `string`
- **Required**: No
- **Default**: `"linear"`
- **Valid Values**:
  - `"linear"`: Linear interpolation between data points
  - `"lower"`: Use lower data point
  - `"higher"`: Use higher data point
  - `"midpoint"`: Average of lower and higher
  - `"nearest"`: Nearest data point
- **Behavior**: Method for calculating quartiles when they fall between data points
- **Mathematical Note**: Different methods can yield slightly different IQR values
- **Common Practice**: Linear interpolation is standard in most software

##### method
- **Type**: `string`
- **Required**: No
- **Default**: `"inclusive"`
- **Valid Values**:
  - `"inclusive"`: Include median in both halves (Tukey's hinges)
  - `"exclusive"`: Exclude median from both halves
  - `"excel"`: Excel-compatible method
  - `"hazen"`: Hazen's method (positions at (k-0.5)/n)
  - `"weibull"`: Weibull's method (positions at k/(n+1))
- **Behavior**: Determines how to partition data for quartile calculation
- **Differences**:
  - inclusive: Best for small samples (n<10)
  - exclusive: More standard for large samples
  - Methods differ mainly when n is small or when quartiles fall on data points
- **Edge Cases**: Results converge for large n regardless of method

##### skipna
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to exclude NaN values from calculation
- **Edge Cases**:
  - If false and NaN present, result is NaN
  - If true and all values NaN, result is NaN
  - Quartile positions calculated after removing NaN

##### axis
- **Type**: `int | string`
- **Required**: No
- **Default**: `0`
- **Valid Values**: `0` (rows), `1` (columns)
- **Behavior**: Direction along which to calculate IQR

##### numeric_only
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Valid Values**: `true`, `false`

##### return_quartiles
- **Type**: `bool`
- **Required**: No
- **Default**: `false`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to return Q1, Q2, Q3 along with IQR
- **Return Format**: If true, returns dict with keys: 'iqr', 'q1', 'q2', 'q3'
- **Use Case**: Useful for box plot construction and detailed analysis

##### scaled
- **Type**: `bool | float`
- **Required**: No
- **Default**: `false`
- **Valid Values**: `false`, `true`, or custom scaling factor
- **Behavior**: Whether to scale IQR to be comparable to standard deviation
- **Scaling Factor**: If true, multiplies by 0.7413 to approximate SD
- **Mathematical Basis**: For normal data, SD ≈ 0.7413 × IQR

#### Return Values
- **Type**: `float | Series | DataFrame | dict`
- **Single Column**: Returns scalar float (or dict if return_quartiles=true)
- **Multiple Columns**: Returns Series with column names as index
- **Units**: Same as original data
- **Range**: Always non-negative (IQR ≥ 0)

#### Statistical Properties
- **Expectation**: For normal N(μ, σ²), E[IQR] ≈ 1.349σ
- **Efficiency**:
  - Relative efficiency vs SD: ≈ 37% for normal distribution
  - Efficiency increases for heavy-tailed distributions
- **Breakdown Point**: 25% (can tolerate up to 25% outliers in either tail)
- **Variance**: Var(IQR) ≈ 0.467σ²/n for large n from normal distribution
- **Distribution**: Asymptotically normal for large samples
- **Robustness**:
  - Moderately robust (less than MAD, more than SD)
  - Not affected by extreme values outside the middle 50%
  - Resistant to up to 25% contamination in each tail

#### Statistical Assumptions
- **Distribution**: No assumptions required (distribution-free)
- **Scale**: Requires at least ordinal measurement level
- **Independence**: Assumes independent observations (for inference)
- **Sample Size**: More reliable with n ≥ 20
- **For Normal Approximation**: Scaling to SD assumes approximate normality

#### Interpretation Guidelines
- **Spread of Central Data**: IQR represents spread of middle 50% of observations
- **Box Plot Interpretation**:
  - Box extends from Q1 to Q3 (height = IQR)
  - Line inside box represents median (Q2)
  - Whiskers extend to most extreme points within 1.5×IQR from box
- **Outlier Detection (Tukey's Rule)**:
  - Mild outliers: Beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR
  - Extreme outliers: Beyond Q1 - 3.0×IQR or Q3 + 3.0×IQR
- **Why 1.5 Multiplier?**:
  - For normal distribution, Q1 - 1.5×IQR ≈ μ - 2.7σ
  - This approximates the traditional 3σ rule (99.7% of data)
  - Provides good balance between sensitivity and specificity
- **Comparison to Range**:
  - IQR focuses on central 50%, ignoring extremes
  - Range uses only two values (min and max)
  - IQR more stable across samples

#### Common Use Cases
1. **Outlier Detection**: Primary method for identifying unusual values
   - Quality Control: Detecting defective products or unusual measurements
   - Finance: Identifying unusual transactions or trading patterns
   - Healthcare: Flagging abnormal lab results or vital signs
2. **Robust Spread Estimation**: When data contains outliers or is skewed
   - Income/Salary Analysis: Resistant to ultra-high earners
   - Real Estate: Housing prices with extreme luxury properties
   - Survey Data: Responses with extreme values
3. **Box Plot Construction**: Standard measure for box-and-whisker plots
   - Data Visualization: Comparing distributions across groups
   - Exploratory Data Analysis: Quick visual assessment of spread
4. **Non-Normal Distributions**: Preferred over SD for skewed data
   - Financial Returns: Heavy-tailed distributions
   - Environmental Data: Concentration measurements with outliers
5. **Data Quality Assessment**: Checking for data entry errors
   - Detecting impossible values outside reasonable bounds
   - Validating measurement ranges
6. **Comparing Variability**: Across groups with different scales
   - Clinical Trials: Comparing treatment effects across sites
   - Manufacturing: Comparing process variation across shifts

#### Related Functions
- [QUARTILES](#45-quartiles) - Individual quartile values (Q1, Q2, Q3)
- [PERCENTILE](#55-quantile-function) - General percentile calculation
- [MAD](#23-median-absolute-deviation) - More robust alternative
- [RANGE](#25-range) - Full data range (max - min)
- [STANDARD_DEVIATION](#22-standard-deviation) - Non-robust dispersion measure
- [OUTLIERS](#151-z-score-method) - Outlier detection methods
- [BOXPLOT](#visualization) - Visualization using IQR

---

## 2.5 RANGE

#### Purpose
Calculate the range of a dataset, representing the simplest measure of statistical dispersion as the difference between the maximum and minimum values, providing the total spread of the data.

#### Mathematical Specification
For a dataset with observations x₁, x₂, ..., xₙ:

**Range:**
```
Range = max(X) - min(X) = x(n) - x(1)
where x(1) ≤ x(2) ≤ ... ≤ x(n) are order statistics
```

**Midrange (center point):**
```
Midrange = (max(X) + min(X)) / 2
```

**Relative Range (coefficient of range):**
```
Coefficient of Range = (max - min) / (max + min)
```

**Expected Range for Normal Distribution:**
```
E[Range] ≈ σ × d₂(n)
where d₂(n) is a tabulated constant depending on sample size n
```

#### Syntax Variations
```
range data column="price" as price_range
range data columns=["price","quantity"] as result
range data column="salary" return_bounds=true as salary_bounds
range data column="score" scaled=true as normalized_range
range data columns=all numeric=true skipna=true as ranges
data_range data column="values" return_midrange=true as range_info
```

#### Parameters

##### column / columns
- **Type**: `string | list[string] | int | list[int] | pattern | "all"`
- **Required**: Yes
- **Valid Values**:
  - Single column: `"price"`, `0`
  - Multiple columns: `["price", "quantity"]`
  - Pattern: `pattern("num_*")`
  - All numeric: `"all"`
- **Behavior**: Specifies which column(s) to calculate range for
- **Edge Cases**:
  - Non-numeric columns raise TypeError
  - Empty selection raises ValueError
  - Single value returns 0.0
  - Two identical values return 0.0

##### skipna
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to exclude NaN values when finding min/max
- **Edge Cases**:
  - If false and NaN present, result is NaN
  - If true and all values NaN, result is NaN

##### axis
- **Type**: `int | string`
- **Required**: No
- **Default**: `0`
- **Valid Values**: `0` (rows), `1` (columns)

##### numeric_only
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Valid Values**: `true`, `false`

##### return_bounds
- **Type**: `bool`
- **Required**: No
- **Default**: `false`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to return min and max values along with range
- **Return Format**: If true, returns dict with keys: 'range', 'min', 'max'
- **Use Case**: Useful for understanding data bounds and validation

##### return_midrange
- **Type**: `bool`
- **Required**: No
- **Default**: `false`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to return midrange (center point between min and max)
- **Mathematical Note**: Midrange = (min + max) / 2
- **Use Case**: Quick estimate of center for symmetric distributions

##### scaled
- **Type**: `bool | string`
- **Required**: No
- **Default**: `false`
- **Valid Values**:
  - `false`: Return absolute range
  - `"coefficient"`: Return coefficient of range (relative range)
  - `"normalized"`: Return range / (max + min) if max + min ≠ 0
- **Behavior**: Type of scaling to apply to range
- **Edge Cases**:
  - If max + min = 0, scaled versions may be undefined

##### outlier_resistant
- **Type**: `bool | float`
- **Required**: No
- **Default**: `false`
- **Valid Values**:
  - `false`: Use true min/max
  - `true`: Use 5th and 95th percentiles
  - Float (0-0.5): Use specified percentile trim (e.g., 0.05 for 5th-95th)
- **Behavior**: Whether to calculate range using percentiles instead of absolute min/max
- **Purpose**: Reduce sensitivity to extreme outliers
- **Edge Cases**: Percentile must be between 0 and 0.5

#### Return Values
- **Type**: `float | Series | DataFrame | dict`
- **Single Column**: Returns scalar float (or dict if return_bounds/midrange=true)
- **Multiple Columns**: Returns Series with column names as index
- **Units**: Same as original data
- **Range**: Always non-negative (Range ≥ 0)

#### Statistical Properties
- **Expectation**: E[Range] increases with sample size
  - For normal N(μ, σ²): E[Range] ≈ σ × d₂(n) where d₂ increases with n
  - Example: d₂(2) ≈ 1.128, d₂(5) ≈ 2.326, d₂(10) ≈ 3.078, d₂(100) ≈ 5.015
- **Variance**: Var(Range) increases with sample size
- **Distribution**: For normal data, range/σ follows studentized range distribution
- **Efficiency**: Very low efficiency compared to standard deviation
  - Efficiency decreases rapidly as sample size increases
  - Only uses 2 out of n data points
- **Robustness**:
  - NOT robust at all (breakdown point = 0%)
  - Single outlier completely determines range
  - Most sensitive measure of dispersion to extreme values

#### Statistical Assumptions
- **Distribution**: No assumptions required for calculation
- **For Expected Range**: Assumes specific distribution (usually normal)
- **Independence**: Assumes independent observations
- **For Quality Control**: Often assumes stable process for range charts

#### Interpretation Guidelines
- **Simplicity**: Most intuitive measure of spread - total data span
- **Data Validation**: Useful for detecting data entry errors
  - Example: Age range of 7-123 in school children indicates error
  - Helps identify impossible or implausible values
- **Sensitivity**: Extremely sensitive to outliers and sample size
  - Increases with sample size even for same population
  - One extreme value dominates the measure
- **Scale Dependence**: Absolute range depends on measurement units
  - Use coefficient of range for scale-free comparison
- **Comparison Guidelines**:
  - Small range: Data tightly clustered
  - Large range: Wide spread in values
  - Range alone doesn't indicate distribution shape

#### Common Use Cases
1. **Data Exploration and Validation**: Quick check of data extent
   - Identifying data entry errors (implausible min/max values)
   - Verifying measurement bounds (temperature, age, etc.)
   - Initial data quality assessment
2. **Quality Control (Small Samples)**: Process monitoring with n ≤ 10
   - R-charts in statistical process control
   - Monitoring variation in small subgroups
   - Pharmaceutical tablet weight variation (small batches)
3. **Threshold Monitoring**: Critical value detection
   - Medical monitoring: Did any measurement exceed safe limits?
   - Financial controls: Were spending limits breached?
   - Environmental monitoring: Did pollution exceed thresholds?
4. **Simple Reporting**: When sophisticated measures not needed
   - Weather reports: "Temperatures ranged from 60° to 75°"
   - Sports statistics: "Scores ranged from 14 to 42"
   - Price ranges in retail: "$19.99 to $49.99"
5. **Educational Settings**: Teaching basic statistical concepts
   - Introduction to variability for students
   - Visual understanding of spread
   - Foundation before learning SD and IQR
6. **Preliminary Analysis**: First step before detailed analysis
   - Quick magnitude check before choosing analysis method
   - Identifying need for outlier investigation
   - Determining appropriate scaling for visualization

#### Related Functions
- [MIN](#4-summary-statistics) - Minimum value
- [MAX](#4-summary-statistics) - Maximum value
- [IQR](#24-interquartile-range) - Robust alternative (middle 50%)
- [STANDARD_DEVIATION](#22-standard-deviation) - Uses all data points
- [MAD](#23-median-absolute-deviation) - Robust dispersion measure
- [PERCENTILE](#55-quantile-function) - For outlier-resistant range
- [OUTLIERS](#151-z-score-method) - Detecting extreme values

---

## 2.6 COEFFICIENT OF VARIATION

#### Purpose
Calculate the Coefficient of Variation (CV), a standardized measure of dispersion representing the ratio of the standard deviation to the mean, expressed as a percentage, enabling comparison of variability across datasets with different units or scales.

#### Mathematical Specification
For a dataset with mean μ and standard deviation σ:

**Coefficient of Variation:**
```
CV = (σ / μ) × 100%
```

**Sample Coefficient of Variation:**
```
CV = (s / x̄) × 100%
where s is sample standard deviation and x̄ is sample mean
```

**Unbiased CV (for normal distribution):**
```
CV* = CV × [1 + 1/(4n)]
where n is sample size (bias correction for small samples)
```

**Relative Standard Deviation (RSD):**
```
RSD = (σ / μ) × 100% = CV
(Alternative name for the same measure)
```

**Coefficient of Dispersion (alternative scaling):**
```
CD = σ / μ (without percentage conversion)
```

#### Syntax Variations
```
cv data column="price" as price_cv
coefficient_variation data columns=["price","quantity"] as result
cv data column="salary" percent=true as salary_cv_pct
cv data column="score" unbiased=true as corrected_cv
cv data columns=all numeric=true skipna=true as cv_values
relative_std data column="values" ddof=1 as rsd
```

#### Parameters

##### column / columns
- **Type**: `string | list[string] | int | list[int] | pattern | "all"`
- **Required**: Yes
- **Valid Values**:
  - Single column: `"price"`, `0`
  - Multiple columns: `["price", "quantity"]`
  - Pattern: `pattern("num_*")`
  - All numeric: `"all"`
- **Behavior**: Specifies which column(s) to calculate CV for
- **Edge Cases**:
  - Non-numeric columns raise TypeError
  - Columns with mean = 0 raise division by zero error
  - Columns with mean near zero give unreliable CV (may be extremely large)

##### percent
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to express CV as percentage (×100) or proportion
- **Standard**: CV typically reported as percentage

##### ddof
- **Type**: `int`
- **Required**: No
- **Default**: `1`
- **Valid Values**: `0`, `1`
- **Behavior**: Degrees of freedom for standard deviation calculation
- **Common Values**:
  - ddof=1: Sample CV (standard)
  - ddof=0: Population CV

##### skipna
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to exclude NaN values from calculation

##### axis
- **Type**: `int | string`
- **Required**: No
- **Default**: `0`
- **Valid Values**: `0` (rows), `1` (columns)

##### numeric_only
- **Type**: `bool`
- **Required**: No
- **Default**: `true`
- **Valid Values**: `true`, `false`

##### unbiased
- **Type**: `bool`
- **Required**: No
- **Default**: `false`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to apply bias correction for small samples
- **Mathematical Note**: Applies correction factor [1 + 1/(4n)]
- **Use Case**: For small samples (n < 30) from normal distribution

##### absolute
- **Type**: `bool`
- **Required**: No
- **Default**: `false`
- **Valid Values**: `true`, `false`
- **Behavior**: Whether to use absolute value of mean
- **Purpose**: Handle negative means (though CV interpretation unclear for negative values)
- **Warning**: CV not well-defined for data crossing zero or with negative mean

##### warn_near_zero
- **Type**: `bool | float`
- **Required**: No
- **Default**: `true`
- **Valid Values**:
  - `true`: Warn if |mean| < 0.01 × SD
  - `false`: No warning
  - Float: Custom threshold for warning
- **Behavior**: Issue warning when mean near zero (unstable CV)

#### Return Values
- **Type**: `float | Series | DataFrame`
- **Single Column**: Returns scalar float
- **Multiple Columns**: Returns Series with column names as index
- **Units**: Percentage (%) if percent=true, otherwise dimensionless
- **Range**: Non-negative for ratio variables (CV ≥ 0%)
- **Interpretation Ranges**:
  - CV < 10%: Low variability (highly consistent)
  - CV 10-20%: Moderate variability (acceptable for most applications)
  - CV 20-30%: High variability (caution in interpretation)
  - CV > 30%: Very high variability (data may be unreliable)

#### Statistical Properties
- **Scale Invariance**: CV is dimensionless and scale-free
  - Unchanged by multiplicative transformations
  - CV(a × X) = CV(X) for a > 0
  - CV(X + b) ≠ CV(X) (not location invariant)
- **Expectation**: E[CV] ≈ CV_true + bias
  - Bias ≈ CV/(2n) for normal distribution
  - Positive bias (overestimates true CV)
- **Variance**: Var(CV) ≈ CV²/(2n) for normal distribution and large n
- **Distribution**: No simple exact distribution
  - Approximately normal for large n and not-too-large CV
  - Can be highly skewed for small n or large CV
- **Robustness**: NOT robust (inherits from mean and SD)
  - Sensitive to outliers through both numerator and denominator
  - Unreliable for data with outliers

#### Statistical Assumptions
- **Ratio Scale**: Requires ratio-level measurement (true zero point)
  - Valid: Height, weight, price, concentration
  - Invalid: Temperature in Celsius/Fahrenheit (arbitrary zero)
- **Positive Values**: Best suited for strictly positive data
  - Undefined for mean = 0
  - Problematic interpretation for data including negative values
- **For Inference**: Assumes underlying distribution (often normal)
- **Homogeneity**: Assumes single homogeneous population

#### Interpretation Guidelines
- **Relative Variability**: CV expresses SD as percentage of mean
  - CV = 20% means SD is 20% of the mean value
  - Enables comparison across different scales/units
- **When to Use CV**:
  - **YES**: Comparing variability of height (cm) vs weight (kg)
  - **YES**: Assessing consistency across different products/batches
  - **YES**: When mean magnitudes differ substantially
  - **NO**: Data with negative values or crossing zero
  - **NO**: Temperature in Celsius (arbitrary zero)
  - **NO**: Mean very close to zero (unstable CV)
- **Comparison Guidelines**:
  - Lower CV: More consistent/precise/homogeneous
  - Higher CV: More variable/dispersed/heterogeneous
  - Compare CVs only for ratio-scale variables
- **Context Matters**: "Acceptable" CV varies by field
  - Laboratory assays: Often require CV < 5-10%
  - Manufacturing: CV < 15% typically acceptable
  - Biological/social sciences: CV 20-30% common
  - Financial markets: CV > 50% not unusual

#### Common Use Cases
1. **Quality Control and Method Validation**: Assessing precision and reproducibility
   - Laboratory Assays: Inter-assay and intra-assay variability (CV < 10% often required)
   - Manufacturing: Batch-to-batch consistency (tablet weights, concentrations)
   - Analytical Chemistry: Method precision assessment
2. **Comparing Variability Across Different Scales**: When variables have different units
   - Clinical Trials: Comparing variability of different biomarkers
   - Economics: Comparing income inequality across countries with different currencies
   - Ecology: Comparing population variability across different species
3. **Investment and Finance**: Risk assessment
   - Sharpe Ratio component: Risk per unit of return
   - Portfolio Risk: Comparing volatility of different assets
   - Price Stability: Assessing commodity price fluctuations
4. **Process Capability**: Manufacturing and service quality
   - Six Sigma: Process consistency assessment
   - Service Industry: Response time variability
   - Production: Output consistency monitoring
5. **Medical and Health Sciences**: Biological variability assessment
   - Biomarker Stability: CV of repeated measurements
   - Clinical Reference Ranges: Defining normal variability
   - Pharmacokinetics: Drug concentration variability
6. **Material Science**: Property consistency
   - Strength Testing: Variability in material properties
   - Quality Assurance: Product specification compliance
   - Experimental Reproducibility: Method reliability

#### Related Functions
- [STANDARD_DEVIATION](#22-standard-deviation) - Numerator of CV
- [MEAN](#11-mean) - Denominator of CV
- [VARIANCE](#21-variance) - Related dispersion measure
- [STANDARDIZED_RANGE](#25-range) - Alternative relative measure
- [RELATIVE_IQR](#24-interquartile-range) - Robust alternative
- [INDEX_OF_DISPERSION](#dispersion) - Variance-to-mean ratio (related concept)

---

