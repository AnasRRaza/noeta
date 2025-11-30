"""
AST Node definitions for Noeta DSL
"""
from dataclasses import dataclass
from typing import List, Optional, Any

# Base AST Node
@dataclass
class ASTNode:
    pass

# Program (root) node
@dataclass
class ProgramNode(ASTNode):
    statements: List[ASTNode]

# Data Manipulation Nodes
@dataclass
class LoadNode(ASTNode):
    file_path: str
    alias: str

# Enhanced Load Nodes with parameters
@dataclass
class LoadCSVNode(ASTNode):
    filepath: str
    params: dict  # All optional parameters
    alias: str

@dataclass
class LoadJSONNode(ASTNode):
    filepath: str
    params: dict
    alias: str

@dataclass
class LoadExcelNode(ASTNode):
    filepath: str
    params: dict
    alias: str

@dataclass
class LoadParquetNode(ASTNode):
    filepath: str
    params: dict
    alias: str

@dataclass
class LoadSQLNode(ASTNode):
    query: str
    connection: str
    params: dict
    alias: str

# Enhanced Save Nodes
@dataclass
class SaveCSVNode(ASTNode):
    source_alias: str
    filepath: str
    params: dict

@dataclass
class SaveJSONNode(ASTNode):
    source_alias: str
    filepath: str
    params: dict

@dataclass
class SaveExcelNode(ASTNode):
    source_alias: str
    filepath: str
    params: dict

@dataclass
class SaveParquetNode(ASTNode):
    source_alias: str
    filepath: str
    params: dict

# Phase 2: Selection & Projection Nodes
@dataclass
class SelectByTypeNode(ASTNode):
    source_alias: str
    dtype: str  # 'numeric', 'string', 'datetime', etc.
    new_alias: str

@dataclass
class HeadNode(ASTNode):
    source_alias: str
    n_rows: int
    new_alias: str

@dataclass
class TailNode(ASTNode):
    source_alias: str
    n_rows: int
    new_alias: str

@dataclass
class ILocNode(ASTNode):
    source_alias: str
    row_slice: tuple  # (start, end) or single int
    col_slice: Optional[tuple]  # Optional column selection
    new_alias: str

@dataclass
class LocNode(ASTNode):
    source_alias: str
    row_labels: Any  # Can be list, slice, or single label
    col_labels: Optional[List[str]]  # Optional column selection
    new_alias: str

@dataclass
class RenameColumnsNode(ASTNode):
    source_alias: str
    mapping: dict  # old_name -> new_name mapping
    new_alias: str

@dataclass
class ReorderColumnsNode(ASTNode):
    source_alias: str
    column_order: List[str]
    new_alias: str

@dataclass
class SelectNode(ASTNode):
    source_alias: str
    columns: List[str]
    new_alias: str

@dataclass
class FilterNode(ASTNode):
    source_alias: str
    condition: 'ConditionNode'
    new_alias: str

# Phase 3: Filtering Nodes
@dataclass
class FilterBetweenNode(ASTNode):
    source_alias: str
    column: str
    min_value: Any
    max_value: Any
    new_alias: str

@dataclass
class FilterIsInNode(ASTNode):
    source_alias: str
    column: str
    values: List[Any]
    new_alias: str

@dataclass
class FilterContainsNode(ASTNode):
    source_alias: str
    column: str
    pattern: str
    new_alias: str

@dataclass
class FilterStartsWithNode(ASTNode):
    source_alias: str
    column: str
    pattern: str
    new_alias: str

@dataclass
class FilterEndsWithNode(ASTNode):
    source_alias: str
    column: str
    pattern: str
    new_alias: str

@dataclass
class FilterRegexNode(ASTNode):
    source_alias: str
    column: str
    pattern: str
    new_alias: str

@dataclass
class FilterNullNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

@dataclass
class FilterNotNullNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

@dataclass
class FilterDuplicatesNode(ASTNode):
    source_alias: str
    subset: Optional[List[str]]  # Columns to consider for duplicates
    keep: str  # 'first', 'last', or False
    new_alias: str

@dataclass
class SortNode(ASTNode):
    source_alias: str
    sort_specs: List['SortSpecNode']
    new_alias: str

@dataclass
class JoinNode(ASTNode):
    alias1: str
    alias2: str
    join_column: str
    new_alias: str

@dataclass
class GroupByNode(ASTNode):
    source_alias: str
    group_columns: List[str]
    aggregations: List['AggregationNode']
    new_alias: str

@dataclass
class SampleNode(ASTNode):
    source_alias: str
    sample_size: int
    is_random: bool
    new_alias: str

@dataclass
class DropNANode(ASTNode):
    source_alias: str
    columns: Optional[List[str]]
    new_alias: str

@dataclass
class FillNANode(ASTNode):
    source_alias: str
    fill_value: Any
    columns: Optional[List[str]]
    new_alias: str

@dataclass
class MutateNode(ASTNode):
    source_alias: str
    mutations: List['MutationNode']
    new_alias: str

@dataclass
class ApplyNode(ASTNode):
    source_alias: str
    columns: List[str]
    function_expr: str
    new_alias: str

# Analysis Nodes
@dataclass
class DescribeNode(ASTNode):
    source_alias: str
    columns: Optional[List[str]]

@dataclass
class SummaryNode(ASTNode):
    source_alias: str

@dataclass
class InfoNode(ASTNode):
    source_alias: str

@dataclass
class OutliersNode(ASTNode):
    source_alias: str
    method: str
    columns: List[str]

@dataclass
class QuantileNode(ASTNode):
    source_alias: str
    column: str
    quantile_value: float

@dataclass
class NormalizeNode(ASTNode):
    source_alias: str
    columns: List[str]
    method: str
    new_alias: str

@dataclass
class BinningNode(ASTNode):
    source_alias: str
    column: str
    num_bins: int
    new_alias: str

@dataclass
class RollingNode(ASTNode):
    source_alias: str
    column: str
    window_size: int
    function_name: str
    new_alias: str

@dataclass
class HypothesisNode(ASTNode):
    alias1: str
    alias2: str
    columns: List[str]
    test_type: str

# Visualization Nodes
@dataclass
class BoxPlotNode(ASTNode):
    source_alias: str
    columns: Optional[List[str]] = None      # Classic syntax
    value_column: Optional[str] = None       # Natural syntax
    group_column: Optional[str] = None       # Natural syntax grouping

@dataclass
class HeatmapNode(ASTNode):
    source_alias: str
    columns: List[str]

@dataclass
class PairPlotNode(ASTNode):
    source_alias: str
    columns: List[str]

@dataclass
class TimeSeriesNode(ASTNode):
    source_alias: str
    x_column: str
    y_column: str

@dataclass
class PieChartNode(ASTNode):
    source_alias: str
    values_column: str
    labels_column: str

# File Operation Nodes
@dataclass
class SaveNode(ASTNode):
    source_alias: str
    file_path: str
    format_type: Optional[str]

@dataclass
class ExportPlotNode(ASTNode):
    file_name: str
    width: Optional[int]
    height: Optional[int]

# Helper Nodes
@dataclass
class ConditionNode(ASTNode):
    left_operand: str
    operator: str
    right_operand: Any  # Can be identifier or literal

@dataclass
class SortSpecNode(ASTNode):
    column_name: str
    direction: str  # 'ASC' or 'DESC'

@dataclass
class AggregationNode(ASTNode):
    function_name: str
    column_name: str

@dataclass
class MutationNode(ASTNode):
    new_column: str
    expression: str

# ============================================================
# PHASE 4: TRANSFORMATION OPERATIONS
# ============================================================

# Phase 4A: Math Operations
@dataclass
class RoundNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str
    decimals: int = 0

@dataclass
class AbsNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

@dataclass
class SqrtNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

@dataclass
class PowerNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str
    exponent: float = 2.0

@dataclass
class LogNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str
    base: str = "e"  # "e", "10", or number

@dataclass
class CeilNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

@dataclass
class FloorNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

# Phase 4B: String Operations
@dataclass
class UpperNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

@dataclass
class LowerNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

@dataclass
class StripNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

@dataclass
class ReplaceNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str
    old: str = ""
    new: str = ""

@dataclass
class SplitNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str
    delimiter: str = " "

@dataclass
class ConcatNode(ASTNode):
    source_alias: str
    columns: List[str]
    new_alias: str
    separator: str = ""

@dataclass
class SubstringNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str
    start: int = 0
    end: Optional[int] = None

@dataclass
class LengthNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

# Phase 4C: Date/Time Operations
@dataclass
class ParseDatetimeNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str
    format: Optional[str] = None

@dataclass
class ExtractYearNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

@dataclass
class ExtractMonthNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

@dataclass
class ExtractDayNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

@dataclass
class DateDiffNode(ASTNode):
    source_alias: str
    start_column: str
    end_column: str
    new_alias: str
    unit: str = "days"

# Phase 4D: Type Operations
@dataclass
class AsTypeNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str
    dtype: str = "str"

@dataclass
class ToNumericNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str
    errors: str = "raise"  # "raise", "coerce", "ignore"

# Phase 4E: Encoding Operations
@dataclass
class OneHotEncodeNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

@dataclass
class LabelEncodeNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

# Phase 4F: Scaling Operations
@dataclass
class StandardScaleNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

@dataclass
class MinMaxScaleNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

# ============================================================
# PHASE 5: CLEANING OPERATIONS
# ============================================================

# Phase 5A: Missing Data Detection
@dataclass
class IsNullNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

@dataclass
class NotNullNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

@dataclass
class CountNANode(ASTNode):
    source_alias: str

# Phase 5B: Missing Data Imputation
@dataclass
class FillForwardNode(ASTNode):
    source_alias: str
    new_alias: str
    column: Optional[str] = None  # If None, fills all columns

@dataclass
class FillBackwardNode(ASTNode):
    source_alias: str
    new_alias: str
    column: Optional[str] = None  # If None, fills all columns

@dataclass
class FillMeanNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

@dataclass
class FillMedianNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str

@dataclass
class InterpolateNode(ASTNode):
    source_alias: str
    new_alias: str
    column: Optional[str] = None  # If None, interpolates all columns
    method: str = "linear"

# Phase 5C: Duplicate Detection
@dataclass
class DuplicatedNode(ASTNode):
    source_alias: str
    new_alias: str
    columns: Optional[List[str]] = None
    keep: str = "first"  # "first", "last", False

@dataclass
class CountDuplicatesNode(ASTNode):
    source_alias: str
    columns: Optional[List[str]] = None

# ============================================================
# PHASE 6: DATA ORDERING OPERATIONS
# ============================================================

@dataclass
class SortIndexNode(ASTNode):
    source_alias: str
    new_alias: str
    ascending: bool = True

@dataclass
class RankNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str
    method: str = "average"  # "average", "min", "max", "first", "dense"
    ascending: bool = True
    pct: bool = False  # Return percentile ranks

# ============================================================
# PHASE 7: AGGREGATION & GROUPING OPERATIONS
# ============================================================

@dataclass
class FilterGroupsNode(ASTNode):
    source_alias: str
    group_columns: List[str]
    condition: str  # e.g., "count > 5" or "sum > 1000"
    new_alias: str

@dataclass
class GroupTransformNode(ASTNode):
    source_alias: str
    group_columns: List[str]
    column: str
    function: str  # "mean", "sum", "std", etc.
    new_alias: str

@dataclass
class WindowRankNode(ASTNode):
    source_alias: str
    column: str
    partition_by: Optional[List[str]]
    new_alias: str
    method: str = "rank"  # "rank", "dense_rank", "row_number"
    ascending: bool = True

@dataclass
class WindowLagNode(ASTNode):
    source_alias: str
    column: str
    periods: int
    new_alias: str
    partition_by: Optional[List[str]] = None
    fill_value: Any = None

@dataclass
class WindowLeadNode(ASTNode):
    source_alias: str
    column: str
    periods: int
    new_alias: str
    partition_by: Optional[List[str]] = None
    fill_value: Any = None

@dataclass
class RollingMeanNode(ASTNode):
    source_alias: str
    column: str
    window: int
    new_alias: str
    min_periods: int = 1

@dataclass
class RollingSumNode(ASTNode):
    source_alias: str
    column: str
    window: int
    new_alias: str
    min_periods: int = 1

@dataclass
class RollingStdNode(ASTNode):
    source_alias: str
    column: str
    window: int
    new_alias: str
    min_periods: int = 1

@dataclass
class RollingMinNode(ASTNode):
    source_alias: str
    column: str
    window: int
    new_alias: str
    min_periods: int = 1

@dataclass
class RollingMaxNode(ASTNode):
    source_alias: str
    column: str
    window: int
    new_alias: str
    min_periods: int = 1

@dataclass
class ExpandingMeanNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str
    min_periods: int = 1

@dataclass
class ExpandingSumNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str
    min_periods: int = 1

@dataclass
class ExpandingMinNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str
    min_periods: int = 1

@dataclass
class ExpandingMaxNode(ASTNode):
    source_alias: str
    column: str
    new_alias: str
    min_periods: int = 1

# ============================================================
# PHASE 8: DATA RESHAPING OPERATIONS
# ============================================================

@dataclass
class PivotNode(ASTNode):
    source_alias: str
    index: str  # Column to use as index
    columns: str  # Column to use as new column headers
    values: str  # Column for values
    new_alias: str

@dataclass
class PivotTableNode(ASTNode):
    source_alias: str
    index: str
    columns: str
    values: str
    new_alias: str
    aggfunc: str = "mean"  # Aggregation function
    fill_value: Any = None

@dataclass
class MeltNode(ASTNode):
    source_alias: str
    id_vars: List[str]  # Columns to keep as identifiers
    value_vars: Optional[List[str]]  # Columns to unpivot (None = all others)
    new_alias: str
    var_name: str = "variable"
    value_name: str = "value"

@dataclass
class StackNode(ASTNode):
    source_alias: str
    new_alias: str
    level: int = -1  # Level to stack

@dataclass
class UnstackNode(ASTNode):
    source_alias: str
    new_alias: str
    level: int = -1  # Level to unstack
    fill_value: Any = None

@dataclass
class TransposeNode(ASTNode):
    source_alias: str
    new_alias: str

@dataclass
class CrosstabNode(ASTNode):
    source_alias: str
    row_column: str  # Column for rows
    col_column: str  # Column for columns
    new_alias: str
    aggfunc: str = "count"  # Aggregation function
    values: Optional[str] = None  # Values column for aggregation

# ============================================================
# PHASE 9: DATA COMBINING OPERATIONS
# ============================================================

@dataclass
class MergeNode(ASTNode):
    left_alias: str
    right_alias: str
    new_alias: str
    on: Optional[str] = None  # Common column
    left_on: Optional[str] = None
    right_on: Optional[str] = None
    how: str = "inner"  # "inner", "left", "right", "outer", "cross"
    suffixes: tuple = ("_x", "_y")

@dataclass
class ConcatVerticalNode(ASTNode):
    sources: List[str]  # List of source aliases
    new_alias: str
    ignore_index: bool = True

@dataclass
class ConcatHorizontalNode(ASTNode):
    sources: List[str]
    new_alias: str
    ignore_index: bool = False

@dataclass
class UnionNode(ASTNode):
    left_alias: str
    right_alias: str
    new_alias: str

@dataclass
class IntersectionNode(ASTNode):
    left_alias: str
    right_alias: str
    new_alias: str

@dataclass
class DifferenceNode(ASTNode):
    left_alias: str
    right_alias: str
    new_alias: str

# ============================================================
# PHASE 10: ADVANCED OPERATIONS
# ============================================================

@dataclass
class SetIndexNode(ASTNode):
    source_alias: str
    column: str  # Column(s) to set as index
    new_alias: str
    drop: bool = True  # Whether to drop the column from data

@dataclass
class ResetIndexNode(ASTNode):
    source_alias: str
    new_alias: str
    drop: bool = False  # Whether to discard the index

@dataclass
class ApplyRowNode(ASTNode):
    source_alias: str
    function_expr: str  # Expression to apply to each row
    new_alias: str

@dataclass
class ApplyColumnNode(ASTNode):
    source_alias: str
    column: str
    function_expr: str  # Expression to apply to column
    new_alias: str

@dataclass
class ResampleNode(ASTNode):
    source_alias: str
    rule: str  # Resampling rule: "D", "W", "M", "Y", etc.
    column: str  # Column to aggregate
    aggfunc: str  # Aggregation function
    new_alias: str

@dataclass
class AssignNode(ASTNode):
    source_alias: str
    column: str
    value: Any  # Constant value or expression
    new_alias: str
