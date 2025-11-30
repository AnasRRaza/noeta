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
