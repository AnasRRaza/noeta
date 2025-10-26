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
    columns: List[str]

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
