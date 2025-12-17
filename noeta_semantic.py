"""
Semantic Analyzer for Noeta DSL

This module provides semantic validation for Noeta programs by:
- Tracking defined datasets in a symbol table
- Validating dataset references before code generation
- Checking column existence (when schema is known)
- Type checking for operations (when types are known)
- Providing helpful error messages for undefined references
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional
from noeta_ast import *
from noeta_errors import (
    NoetaError, ErrorCategory, ErrorContext,
    create_semantic_error, suggest_similar
)


class DataType(Enum):
    """Column data types for type checking."""
    STRING = "string"
    NUMERIC = "numeric"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


@dataclass
class ColumnInfo:
    """Information about a column in a dataset."""
    name: str
    dtype: DataType = DataType.UNKNOWN
    nullable: bool = True


@dataclass
class DatasetInfo:
    """
    Information about a dataset.

    Tracks the dataset name, its columns (if known), and the operation
    that created it.
    """
    name: str
    columns: Dict[str, ColumnInfo] = field(default_factory=dict)
    source: Optional[str] = None  # file path or operation that created it

    def has_column(self, col_name: str) -> bool:
        """Check if dataset has a specific column."""
        return col_name in self.columns

    def get_column_type(self, col_name: str) -> DataType:
        """Get the type of a column (returns UNKNOWN if not found)."""
        if col_name in self.columns:
            return self.columns[col_name].dtype
        return DataType.UNKNOWN

    def add_column(self, col_name: str, dtype: DataType = DataType.UNKNOWN, nullable: bool = True):
        """Add a column to the dataset schema."""
        self.columns[col_name] = ColumnInfo(col_name, dtype, nullable)


class SymbolTable:
    """
    Track all defined datasets and their schemas.

    This is the core of semantic validation - keeps track of what datasets
    exist at any point in the program execution.
    """

    def __init__(self):
        self.datasets: Dict[str, DatasetInfo] = {}
        self.history: List[str] = []  # Track order of definitions

    def define(self, name: str, info: DatasetInfo):
        """
        Register a new dataset.

        Args:
            name: Dataset alias
            info: DatasetInfo object with schema information
        """
        self.datasets[name] = info
        self.history.append(name)

    def lookup(self, name: str) -> Optional[DatasetInfo]:
        """
        Look up a dataset by name.

        Args:
            name: Dataset alias

        Returns:
            DatasetInfo if found, None otherwise
        """
        return self.datasets.get(name)

    def exists(self, name: str) -> bool:
        """Check if a dataset has been defined."""
        return name in self.datasets

    def get_all_names(self) -> List[str]:
        """Get list of all defined dataset names."""
        return list(self.datasets.keys())

    def clear(self):
        """Clear all dataset definitions (useful for testing)."""
        self.datasets.clear()
        self.history.clear()


class SemanticAnalyzer:
    """
    Validates AST for semantic correctness.

    Performs compile-time validation to catch errors like:
    - Undefined dataset references
    - Undefined column references (when schema is known)
    - Type mismatches (when types are known)
    - Invalid operation combinations
    """

    def __init__(self, source_code: str = ""):
        self.source_code = source_code
        self.source_lines = source_code.split('\n') if source_code else []
        self.symbol_table = SymbolTable()
        self.errors: List[NoetaError] = []

    def analyze(self, ast: ProgramNode) -> List[NoetaError]:
        """
        Validate the entire AST.

        Args:
            ast: The program AST to validate

        Returns:
            List of errors found (empty list if no errors)
        """
        self.errors = []

        for statement in ast.statements:
            try:
                self.visit(statement)
            except NoetaError as e:
                self.errors.append(e)

        return self.errors

    def visit(self, node: ASTNode):
        """
        Dispatch to appropriate visitor method.

        Uses visitor pattern: calls visit_<NodeType> for each node.
        Falls back to generic_visit if no specific visitor exists.
        """
        method_name = f'visit_{node.__class__.__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ASTNode):
        """
        Called if no explicit visitor exists for a node type.

        This is fine - it means we don't need special validation for that node.
        """
        pass

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_source_line(self, line_num: int) -> str:
        """Get source line for error context."""
        if not self.source_lines or line_num < 1 or line_num > len(self.source_lines):
            return ""
        return self.source_lines[line_num - 1]

    def _suggest_dataset(self, attempted: str) -> Optional[str]:
        """
        Suggest similar dataset name using Levenshtein distance.

        Args:
            attempted: The dataset name that was attempted

        Returns:
            Suggestion message or None
        """
        available = self.symbol_table.get_all_names()
        if not available:
            return None

        suggestions = suggest_similar(attempted, available, max_suggestions=1)
        if suggestions:
            return f"Did you mean '{suggestions[0]}'?"
        return None

    def _check_dataset_exists(self, name: str, node: ASTNode) -> DatasetInfo:
        """
        Check that a dataset has been defined, raise error if not.

        Args:
            name: Dataset name to check
            node: AST node (for error context)

        Returns:
            DatasetInfo for the dataset

        Raises:
            NoetaError: If dataset is not defined
        """
        info = self.symbol_table.lookup(name)
        if not info:
            available = self.symbol_table.get_all_names()
            hint = f"Available datasets: {', '.join(available)}" if available else "No datasets have been loaded yet"

            raise create_semantic_error(
                message=f"Dataset '{name}' has not been loaded or created",
                line=node.line,
                column=node.column,
                source_line=self._get_source_line(node.line),
                length=len(name),
                hint=hint,
                suggestion=self._suggest_dataset(name)
            )
        return info

    def _check_column_exists(self, dataset_info: DatasetInfo, column: str, node: ASTNode):
        """
        Check that a column exists in a dataset (if we know the schema).

        Args:
            dataset_info: Dataset information
            column: Column name to check
            node: AST node (for error context)

        Raises:
            NoetaError: If column doesn't exist (and we know the schema)
        """
        # If we don't know the columns, can't validate
        if not dataset_info.columns:
            return

        if not dataset_info.has_column(column):
            available = list(dataset_info.columns.keys())
            hint = f"Available columns in '{dataset_info.name}': {', '.join(available)}"

            raise create_semantic_error(
                message=f"Column '{column}' does not exist in dataset '{dataset_info.name}'",
                line=node.line,
                column=node.column,
                source_line=self._get_source_line(node.line),
                length=len(column),
                hint=hint,
                suggestion=None  # Could add column name suggestions here
            )

    def _check_column_type(self, dataset_info: DatasetInfo, column: str, expected_type: DataType, node: ASTNode):
        """
        Check that a column has the expected type (if we know types).

        Args:
            dataset_info: Dataset information
            column: Column name to check
            expected_type: Expected data type
            node: AST node (for error context)

        Raises:
            NoetaError: If column type doesn't match
        """
        # If we don't know the columns or types, can't validate
        if not dataset_info.columns or column not in dataset_info.columns:
            return

        actual_type = dataset_info.get_column_type(column)

        # If type is unknown, can't validate
        if actual_type == DataType.UNKNOWN or expected_type == DataType.UNKNOWN:
            return

        if actual_type != expected_type:
            raise create_semantic_error(
                message=f"Column '{column}' has type {actual_type.value}, expected {expected_type.value}",
                line=node.line,
                column=node.column,
                source_line=self._get_source_line(node.line),
                length=len(column),
                hint=f"This operation requires a {expected_type.value} column",
                suggestion=None
            )

    # =========================================================================
    # VISITOR METHODS - Will be implemented in Days 2-4
    # =========================================================================

    def visit_LoadNode(self, node: LoadNode):
        """
        Validate load operation and register dataset.

        For load operations, we register the dataset but don't know the schema
        until runtime (unless we want to actually load the file during validation,
        which would be slow).
        """
        # Register the dataset in symbol table
        # We don't know columns until runtime, so register with empty schema
        dataset_info = DatasetInfo(
            name=node.alias,
            columns={},  # Unknown until runtime
            source=node.filepath
        )
        self.symbol_table.define(node.alias, dataset_info)

    def visit_SaveNode(self, node: SaveNode):
        """Validate save operation."""
        # Check source dataset exists
        self._check_dataset_exists(node.source_alias, node)
        # Save doesn't create a new dataset, so no need to register anything

    # Placeholder visitors for other operations - will be implemented in Days 2-4
    # These will follow the pattern:
    # 1. Check source dataset exists
    # 2. Check columns exist (if applicable)
    # 3. Check types (if applicable)
    # 4. Register result dataset (if creates new dataset)

    def visit_SelectNode(self, node: SelectNode):
        """Validate select operation (Day 2-3)."""
        # Check source exists
        source_info = self._check_dataset_exists(node.source_alias, node)

        # TODO: Validate columns exist (if we know them)

        # Register result dataset
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},  # Would need to track selected columns
            source=f"select from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_FilterNode(self, node: FilterNode):
        """Validate filter operation (Day 2-3)."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        # Register result dataset (same columns as source)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"filter from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_UpdatedFilterNode(self, node):
        """Validate updated filter operation with rich where clause."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        # Register result dataset (same columns as source)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"filter from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_GroupByNode(self, node: GroupByNode):
        """Validate groupby operation (Day 3)."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        # Register result dataset
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},  # Columns depend on aggregation
            source=f"groupby from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_JoinNode(self, node: JoinNode):
        """Validate join operation (Day 3)."""
        # Check both datasets exist
        left_info = self._check_dataset_exists(node.alias1, node)
        right_info = self._check_dataset_exists(node.alias2, node)

        # Register result dataset
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},  # Would be union of left and right columns
            source=f"join {node.alias1} with {node.alias2}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_MergeNode(self, node: MergeNode):
        """Validate merge operation (Day 3)."""
        # Check both datasets exist
        left_info = self._check_dataset_exists(node.left_alias, node)
        right_info = self._check_dataset_exists(node.right_alias, node)

        # Register result dataset
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},  # Would be union of left and right columns
            source=f"merge {node.left_alias} with {node.right_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # Display operations don't create new datasets
    def visit_DescribeNode(self, node: DescribeNode):
        """Validate describe operation."""
        self._check_dataset_exists(node.source_alias, node)

    def visit_InfoNode(self, node: InfoNode):
        """Validate info operation."""
        self._check_dataset_exists(node.source_alias, node)

    def visit_HeadNode(self, node: HeadNode):
        """Validate head operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        # If new_alias is provided, register as new dataset
        if node.new_alias:
            result_info = DatasetInfo(
                name=node.new_alias,
                columns=source_info.columns.copy(),
                source=f"head from {node.source_alias}"
            )
            self.symbol_table.define(node.new_alias, result_info)

    def visit_TailNode(self, node: TailNode):
        """Validate tail operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        # If new_alias is provided, register as new dataset
        if node.new_alias:
            result_info = DatasetInfo(
                name=node.new_alias,
                columns=source_info.columns.copy(),
                source=f"tail from {node.source_alias}"
            )
            self.symbol_table.define(node.new_alias, result_info)

    def visit_SampleNode(self, node: SampleNode):
        """Validate sample operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        # If new_alias is provided, register as new dataset
        if node.new_alias:
            result_info = DatasetInfo(
                name=node.new_alias,
                columns=source_info.columns.copy(),
                source=f"sample from {node.source_alias}"
            )
            self.symbol_table.define(node.new_alias, result_info)

    # =========================================================================
    # SELECTION & PROJECTION OPERATIONS
    # =========================================================================

    def visit_SelectByTypeNode(self, node: SelectByTypeNode):
        """Validate select_by_type operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        # Register result dataset
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},  # Would need to filter by type
            source=f"select_by_type from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ILocNode(self, node: ILocNode):
        """Validate iloc operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        # Register result dataset
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"iloc from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_LocNode(self, node: LocNode):
        """Validate loc operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        # Register result dataset
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"loc from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_RenameColumnsNode(self, node: RenameColumnsNode):
        """Validate rename operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        # Register result dataset with renamed columns
        new_columns = source_info.columns.copy()
        # TODO: Update column names in schema based on rename_map
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=new_columns,
            source=f"rename from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ReorderColumnsNode(self, node: ReorderColumnsNode):
        """Validate reorder operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        # Register result dataset (same columns, different order)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"reorder from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # =========================================================================
    # FILTERING OPERATIONS
    # =========================================================================

    def visit_FilterBetweenNode(self, node: FilterBetweenNode):
        """Validate filter_between operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        # Register result dataset
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"filter_between from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_FilterIsInNode(self, node: FilterIsInNode):
        """Validate filter_isin operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"filter_isin from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_FilterContainsNode(self, node: FilterContainsNode):
        """Validate filter_contains operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"filter_contains from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_FilterStartsWithNode(self, node: FilterStartsWithNode):
        """Validate filter_startswith operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"filter_startswith from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_FilterEndsWithNode(self, node: FilterEndsWithNode):
        """Validate filter_endswith operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"filter_endswith from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_FilterRegexNode(self, node: FilterRegexNode):
        """Validate filter_regex operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"filter_regex from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_FilterNullNode(self, node: FilterNullNode):
        """Validate filter_null operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"filter_null from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_FilterNotNullNode(self, node: FilterNotNullNode):
        """Validate filter_notnull operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"filter_notnull from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_FilterDuplicatesNode(self, node: FilterDuplicatesNode):
        """Validate filter_duplicates operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"filter_duplicates from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # =========================================================================
    # SORTING OPERATIONS
    # =========================================================================

    def visit_SortNode(self, node: SortNode):
        """Validate sort operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"sort from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # =========================================================================
    # STATISTICAL & DISPLAY OPERATIONS
    # =========================================================================

    def visit_SummaryNode(self, node: SummaryNode):
        """Validate summary operation."""
        self._check_dataset_exists(node.source_alias, node)

    def visit_UniqueNode(self, node: UniqueNode):
        """Validate unique operation."""
        self._check_dataset_exists(node.source_alias, node)

    def visit_ValueCountsNode(self, node: ValueCountsNode):
        """Validate value_counts operation."""
        self._check_dataset_exists(node.source_alias, node)

    # =========================================================================
    # DATA CLEANING OPERATIONS
    # =========================================================================

    def visit_DropNANode(self, node: DropNANode):
        """Validate dropna operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"dropna from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_FillNANode(self, node: FillNANode):
        """Validate fillna operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"fillna from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # =========================================================================
    # TRANSFORMATION OPERATIONS
    # =========================================================================

    def visit_MutateNode(self, node: MutateNode):
        """Validate mutate operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        # Mutate adds/modifies columns
        new_columns = source_info.columns.copy()
        # TODO: Track new/modified columns
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=new_columns,
            source=f"mutate from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ApplyNode(self, node: ApplyNode):
        """Validate apply operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)

        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"apply from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # =========================================================================
    # DAY 3: TRANSFORMATION OPERATIONS
    # =========================================================================

    # Math Operations (7 operations)
    def visit_RoundNode(self, node):
        """Validate round operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"round from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_AbsNode(self, node):
        """Validate abs operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"abs from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_SqrtNode(self, node):
        """Validate sqrt operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"sqrt from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_PowerNode(self, node):
        """Validate power operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"power from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_LogNode(self, node):
        """Validate log operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"log from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_CeilNode(self, node):
        """Validate ceil operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"ceil from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_FloorNode(self, node):
        """Validate floor operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"floor from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # String Operations (14 operations)
    def visit_UpperNode(self, node):
        """Validate upper operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"upper from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_LowerNode(self, node):
        """Validate lower operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"lower from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_StripNode(self, node):
        """Validate strip operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"strip from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_LStripNode(self, node):
        """Validate lstrip operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"lstrip from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_RStripNode(self, node):
        """Validate rstrip operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"rstrip from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ReplaceNode(self, node):
        """Validate replace operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"replace from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_SplitNode(self, node):
        """Validate split operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"split from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ConcatNode(self, node):
        """Validate concat operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"concat from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_SubstringNode(self, node):
        """Validate substring operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"substring from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_LengthNode(self, node):
        """Validate length operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"length from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_TitleNode(self, node):
        """Validate title operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"title from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_CapitalizeNode(self, node):
        """Validate capitalize operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"capitalize from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ExtractRegexNode(self, node):
        """Validate extract_regex operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"extract_regex from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_FindNode(self, node):
        """Validate find operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"find from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # Date/Time Operations (14+ operations)
    def visit_ParseDatetimeNode(self, node):
        """Validate parse_datetime operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"parse_datetime from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ExtractYearNode(self, node):
        """Validate extract_year operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"extract_year from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ExtractMonthNode(self, node):
        """Validate extract_month operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"extract_month from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ExtractDayNode(self, node):
        """Validate extract_day operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"extract_day from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ExtractHourNode(self, node):
        """Validate extract_hour operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"extract_hour from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ExtractMinuteNode(self, node):
        """Validate extract_minute operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"extract_minute from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ExtractSecondNode(self, node):
        """Validate extract_second operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"extract_second from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ExtractDayOfWeekNode(self, node):
        """Validate extract_dayofweek operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"extract_dayofweek from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ExtractDayOfYearNode(self, node):
        """Validate extract_dayofyear operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"extract_dayofyear from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ExtractWeekOfYearNode(self, node):
        """Validate extract_weekofyear operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"extract_weekofyear from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ExtractQuarterNode(self, node):
        """Validate extract_quarter operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"extract_quarter from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_DateAddNode(self, node):
        """Validate date_add operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"date_add from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_DateSubtractNode(self, node):
        """Validate date_subtract operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"date_subtract from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_FormatDateTimeNode(self, node):
        """Validate format_datetime operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"format_datetime from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_DateDiffNode(self, node):
        """Validate date_diff operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"date_diff from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # Type Operations (2 operations)
    def visit_AsTypeNode(self, node):
        """Validate astype operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"astype from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ToNumericNode(self, node):
        """Validate to_numeric operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"to_numeric from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # Encoding Operations (6 operations)
    def visit_OneHotEncodeNode(self, node):
        """Validate one_hot_encode operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},  # Schema changes with one-hot encoding
            source=f"one_hot_encode from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_LabelEncodeNode(self, node):
        """Validate label_encode operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"label_encode from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_OrdinalEncodeNode(self, node):
        """Validate ordinal_encode operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"ordinal_encode from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_TargetEncodeNode(self, node):
        """Validate target_encode operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"target_encode from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # Scaling Operations (4 operations)
    def visit_StandardScaleNode(self, node):
        """Validate standard_scale operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"standard_scale from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_MinMaxScaleNode(self, node):
        """Validate minmax_scale operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"minmax_scale from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_RobustScaleNode(self, node):
        """Validate robust_scale operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"robust_scale from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_MaxAbsScaleNode(self, node):
        """Validate maxabs_scale operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"maxabs_scale from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # =========================================================================
    # DAY 4: REMAINING OPERATIONS (Reshaping, Combining, Cumulative, etc.)
    # =========================================================================

    # Reshaping Operations (9 operations)
    def visit_PivotNode(self, node):
        """Validate pivot operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},  # Schema changes with pivot
            source=f"pivot from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_PivotTableNode(self, node):
        """Validate pivot_table operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},  # Schema changes with pivot
            source=f"pivot_table from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_MeltNode(self, node):
        """Validate melt operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},  # Schema changes with melt
            source=f"melt from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_StackNode(self, node):
        """Validate stack operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},  # Schema changes with stack
            source=f"stack from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_UnstackNode(self, node):
        """Validate unstack operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},  # Schema changes with unstack
            source=f"unstack from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_TransposeNode(self, node):
        """Validate transpose operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},  # Schema completely changes
            source=f"transpose from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ExplodeNode(self, node):
        """Validate explode operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"explode from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_NormalizeNode(self, node):
        """Validate normalize operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},  # Schema changes with normalization
            source=f"normalize from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_CrosstabNode(self, node):
        """Validate crosstab operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},  # Schema changes
            source=f"crosstab from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # Combining Operations (5 operations)
    def visit_ConcatVerticalNode(self, node):
        """Validate concat_vertical operation."""
        # Check both datasets exist
        self._check_dataset_exists(node.alias1, node)
        self._check_dataset_exists(node.alias2, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},
            source=f"concat_vertical {node.alias1}, {node.alias2}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ConcatHorizontalNode(self, node):
        """Validate concat_horizontal operation."""
        # Check both datasets exist
        self._check_dataset_exists(node.alias1, node)
        self._check_dataset_exists(node.alias2, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},
            source=f"concat_horizontal {node.alias1}, {node.alias2}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_AppendNode(self, node):
        """Validate append operation."""
        # Check both datasets exist
        self._check_dataset_exists(node.alias1, node)
        self._check_dataset_exists(node.alias2, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},
            source=f"append {node.alias1}, {node.alias2}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_CrossJoinNode(self, node):
        """Validate cross_join operation."""
        # Check both datasets exist
        self._check_dataset_exists(node.alias1, node)
        self._check_dataset_exists(node.alias2, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},
            source=f"cross_join {node.alias1}, {node.alias2}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_DifferenceNode(self, node):
        """Validate difference operation."""
        # Check both datasets exist
        self._check_dataset_exists(node.alias1, node)
        self._check_dataset_exists(node.alias2, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns={},
            source=f"difference {node.alias1}, {node.alias2}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # Cumulative Operations (4 operations)
    def visit_CumSumNode(self, node):
        """Validate cumsum operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"cumsum from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_CumMaxNode(self, node):
        """Validate cummax operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"cummax from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_CumMinNode(self, node):
        """Validate cummin operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"cummin from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_CumProdNode(self, node):
        """Validate cumprod operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"cumprod from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # Time Series Operations (3 operations)
    def visit_PctChangeNode(self, node):
        """Validate pct_change operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"pct_change from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_DiffNode(self, node):
        """Validate diff operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"diff from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ShiftNode(self, node):
        """Validate shift operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"shift from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # Validation Operations (3 operations)
    def visit_AssertUniqueNode(self, node):
        """Validate assert_unique operation."""
        self._check_dataset_exists(node.source_alias, node)
        # Assert operations don't create new datasets

    def visit_AssertNoNullsNode(self, node):
        """Validate assert_no_nulls operation."""
        self._check_dataset_exists(node.source_alias, node)
        # Assert operations don't create new datasets

    def visit_AssertRangeNode(self, node):
        """Validate assert_range operation."""
        self._check_dataset_exists(node.source_alias, node)
        # Assert operations don't create new datasets

    # Index Operations (5 operations)
    def visit_SetIndexNode(self, node):
        """Validate set_index operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"set_index from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ResetIndexNode(self, node):
        """Validate reset_index operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"reset_index from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_SortIndexNode(self, node):
        """Validate sort_index operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"sort_index from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ReindexNode(self, node):
        """Validate reindex operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"reindex from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_SetMultiIndexNode(self, node):
        """Validate set_multiindex operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"set_multiindex from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # Boolean Operations (4 operations)
    def visit_AnyNode(self, node):
        """Validate any operation."""
        self._check_dataset_exists(node.source_alias, node)
        # Boolean operations typically don't create new datasets

    def visit_AllNode(self, node):
        """Validate all operation."""
        self._check_dataset_exists(node.source_alias, node)
        # Boolean operations typically don't create new datasets

    def visit_CountTrueNode(self, node):
        """Validate count_true operation."""
        self._check_dataset_exists(node.source_alias, node)
        # Count operations typically don't create new datasets

    def visit_CompareNode(self, node):
        """Validate compare operation."""
        # Check both datasets exist
        self._check_dataset_exists(node.alias1, node)
        self._check_dataset_exists(node.alias2, node)
        # Compare typically returns a result, not a dataset

    # Window Operations (4 operations)
    def visit_RankNode(self, node):
        """Validate rank operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"rank from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_WindowRankNode(self, node):
        """Validate window_rank operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"window_rank from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_WindowLagNode(self, node):
        """Validate window_lag operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"window_lag from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_WindowLeadNode(self, node):
        """Validate window_lead operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"window_lead from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # Rolling/Expanding Operations (10 operations)
    def visit_RollingNode(self, node):
        """Validate rolling operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"rolling from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_RollingMeanNode(self, node):
        """Validate rolling_mean operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"rolling_mean from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_RollingSumNode(self, node):
        """Validate rolling_sum operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"rolling_sum from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_RollingStdNode(self, node):
        """Validate rolling_std operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"rolling_std from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_RollingMinNode(self, node):
        """Validate rolling_min operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"rolling_min from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_RollingMaxNode(self, node):
        """Validate rolling_max operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"rolling_max from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ExpandingMeanNode(self, node):
        """Validate expanding_mean operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"expanding_mean from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ExpandingSumNode(self, node):
        """Validate expanding_sum operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"expanding_sum from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ExpandingMinNode(self, node):
        """Validate expanding_min operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"expanding_min from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_ExpandingMaxNode(self, node):
        """Validate expanding_max operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"expanding_max from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # Binning Operations (2 operations)
    def visit_BinningNode(self, node):
        """Validate binning operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"binning from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_QcutNode(self, node):
        """Validate qcut operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"qcut from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # Map Operations (2 operations)
    def visit_ApplyMapNode(self, node):
        """Validate applymap operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"applymap from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    def visit_MapValuesNode(self, node):
        """Validate map_values operation."""
        source_info = self._check_dataset_exists(node.source_alias, node)
        result_info = DatasetInfo(
            name=node.new_alias,
            columns=source_info.columns.copy(),
            source=f"map_values from {node.source_alias}"
        )
        self.symbol_table.define(node.new_alias, result_info)

    # Correlation/Covariance Operations (2 operations)
    def visit_CorrNode(self, node):
        """Validate corr operation."""
        self._check_dataset_exists(node.source_alias, node)
        # Correlation typically returns a matrix, not a new dataset

    def visit_CovNode(self, node):
        """Validate cov operation."""
        self._check_dataset_exists(node.source_alias, node)
        # Covariance typically returns a matrix, not a new dataset

    # =========================================================================
    # SEMANTIC VALIDATION COMPLETE
    # =========================================================================
    # All major operation categories now have visitor methods:
    # - Data I/O (Load, Save)
    # - Selection & Projection
    # - Filtering (basic and advanced)
    # - Transformation (Math, String, Date/Time, Type, Encoding, Scaling)
    # - Reshaping (Pivot, Melt, Stack, Unstack, Transpose, etc.)
    # - Combining (Join, Merge, Concat, Append, etc.)
    # - Cumulative (CumSum, CumMax, CumMin, CumProd)
    # - Time Series (PctChange, Diff, Shift)
    # - Validation (Assert operations)
    # - Index Operations (SetIndex, ResetIndex, etc.)
    # - Boolean Operations (Any, All, CountTrue, Compare)
    # - Window Operations (Rank, Lag, Lead)
    # - Rolling/Expanding (Rolling/Expanding Mean/Sum/Min/Max/Std)
    # - Binning (Binning, Qcut)
    # - Map Operations (ApplyMap, MapValues)
    # - Statistical (Corr, Cov)
    #
    # Total visitor methods: ~130+
    # =========================================================================
