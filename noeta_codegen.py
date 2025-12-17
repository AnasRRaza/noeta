"""
Noeta Code Generator - Converts AST to executable Python/Pandas code
"""
from noeta_ast import *
from typing import Dict, Any

class CodeGenerator:
    def __init__(self):
        self.symbol_table: Dict[str, Any] = {}
        self.imports = set()
        self.code_lines = []
        self.last_plot = None
    
    def generate(self, ast: ProgramNode) -> str:
        # Add standard imports
        self.imports.add("import pandas as pd")
        self.imports.add("import numpy as np")
        self.imports.add("import matplotlib.pyplot as plt")
        self.imports.add("import seaborn as sns")
        self.imports.add("from scipy import stats")
        
        # Generate code for each statement
        for stmt in ast.statements:
            self.visit(stmt)
        
        # Combine imports and code
        result = "\n".join(sorted(self.imports))
        result += "\n\n# Configure visualization settings\n"
        result += "plt.style.use('seaborn-v0_8-darkgrid')\n"
        result += "sns.set_palette('husl')\n\n"
        result += "\n".join(self.code_lines)

        # Show plots if any visualization was created
        # Behavior depends on execution environment:
        # - Jupyter: plots display inline (kernel handles it)
        # - VS Code/CLI: plots open in separate windows (plt.show())
        if self.last_plot:
            result += "\n\n# Display plots\n"
            result += "plt.tight_layout()\n"
            result += "try:\n"
            result += "    get_ipython()\n"
            result += "    # Running in Jupyter/IPython - don't show (kernel will display inline)\n"
            result += "except NameError:\n"
            result += "    # Running in VS Code/CLI - show plots in separate windows\n"
            result += "    plt.show()"

        return result
    
    def visit(self, node: ASTNode):
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def generic_visit(self, node):
        raise NotImplementedError(f"No visitor for {node.__class__.__name__}")
    
    # Data manipulation visitors
    def visit_LoadNode(self, node: LoadNode):
        """
        UNIFIED SYNTAX v2.0: Generate code for consolidated load operation

        Handles all formats (csv, json, excel, parquet, sql) with optional parameters.
        Format is auto-detected from file extension or explicitly specified.
        """
        format_type = node.format
        filepath = node.filepath
        params = node.params or {}

        # Build parameters string
        params_str = self._build_params_str(params) if params else ""

        # Generate appropriate pandas read function based on format
        if format_type == 'csv':
            if params_str:
                code = f"{node.alias} = pd.read_csv('{filepath}', {params_str})"
            else:
                code = f"{node.alias} = pd.read_csv('{filepath}')"

        elif format_type == 'json':
            if params_str:
                code = f"{node.alias} = pd.read_json('{filepath}', {params_str})"
            else:
                code = f"{node.alias} = pd.read_json('{filepath}')"

        elif format_type == 'excel':
            if params_str:
                code = f"{node.alias} = pd.read_excel('{filepath}', {params_str})"
            else:
                code = f"{node.alias} = pd.read_excel('{filepath}')"

        elif format_type == 'parquet':
            if params_str:
                code = f"{node.alias} = pd.read_parquet('{filepath}', {params_str})"
            else:
                code = f"{node.alias} = pd.read_parquet('{filepath}')"

        elif format_type == 'sql':
            # Add SQLAlchemy import for SQL
            self.imports.add("from sqlalchemy import create_engine")

            # Get connection string from params
            connection = params.get('connection', filepath)
            self.code_lines.append(f"_engine = create_engine('{connection}')")

            # Build SQL-specific parameters
            sql_params = {k: v for k, v in params.items() if k not in ['connection', 'params']}
            sql_params_str = self._build_params_str(sql_params) if sql_params else ""

            # Handle query parameters
            if 'params' in params:
                query_params = params['params']
                params_dict_str = self._format_value(query_params)
                if sql_params_str:
                    code = f"{node.alias} = pd.read_sql('''{filepath}''', con=_engine, params={params_dict_str}, {sql_params_str})"
                else:
                    code = f"{node.alias} = pd.read_sql('''{filepath}''', con=_engine, params={params_dict_str})"
            else:
                if sql_params_str:
                    code = f"{node.alias} = pd.read_sql('''{filepath}''', con=_engine, {sql_params_str})"
                else:
                    code = f"{node.alias} = pd.read_sql('''{filepath}''', con=_engine)"
        else:
            # Fallback to CSV if format unknown
            if params_str:
                code = f"{node.alias} = pd.read_csv('{filepath}', {params_str})"
            else:
                code = f"{node.alias} = pd.read_csv('{filepath}')"

        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Loaded {filepath} as {node.alias}: {{len({node.alias})}} rows, {{len({node.alias}.columns)}} columns')")

    def visit_LoadCSVNode(self, node: LoadCSVNode):
        """Generate code for enhanced CSV loading with parameters"""
        params_str = self._build_params_str(node.params)
        if params_str:
            code = f"{node.alias} = pd.read_csv('{node.filepath}', {params_str})"
        else:
            code = f"{node.alias} = pd.read_csv('{node.filepath}')"

        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Loaded {node.filepath} as {node.alias}: {{len({node.alias})}} rows, {{len({node.alias}.columns)}} columns')")

    def visit_LoadJSONNode(self, node: LoadJSONNode):
        """Generate code for JSON loading with parameters"""
        params_str = self._build_params_str(node.params)
        if params_str:
            code = f"{node.alias} = pd.read_json('{node.filepath}', {params_str})"
        else:
            code = f"{node.alias} = pd.read_json('{node.filepath}')"

        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Loaded {node.filepath} as {node.alias}')")

    def visit_LoadExcelNode(self, node: LoadExcelNode):
        """Generate code for Excel loading with parameters"""
        params_str = self._build_params_str(node.params)
        if params_str:
            code = f"{node.alias} = pd.read_excel('{node.filepath}', {params_str})"
        else:
            code = f"{node.alias} = pd.read_excel('{node.filepath}')"

        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Loaded {node.filepath} as {node.alias}: {{len({node.alias})}} rows, {{len({node.alias}.columns)}} columns')")

    def visit_LoadParquetNode(self, node: LoadParquetNode):
        """Generate code for Parquet loading with parameters"""
        params_str = self._build_params_str(node.params)
        if params_str:
            code = f"{node.alias} = pd.read_parquet('{node.filepath}', {params_str})"
        else:
            code = f"{node.alias} = pd.read_parquet('{node.filepath}')"

        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Loaded {node.filepath} as {node.alias}: {{len({node.alias})}} rows, {{len({node.alias}.columns)}} columns')")

    def visit_LoadSQLNode(self, node: LoadSQLNode):
        """Generate code for SQL loading with parameters"""
        # Add SQLAlchemy import if needed
        self.imports.add("from sqlalchemy import create_engine")

        # Create engine from connection string
        self.code_lines.append(f"_engine = create_engine('{node.connection}')")

        # Build parameters (exclude 'params' from params dict, handle separately)
        sql_params = {k: v for k, v in node.params.items() if k != 'params'}
        params_str = self._build_params_str(sql_params)

        # Handle query parameters
        if 'params' in node.params:
            query_params = node.params['params']
            params_dict_str = self._format_value(query_params)
            if params_str:
                code = f"{node.alias} = pd.read_sql('''{node.query}''', con=_engine, params={params_dict_str}, {params_str})"
            else:
                code = f"{node.alias} = pd.read_sql('''{node.query}''', con=_engine, params={params_dict_str})"
        else:
            if params_str:
                code = f"{node.alias} = pd.read_sql('''{node.query}''', con=_engine, {params_str})"
            else:
                code = f"{node.alias} = pd.read_sql('''{node.query}''', con=_engine)"

        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Loaded from SQL as {node.alias}: {{len({node.alias})}} rows, {{len({node.alias}.columns)}} columns')")

    def visit_SaveCSVNode(self, node: SaveCSVNode):
        """Generate code for enhanced CSV saving with parameters"""
        params_str = self._build_params_str(node.params)
        if params_str:
            code = f"{node.source_alias}.to_csv('{node.filepath}', {params_str})"
        else:
            code = f"{node.source_alias}.to_csv('{node.filepath}')"

        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Saved {node.source_alias} to {node.filepath}')")

    def visit_SaveJSONNode(self, node: SaveJSONNode):
        """Generate code for JSON saving with parameters"""
        params_str = self._build_params_str(node.params)
        if params_str:
            code = f"{node.source_alias}.to_json('{node.filepath}', {params_str})"
        else:
            code = f"{node.source_alias}.to_json('{node.filepath}')"

        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Saved {node.source_alias} to {node.filepath}')")

    def visit_SaveExcelNode(self, node: SaveExcelNode):
        """Generate code for Excel saving with parameters"""
        params_str = self._build_params_str(node.params)
        if params_str:
            code = f"{node.source_alias}.to_excel('{node.filepath}', {params_str})"
        else:
            code = f"{node.source_alias}.to_excel('{node.filepath}')"

        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Saved {node.source_alias} to {node.filepath}')")

    def visit_SaveParquetNode(self, node: SaveParquetNode):
        """Generate code for Parquet saving with parameters"""
        params_str = self._build_params_str(node.params)
        if params_str:
            code = f"{node.source_alias}.to_parquet('{node.filepath}', {params_str})"
        else:
            code = f"{node.source_alias}.to_parquet('{node.filepath}')"

        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Saved {node.source_alias} to {node.filepath}')")

    # Phase 2: Selection & Projection Code Generators

    def visit_SelectByTypeNode(self, node: SelectByTypeNode):
        """Generate code for selecting columns by data type"""
        # Map DSL type names to pandas dtype categories
        type_mapping = {
            'numeric': 'number',
            'number': 'number',
            'int': 'int',
            'integer': 'int',
            'float': 'float',
            'string': 'object',
            'str': 'object',
            'object': 'object',
            'datetime': 'datetime',
            'date': 'datetime',
            'bool': 'bool',
            'boolean': 'bool',
            'category': 'category'
        }

        pandas_type = type_mapping.get(node.dtype.lower(), node.dtype)
        code = f"{node.new_alias} = {node.source_alias}.select_dtypes(include=['{pandas_type}'])"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Selected {{len({node.new_alias}.columns)}} columns of type {node.dtype} from {node.source_alias}')")
        self.symbol_table[node.new_alias] = True

    def visit_HeadNode(self, node: HeadNode):
        """Generate code for getting first N rows"""
        if node.new_alias:
            # Create alias
            code = f"{node.new_alias} = {node.source_alias}.head({node.n_rows})"
            self.code_lines.append(code)
            self.code_lines.append(f"print(f'Created alias {node.new_alias} with first {node.n_rows} rows')")
            self.symbol_table[node.new_alias] = True
        else:
            # Just display
            self.code_lines.append(f"print(f'\\nFirst {node.n_rows} rows of {node.source_alias}:')")
            self.code_lines.append(f"print({node.source_alias}.head({node.n_rows}))")

    def visit_TailNode(self, node: TailNode):
        """Generate code for getting last N rows"""
        if node.new_alias:
            # Create alias
            code = f"{node.new_alias} = {node.source_alias}.tail({node.n_rows})"
            self.code_lines.append(code)
            self.code_lines.append(f"print(f'Created alias {node.new_alias} with last {node.n_rows} rows')")
            self.symbol_table[node.new_alias] = True
        else:
            # Just display
            self.code_lines.append(f"print(f'\\nLast {node.n_rows} rows of {node.source_alias}:')")
            self.code_lines.append(f"print({node.source_alias}.tail({node.n_rows}))")

    def visit_ILocNode(self, node: ILocNode):
        """Generate code for position-based indexing"""
        # Handle row slicing
        if isinstance(node.row_slice, tuple):
            row_slice_str = f"{node.row_slice[0]}:{node.row_slice[1]}"
        else:
            row_slice_str = str(node.row_slice)

        # Handle optional column slicing
        if node.col_slice:
            if isinstance(node.col_slice, tuple):
                col_slice_str = f"{node.col_slice[0]}:{node.col_slice[1]}"
            else:
                col_slice_str = str(node.col_slice)
            code = f"{node.new_alias} = {node.source_alias}.iloc[{row_slice_str}, {col_slice_str}]"
        else:
            code = f"{node.new_alias} = {node.source_alias}.iloc[{row_slice_str}]"

        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Selected rows by position from {node.source_alias}')")
        self.symbol_table[node.new_alias] = True

    def visit_LocNode(self, node: LocNode):
        """Generate code for label-based indexing"""
        # Format row labels
        if isinstance(node.row_labels, list):
            row_labels_str = str(node.row_labels)
        else:
            row_labels_str = self._format_value(node.row_labels)

        # Handle optional column labels
        if node.col_labels:
            col_labels_str = str(node.col_labels)
            code = f"{node.new_alias} = {node.source_alias}.loc[{row_labels_str}, {col_labels_str}]"
        else:
            code = f"{node.new_alias} = {node.source_alias}.loc[{row_labels_str}]"

        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Selected rows by label from {node.source_alias}')")
        self.symbol_table[node.new_alias] = True

    def visit_RenameColumnsNode(self, node: RenameColumnsNode):
        """Generate code for renaming columns"""
        mapping_str = str(node.mapping)
        code = f"{node.new_alias} = {node.source_alias}.rename(columns={mapping_str})"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Renamed {{len({mapping_str})}} columns in {node.source_alias}')")
        self.symbol_table[node.new_alias] = True

    def visit_ReorderColumnsNode(self, node: ReorderColumnsNode):
        """Generate code for reordering columns"""
        column_order_str = str(node.column_order)
        code = f"{node.new_alias} = {node.source_alias}[{column_order_str}]"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Reordered columns in {node.source_alias}')")
        self.symbol_table[node.new_alias] = True

    # Phase 3: Filtering Code Generators

    def visit_FilterBetweenNode(self, node: FilterBetweenNode):
        """Generate code for filtering rows where column value is between min and max"""
        min_str = self._format_value(node.min_value)
        max_str = self._format_value(node.max_value)
        code = f"{node.new_alias} = {node.source_alias}[{node.source_alias}['{node.column}'].between({min_str}, {max_str})]"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Filtered {{len({node.new_alias})}} rows where {node.column} is between {min_str} and {max_str}')")
        self.symbol_table[node.new_alias] = True

    def visit_FilterIsInNode(self, node: FilterIsInNode):
        """Generate code for filtering rows where column value is in a list"""
        values_str = str(node.values)
        code = f"{node.new_alias} = {node.source_alias}[{node.source_alias}['{node.column}'].isin({values_str})]"
        self.code_lines.append(code)
        # Don't use f-string for values to avoid quote conflicts
        self.code_lines.append(f"print(f'Filtered {{len({node.new_alias})}} rows where {node.column} is in specified values')")
        self.symbol_table[node.new_alias] = True

    def visit_FilterContainsNode(self, node: FilterContainsNode):
        """Generate code for filtering rows where column contains a pattern"""
        code = f"{node.new_alias} = {node.source_alias}[{node.source_alias}['{node.column}'].str.contains('{node.pattern}', na=False)]"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Filtered {{len({node.new_alias})}} rows where {node.column} contains \"{node.pattern}\"')")
        self.symbol_table[node.new_alias] = True

    def visit_FilterStartsWithNode(self, node: FilterStartsWithNode):
        """Generate code for filtering rows where column starts with a pattern"""
        code = f"{node.new_alias} = {node.source_alias}[{node.source_alias}['{node.column}'].str.startswith('{node.pattern}', na=False)]"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Filtered {{len({node.new_alias})}} rows where {node.column} starts with \"{node.pattern}\"')")
        self.symbol_table[node.new_alias] = True

    def visit_FilterEndsWithNode(self, node: FilterEndsWithNode):
        """Generate code for filtering rows where column ends with a pattern"""
        code = f"{node.new_alias} = {node.source_alias}[{node.source_alias}['{node.column}'].str.endswith('{node.pattern}', na=False)]"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Filtered {{len({node.new_alias})}} rows where {node.column} ends with \"{node.pattern}\"')")
        self.symbol_table[node.new_alias] = True

    def visit_FilterRegexNode(self, node: FilterRegexNode):
        """Generate code for filtering rows where column matches a regex pattern"""
        code = f"{node.new_alias} = {node.source_alias}[{node.source_alias}['{node.column}'].str.match('{node.pattern}', na=False)]"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Filtered {{len({node.new_alias})}} rows where {node.column} matches regex \"{node.pattern}\"')")
        self.symbol_table[node.new_alias] = True

    def visit_FilterNullNode(self, node: FilterNullNode):
        """Generate code for filtering rows where column is null"""
        code = f"{node.new_alias} = {node.source_alias}[{node.source_alias}['{node.column}'].isnull()]"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Filtered {{len({node.new_alias})}} rows where {node.column} is null')")
        self.symbol_table[node.new_alias] = True

    def visit_FilterNotNullNode(self, node: FilterNotNullNode):
        """Generate code for filtering rows where column is not null"""
        code = f"{node.new_alias} = {node.source_alias}[{node.source_alias}['{node.column}'].notnull()]"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Filtered {{len({node.new_alias})}} rows where {node.column} is not null')")
        self.symbol_table[node.new_alias] = True

    def visit_FilterDuplicatesNode(self, node: FilterDuplicatesNode):
        """Generate code for filtering duplicate rows"""
        if node.subset:
            subset_str = str(node.subset)
            code = f"{node.new_alias} = {node.source_alias}[{node.source_alias}.duplicated(subset={subset_str}, keep='{node.keep}')]"
        else:
            code = f"{node.new_alias} = {node.source_alias}[{node.source_alias}.duplicated(keep='{node.keep}')]"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Filtered {{len({node.new_alias})}} duplicate rows from {node.source_alias}')")
        self.symbol_table[node.new_alias] = True

    def _build_params_str(self, params: dict) -> str:
        """
        Convert parameter dictionary to string for pandas function call.
        Handles proper formatting for strings, numbers, lists, dicts, etc.
        """
        if not params:
            return ""

        param_parts = []
        for key, value in params.items():
            formatted_value = self._format_value(value)
            param_parts.append(f"{key}={formatted_value}")

        return ", ".join(param_parts)

    def _format_value(self, value):
        """Format a value appropriately for Python code"""
        if value is None:
            return "None"
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, str):
            # Escape quotes in strings
            escaped = value.replace("'", "\\'")
            return f"'{escaped}'"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            # Format list elements
            formatted_items = [self._format_value(item) for item in value]
            return f"[{', '.join(formatted_items)}]"
        elif isinstance(value, dict):
            # Format dict items
            formatted_items = [f"{self._format_value(k)}: {self._format_value(v)}" for k, v in value.items()]
            return f"{{{', '.join(formatted_items)}}}"
        else:
            return str(value)

    def visit_SelectNode(self, node: SelectNode):
        columns_str = str(node.columns).replace("'", '"')
        code = f"{node.new_alias} = {node.source_alias}[{columns_str}].copy()"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Selected {{len({node.new_alias}.columns)}} columns from {node.source_alias}')")
    
    def visit_FilterNode(self, node: FilterNode):
        condition = node.condition

        # Build the filter expression
        left = f"{node.source_alias}['{condition.left_operand}']"

        # Handle right operand
        if isinstance(condition.right_operand, str):
            # Check if it's a column reference or a string literal
            if condition.right_operand in ['True', 'False', 'None']:
                right = condition.right_operand
            elif any(c in condition.right_operand for c in [' ', '.', '(', ')', '[', ']']):
                # Likely a string literal
                right = f"'{condition.right_operand}'"
            else:
                # Could be column reference - check if it exists
                right = f"'{condition.right_operand}'"
        else:
            right = str(condition.right_operand)

        filter_expr = f"{left} {condition.operator} {right}"
        code = f"{node.new_alias} = {node.source_alias}[{filter_expr}].copy()"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Filtered {node.source_alias}: {{len({node.new_alias})}} rows match condition')")

    def visit_UpdatedFilterNode(self, node):
        """
        UNIFIED SYNTAX v2.0: Generate code for rich where clause filtering
        Handles all filter modes: comparisons, between, in, contains, etc.
        """
        filter_expr = self._generate_condition_code(node.source_alias, node.condition)
        code = f"{node.new_alias} = {node.source_alias}[{filter_expr}].copy()"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Filtered {node.source_alias}: {{len({node.new_alias})}} rows match condition')")

    def _generate_condition_code(self, source_alias: str, condition) -> str:
        """Generate pandas filter expression from CompoundConditionNode"""
        from noeta_ast import (BinaryConditionNode, NotConditionNode, ComparisonNode,
                               BetweenNode, InNode, StringMatchNode, NullCheckNode)

        if isinstance(condition, BinaryConditionNode):
            # Binary condition: left and/or right
            left_expr = self._generate_condition_code(source_alias, condition.left)
            right_expr = self._generate_condition_code(source_alias, condition.right)
            op = '&' if condition.operator == 'and' else '|'
            return f"({left_expr}) {op} ({right_expr})"

        elif isinstance(condition, NotConditionNode):
            # Negation: ~condition
            inner_expr = self._generate_condition_code(source_alias, condition.condition)
            return f"~({inner_expr})"

        elif isinstance(condition, ComparisonNode):
            # Simple comparison: column op value
            left = f"{source_alias}['{condition.left}']"
            right = self._format_value_for_pandas(condition.right)
            return f"{left} {condition.operator} {right}"

        elif isinstance(condition, BetweenNode):
            # Between: column.between(min, max)
            column = f"{source_alias}['{condition.column}']"
            min_val = self._format_value_for_pandas(condition.min_value)
            max_val = self._format_value_for_pandas(condition.max_value)
            return f"{column}.between({min_val}, {max_val})"

        elif isinstance(condition, InNode):
            # In: column.isin(values)
            column = f"{source_alias}['{condition.column}']"
            values = [self._format_value_for_pandas(v) for v in condition.values]
            values_str = '[' + ', '.join(values) + ']'
            return f"{column}.isin({values_str})"

        elif isinstance(condition, StringMatchNode):
            # String matching: column.str.contains/startswith/endswith/match
            column = f"{source_alias}['{condition.column}']"
            pattern = self._format_value_for_pandas(condition.pattern)

            if condition.match_type == 'contains':
                return f"{column}.str.contains({pattern}, na=False)"
            elif condition.match_type == 'starts_with':
                return f"{column}.str.startswith({pattern}, na=False)"
            elif condition.match_type == 'ends_with':
                return f"{column}.str.endswith({pattern}, na=False)"
            elif condition.match_type == 'matches':
                return f"{column}.str.match({pattern}, na=False)"

        elif isinstance(condition, NullCheckNode):
            # Null check: column.isnull() or column.notnull()
            column = f"{source_alias}['{condition.column}']"
            if condition.is_not_null:
                return f"{column}.notnull()"
            else:
                return f"{column}.isnull()"

        else:
            raise ValueError(f"Unknown condition type: {type(condition)}")

    def _format_value_for_pandas(self, value):
        """Format a value for use in pandas expressions"""
        if isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, (int, float)):
            return str(value)
        elif value is None:
            return 'None'
        else:
            return repr(value)
    
    def visit_SortNode(self, node: SortNode):
        columns = [spec.column_name for spec in node.sort_specs]
        ascending = [spec.direction == 'ASC' for spec in node.sort_specs]

        code = f"{node.new_alias} = {node.source_alias}.sort_values("
        code += f"by={columns}, ascending={ascending}).copy()"
        self.code_lines.append(code)
        columns_str = ", ".join([f"'{col}'" for col in columns])
        self.code_lines.append(f"print(f\"Sorted {node.source_alias} by [{columns_str}]\")")
    
    def visit_JoinNode(self, node: JoinNode):
        code = f"{node.new_alias} = pd.merge({node.alias1}, {node.alias2}, "
        code += f"on='{node.join_column}', how='inner')"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Joined {node.alias1} and {node.alias2}: {{len({node.new_alias})}} rows')")
    
    def visit_GroupByNode(self, node: GroupByNode):
        if node.aggregations:
            # Classic syntax with aggregations
            agg_dict = {}
            for agg in node.aggregations:
                func = agg.function_name
                # Map common aggregation names
                func_map = {
                    'avg': 'mean',
                    'count': 'count',
                    'sum': 'sum',
                    'min': 'min',
                    'max': 'max',
                    'mean': 'mean',
                    'std': 'std'
                }
                func = func_map.get(func, func)

                if agg.column_name not in agg_dict:
                    agg_dict[agg.column_name] = []
                agg_dict[agg.column_name].append(func)

            code = f"{node.new_alias} = {node.source_alias}.groupby({node.group_columns}).agg("
            code += str(agg_dict) + ").reset_index()"

            # Flatten column names if needed
            code += f"\n{node.new_alias}.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in {node.new_alias}.columns]"

            self.code_lines.append(code)
            group_cols_str = ", ".join([f"'{col}'" for col in node.group_columns])
            self.code_lines.append(f"print(f\"Grouped by [{group_cols_str}]: {{len({node.new_alias})}} groups\")")
        else:
            # Natural syntax without aggregations - return grouped counts
            code = f"{node.new_alias} = {node.source_alias}.groupby({node.group_columns}).size().reset_index(name='count')"
            self.code_lines.append(code)
            group_cols_str = ", ".join([f"'{col}'" for col in node.group_columns])
            self.code_lines.append(f"print(f\"Grouped by [{group_cols_str}]: {{len({node.new_alias})}} groups\")")

    
    def visit_SampleNode(self, node: SampleNode):
        if node.new_alias:
            # Create alias
            if node.is_random:
                code = f"{node.new_alias} = {node.source_alias}.sample(n={node.sample_size}, random_state=42).copy()"
            else:
                code = f"{node.new_alias} = {node.source_alias}.head({node.sample_size}).copy()"
            self.code_lines.append(code)
            self.code_lines.append(f"print(f'Created alias {node.new_alias} with {node.sample_size} sampled rows')")
            self.symbol_table[node.new_alias] = True
        else:
            # Just display
            if node.is_random:
                sample_expr = f"{node.source_alias}.sample(n={node.sample_size}, random_state=42)"
            else:
                sample_expr = f"{node.source_alias}.head({node.sample_size})"
            self.code_lines.append(f"print(f'\\nSample of {node.sample_size} rows from {node.source_alias}:')")
            self.code_lines.append(f"print({sample_expr})")
    
    def visit_DropNANode(self, node: DropNANode):
        if node.columns:
            code = f"{node.new_alias} = {node.source_alias}.dropna(subset={node.columns}).copy()"
        else:
            code = f"{node.new_alias} = {node.source_alias}.dropna().copy()"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Dropped NA values: {{len({node.source_alias}) - len({node.new_alias})}} rows removed')")
    
    def visit_FillNANode(self, node: FillNANode):
        """
        UNIFIED SYNTAX v2.0: Consolidated fillna operation

        Handles both literal values and method-based filling:
        - fillna data column age with value=0 as filled
        - fillna data column age with method="mean" as filled
        - fillna data column age with method="median" as filled
        - fillna data column age with method="forward" as filled (ffill)
        - fillna data column age with method="backward" as filled (bfill)
        - fillna data column age with method="mode" as filled
        """
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"

        if node.method:
            # Method-based filling
            method_lower = node.method.lower()

            if method_lower == 'mean':
                fill_expr = f"{node.source_alias}['{node.column}'].mean()"
                code += f"{node.new_alias}['{node.column}'] = {node.new_alias}['{node.column}'].fillna({fill_expr})\n"
            elif method_lower == 'median':
                fill_expr = f"{node.source_alias}['{node.column}'].median()"
                code += f"{node.new_alias}['{node.column}'] = {node.new_alias}['{node.column}'].fillna({fill_expr})\n"
            elif method_lower == 'mode':
                code += f"_mode_value = {node.source_alias}['{node.column}'].mode()[0] if not {node.source_alias}['{node.column}'].mode().empty else None\n"
                code += f"{node.new_alias}['{node.column}'] = {node.new_alias}['{node.column}'].fillna(_mode_value)\n"
            elif method_lower in ['forward', 'ffill']:
                code += f"{node.new_alias}['{node.column}'] = {node.new_alias}['{node.column}'].ffill()\n"
            elif method_lower in ['backward', 'bfill']:
                code += f"{node.new_alias}['{node.column}'] = {node.new_alias}['{node.column}'].bfill()\n"
            else:
                raise ValueError(f"Unknown fill method: {node.method}")

            code += f"print(f'Filled NA values in {node.column} using method: {method_lower}')"
        elif node.fill_value is not None:
            # Literal value filling
            if isinstance(node.fill_value, str):
                code += f"{node.new_alias}['{node.column}'] = {node.new_alias}['{node.column}'].fillna('{node.fill_value}')\n"
            else:
                code += f"{node.new_alias}['{node.column}'] = {node.new_alias}['{node.column}'].fillna({node.fill_value})\n"
            code += f"print(f'Filled NA values in {node.column} with: {node.fill_value}')"
        else:
            raise ValueError("FillNA requires either fill_value or method")

        self.code_lines.append(code)
    
    def visit_MutateNode(self, node: MutateNode):
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        for mutation in node.mutations:
            # Replace column references in expression
            expr = mutation.expression
            # Simple replacement - in production, use proper expression parser
            code += f"{node.new_alias}['{mutation.new_column}'] = {node.new_alias}.eval('{expr}')\n"
        
        self.code_lines.append(code)
        self.code_lines.append(f"print('Added/modified {len(node.mutations)} columns')")
    
    def visit_ApplyNode(self, node: ApplyNode):
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        for col in node.columns:
            # Replace 'x' with the actual column reference
            expr = node.function_expr.replace('x', f'{node.new_alias}["{col}"]')
            code += f"{node.new_alias}['{col}_transformed'] = {expr}\n"
        
        self.code_lines.append(code)
        self.code_lines.append(f'print("Applied transformation to {node.columns}")')
    
    # Analysis visitors
    def visit_DescribeNode(self, node: DescribeNode):
        if node.columns:
            code = f'print("\\nDescriptive Statistics for {node.columns}:")\n'
            code += f"print({node.source_alias}[{node.columns}].describe())"
        else:
            code = f"print('\\nDescriptive Statistics for {node.source_alias}:')\n"
            code += f"print({node.source_alias}.describe())"
        
        self.code_lines.append(code)
    
    def visit_SummaryNode(self, node: SummaryNode):
        code = f"print('\\nDataset Summary for {node.source_alias}:')\n"
        code += f"print(f'Shape: {{{node.source_alias}.shape}}')\n"
        code += f"print(f'Columns: {{list({node.source_alias}.columns)}}')\n"
        code += f"print('\\nData types:')\n"
        code += f"print({node.source_alias}.dtypes)\n"
        code += f"print('\\nMissing values:')\n"
        code += f"print({node.source_alias}.isnull().sum())"
        
        self.code_lines.append(code)
    
    def visit_InfoNode(self, node: InfoNode):
        code = f"print('\\nDataset Info for {node.source_alias}:')\n"
        code += f"{node.source_alias}.info()"

        self.code_lines.append(code)

    def visit_UniqueNode(self, node):
        """Generate: print(df['column'].unique())"""
        code = f"unique_vals = {node.source_alias}['{node.column}'].unique()\n"
        code += f"print(f'\\nUnique values in column {node.column}:')\n"
        code += f"print(unique_vals)\n"
        code += f"print(f'Count: {{len(unique_vals)}}')"
        self.code_lines.append(code)

    def visit_ValueCountsNode(self, node):
        """Generate: print(df['column'].value_counts())"""
        code = f"counts = {node.source_alias}['{node.column}'].value_counts("
        code += f"normalize={node.normalize}, ascending={node.ascending})\n"
        code += f"print(f'\\nValue counts for column {node.column}:')\n"
        code += "print(counts)"
        self.code_lines.append(code)

    def visit_OutliersNode(self, node: OutliersNode):
        code = f"# Detect outliers using {node.method} method\n"
        
        if node.method == 'iqr':
            code += f"for col in {node.columns}:\n"
            code += f"    Q1 = {node.source_alias}[col].quantile(0.25)\n"
            code += f"    Q3 = {node.source_alias}[col].quantile(0.75)\n"
            code += f"    IQR = Q3 - Q1\n"
            code += f"    lower_bound = Q1 - 1.5 * IQR\n"
            code += f"    upper_bound = Q3 + 1.5 * IQR\n"
            code += f"    outliers = {node.source_alias}[(({node.source_alias}[col] < lower_bound) | ({node.source_alias}[col] > upper_bound))]\n"
            code += f"    print(f'Outliers in {{col}}: {{len(outliers)}} rows')\n"
        elif node.method == 'zscore':
            code += f"from scipy import stats\n"
            code += f"for col in {node.columns}:\n"
            code += f"    z_scores = np.abs(stats.zscore({node.source_alias}[col].dropna()))\n"
            code += f"    outliers = np.sum(z_scores > 3)\n"
            code += f"    print(f'Outliers in {{col}} (|z| > 3): {{outliers}} values')\n"
        
        self.code_lines.append(code)
    
    def visit_QuantileNode(self, node: QuantileNode):
        code = f"quantile_val = {node.source_alias}['{node.column}'].quantile({node.quantile_value})\n"
        code += f"print(f'{node.quantile_value} quantile of {node.column}: {{quantile_val:.4f}}')"
        
        self.code_lines.append(code)
    
    def visit_NormalizeNode(self, node: NormalizeNode):
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        
        if node.method == 'zscore':
            code += f"from sklearn.preprocessing import StandardScaler\n"
            code += f"scaler = StandardScaler()\n"
            code += f"{node.new_alias}[{node.columns}] = scaler.fit_transform({node.new_alias}[{node.columns}])"
        elif node.method == 'minmax':
            code += f"from sklearn.preprocessing import MinMaxScaler\n"
            code += f"scaler = MinMaxScaler()\n"
            code += f"{node.new_alias}[{node.columns}] = scaler.fit_transform({node.new_alias}[{node.columns}])"
        
        self.code_lines.append(code)
        self.code_lines.append(f'print("Normalized columns {node.columns} using {node.method}")')
    
    def visit_BinningNode(self, node: BinningNode):
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_binned'] = pd.cut({node.new_alias}['{node.column}'], bins={node.num_bins})"
        
        self.code_lines.append(code)
        self.code_lines.append(f"print('Created {node.num_bins} bins for column {node.column}')")
    
    def visit_RollingNode(self, node: RollingNode):
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        
        func_map = {
            'mean': 'mean',
            'sum': 'sum',
            'min': 'min',
            'max': 'max',
            'std': 'std',
            'count': 'count'
        }
        func = func_map.get(node.function_name, 'mean')
        
        code += f"{node.new_alias}['{node.column}_rolling_{node.function_name}'] = "
        code += f"{node.new_alias}['{node.column}'].rolling(window={node.window_size}).{func}()"
        
        self.code_lines.append(code)
        self.code_lines.append(f"print('Applied rolling {node.function_name} with window {node.window_size}')")
    
    def visit_HypothesisNode(self, node: HypothesisNode):
        code = "# Hypothesis testing\n"
        
        if node.test_type in ['ttest', 'ttest_ind']:
            code += f"from scipy import stats\n"
            for col in node.columns:
                code += f"t_stat, p_value = stats.ttest_ind({node.alias1}['{col}'], {node.alias2}['{col}'])\n"
                code += f"print(f'T-test for {{col}}: t-statistic={{t_stat:.4f}}, p-value={{p_value:.4f}}')\n"
        elif node.test_type == 'chi2':
            code += f"from scipy import stats\n"
            code += f"# Implement chi-square test\n"
            code += f"print('Chi-square test to be implemented')\n"
        
        self.code_lines.append(code)
    
    # Visualization visitors
    def visit_BoxPlotNode(self, node: BoxPlotNode):
        code = f"# Box plot\n"
        code += f"plt.figure(figsize=(10, 6))\n"

        if node.columns:
            # Classic syntax: multiple columns
            code += f"{node.source_alias}[{node.columns}].boxplot()\n"
        elif node.value_column and node.group_column:
            # Natural syntax: value by group
            code += f"{node.source_alias}.boxplot(column='{node.value_column}', by='{node.group_column}')\n"
        elif node.value_column:
            # Natural syntax: single value column
            code += f"{node.source_alias}[['{node.value_column}']].boxplot()\n"
        else:
            raise ValueError("BoxPlot requires either columns or value_column")

        code += f"plt.title('Box Plot')\n"
        code += f"plt.xticks(rotation=45)\n"

        self.code_lines.append(code)
        self.last_plot = True
    
    def visit_HeatmapNode(self, node: HeatmapNode):
        code = f"# Heatmap\n"
        code += f"plt.figure(figsize=(10, 8))\n"
        code += f"correlation_matrix = {node.source_alias}[{node.columns}].corr()\n"
        code += f"sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)\n"
        code += f"plt.title('Correlation Heatmap')\n"
        
        self.code_lines.append(code)
        self.last_plot = True
    
    def visit_PairPlotNode(self, node: PairPlotNode):
        code = f"# Pair plot\n"
        code += f"pairplot_fig = sns.pairplot({node.source_alias}[{node.columns}])\n"
        code += f"pairplot_fig.fig.suptitle('Pair Plot', y=1.02)\n"
        
        self.code_lines.append(code)
        self.last_plot = True
    
    def visit_TimeSeriesNode(self, node: TimeSeriesNode):
        code = f"# Time series plot\n"
        code += f"plt.figure(figsize=(12, 6))\n"
        code += f"plt.plot({node.source_alias}['{node.x_column}'], {node.source_alias}['{node.y_column}'])\n"
        code += f"plt.xlabel('{node.x_column}')\n"
        code += f"plt.ylabel('{node.y_column}')\n"
        code += f"plt.title('Time Series Plot')\n"
        code += f"plt.xticks(rotation=45)\n"
        code += f"plt.grid(True, alpha=0.3)\n"
        
        self.code_lines.append(code)
        self.last_plot = True
    
    def visit_PieChartNode(self, node: PieChartNode):
        code = f"# Pie chart\n"
        code += f"plt.figure(figsize=(8, 8))\n"
        code += f"plt.pie({node.source_alias}['{node.values_column}'], labels={node.source_alias}['{node.labels_column}'], autopct='%1.1f%%')\n"
        code += f"plt.title('Pie Chart')\n"
        
        self.code_lines.append(code)
        self.last_plot = True
    
    # File operation visitors
    def visit_SaveNode(self, node: SaveNode):
        """
        UNIFIED SYNTAX v2.0: Generate code for consolidated save operation

        Handles all formats (csv, json, excel, parquet) with optional parameters.
        Format is auto-detected from file extension or explicitly specified.
        """
        format_type = node.format
        filepath = node.filepath
        params = node.params or {}

        # Build parameters string, with defaults
        if format_type == 'csv':
            # Default CSV params
            default_params = {'index': False}
            default_params.update(params)
            params_str = self._build_params_str(default_params)
            code = f"{node.source_alias}.to_csv('{filepath}', {params_str})\n"

        elif format_type == 'json':
            # Default JSON params
            default_params = {'orient': 'records', 'indent': 2}
            default_params.update(params)
            params_str = self._build_params_str(default_params)
            code = f"{node.source_alias}.to_json('{filepath}', {params_str})\n"

        elif format_type == 'excel':
            # Default Excel params
            default_params = {'index': False}
            default_params.update(params)
            params_str = self._build_params_str(default_params)
            code = f"{node.source_alias}.to_excel('{filepath}', {params_str})\n"

        elif format_type == 'parquet':
            # Default Parquet params
            default_params = {'index': False}
            default_params.update(params)
            params_str = self._build_params_str(default_params)
            code = f"{node.source_alias}.to_parquet('{filepath}', {params_str})\n"

        else:
            # Fallback to CSV
            default_params = {'index': False}
            default_params.update(params)
            params_str = self._build_params_str(default_params)
            code = f"{node.source_alias}.to_csv('{filepath}', {params_str})\n"

        code += f"print(f'Saved {node.source_alias} to {filepath}')"
        self.code_lines.append(code)
    
    def visit_ExportPlotNode(self, node: ExportPlotNode):
        code = f"# Export plot\n"
        if node.width and node.height:
            code += f"plt.gcf().set_size_inches({node.width}/100, {node.height}/100)\n"
        code += f"plt.savefig('{node.file_name}', dpi=100, bbox_inches='tight')\n"
        code += f"print(f'Exported plot to {node.file_name}')"

        self.code_lines.append(code)

    # ============================================================
    # PHASE 4: TRANSFORMATION OPERATIONS - CODE GENERATORS
    # ============================================================

    # Phase 4A: Math Operations
    def visit_RoundNode(self, node):
        """Generate: df['col_round'] = df['col'].round(decimals)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}'] = {node.source_alias}['{node.column}'].round({node.decimals})\n"
        code += f"print(f'Rounded {node.column} to {node.decimals} decimals')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_AbsNode(self, node):
        """Generate: df['col_abs'] = df['col'].abs()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}'] = {node.source_alias}['{node.column}'].abs()\n"
        code += f"print(f'Computed absolute value for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_SqrtNode(self, node):
        """Generate: df['col_sqrt'] = np.sqrt(df['col'])"""
        self.imports.add("import numpy as np")
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}'] = np.sqrt({node.source_alias}['{node.column}'])\n"
        code += f"print(f'Computed square root for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_PowerNode(self, node):
        """Generate: df['col_pow'] = np.power(df['col'], exp)"""
        self.imports.add("import numpy as np")
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}'] = np.power({node.source_alias}['{node.column}'], {node.exponent})\n"
        code += f"print(f'Raised {node.column} to power {node.exponent}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_LogNode(self, node):
        """Generate: df['col_log'] = np.log(df['col']) or np.log10(df['col'])"""
        self.imports.add("import numpy as np")
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        if node.base == "e":
            code += f"{node.new_alias}['{node.column}'] = np.log({node.source_alias}['{node.column}'])\n"
        elif node.base == "10":
            code += f"{node.new_alias}['{node.column}'] = np.log10({node.source_alias}['{node.column}'])\n"
        else:
            code += f"{node.new_alias}['{node.column}'] = np.log({node.source_alias}['{node.column}']) / np.log({node.base})\n"
        code += f"print(f'Computed log (base {node.base}) for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_CeilNode(self, node):
        """Generate: df['col_ceil'] = np.ceil(df['col'])"""
        self.imports.add("import numpy as np")
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}'] = np.ceil({node.source_alias}['{node.column}'])\n"
        code += f"print(f'Computed ceiling for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_FloorNode(self, node):
        """Generate: df['col_floor'] = np.floor(df['col'])"""
        self.imports.add("import numpy as np")
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}'] = np.floor({node.source_alias}['{node.column}'])\n"
        code += f"print(f'Computed floor for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # Phase 4B: String Operations
    def visit_UpperNode(self, node):
        """Generate: df['col_upper'] = df['col'].str.upper()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}'] = {node.source_alias}['{node.column}'].str.upper()\n"
        code += f"print(f'Converted {node.column} to uppercase')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_LowerNode(self, node):
        """Generate: df['col_lower'] = df['col'].str.lower()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}'] = {node.source_alias}['{node.column}'].str.lower()\n"
        code += f"print(f'Converted {node.column} to lowercase')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_StripNode(self, node):
        """Generate: df['col_stripped'] = df['col'].str.strip()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}'] = {node.source_alias}['{node.column}'].str.strip()\n"
        code += f"print(f'Stripped whitespace from {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ReplaceNode(self, node):
        """Generate: df['col'] = df['col'].str.replace(old, new)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}'] = {node.source_alias}['{node.column}'].str.replace({self._format_value(node.old)}, {self._format_value(node.new)})\n"
        code += f"print(f'Replaced values in {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_SplitNode(self, node):
        """Generate: df[['col1', 'col2']] = df['col'].str.split(delimiter, expand=True)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"split_result = {node.source_alias}['{node.column}'].str.split({self._format_value(node.delimiter)}, expand=True)\n"
        code += f"{node.new_alias} = pd.concat([{node.source_alias}, split_result], axis=1)\n"
        code += f"print(f'Split {node.column} by delimiter')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ConcatNode(self, node):
        """Generate: df['col'] = df['col1'] + sep + df['col2']"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        # Build concatenation expression
        concat_expr = " + ".join([f"{node.source_alias}['{col}'].astype(str)" for col in node.columns])
        if node.separator:
            sep_additions = [f"{self._format_value(node.separator)}"] * (len(node.columns) - 1)
            concat_parts = []
            for i, col in enumerate(node.columns):
                concat_parts.append(f"{node.source_alias}['{col}'].astype(str)")
                if i < len(node.columns) - 1:
                    concat_parts.append(self._format_value(node.separator))
            concat_expr = " + ".join(concat_parts)
        code += f"{node.new_alias}['concatenated'] = {concat_expr}\n"
        code += f"print(f'Concatenated {len(node.columns)} columns')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_SubstringNode(self, node):
        """Generate: df['col_sub'] = df['col'].str[start:end]"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        if node.end:
            code += f"{node.new_alias}['{node.column}'] = {node.source_alias}['{node.column}'].str[{node.start}:{node.end}]\n"
        else:
            code += f"{node.new_alias}['{node.column}'] = {node.source_alias}['{node.column}'].str[{node.start}:]\n"
        code += f"print(f'Extracted substring from {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_LengthNode(self, node):
        """Generate: df['col_len'] = df['col'].str.len()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_length'] = {node.source_alias}['{node.column}'].str.len()\n"
        code += f"print(f'Computed length for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # Phase 4C: Date Operations
    def visit_ParseDatetimeNode(self, node):
        """Generate: df['col'] = pd.to_datetime(df['col'], format=...)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        if node.format:
            code += f"{node.new_alias}['{node.column}'] = pd.to_datetime({node.source_alias}['{node.column}'], format={self._format_value(node.format)})\n"
        else:
            code += f"{node.new_alias}['{node.column}'] = pd.to_datetime({node.source_alias}['{node.column}'])\n"
        code += f"print(f'Parsed {node.column} as datetime')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ExtractYearNode(self, node):
        """Generate: df['year'] = df['col'].dt.year"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_year'] = {node.source_alias}['{node.column}'].dt.year\n"
        code += f"print(f'Extracted year from {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ExtractMonthNode(self, node):
        """Generate: df['month'] = df['col'].dt.month"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_month'] = {node.source_alias}['{node.column}'].dt.month\n"
        code += f"print(f'Extracted month from {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ExtractDayNode(self, node):
        """Generate: df['day'] = df['col'].dt.day"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_day'] = {node.source_alias}['{node.column}'].dt.day\n"
        code += f"print(f'Extracted day from {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ExtractNode(self, node):
        """
        UNIFIED SYNTAX v2.0: Consolidated date extraction operation

        Generates code for extracting any date/time component using pandas dt accessor.
        Supported parts: year, month, day, hour, minute, second, dayofweek,
                        dayofyear, weekofyear, quarter

        Example: extract data column timestamp with part="year" as year
        Generates: year = data.copy()
                  year['timestamp_year'] = data['timestamp'].dt.year
        """
        # Map part names to pandas dt properties
        part_map = {
            'year': 'year',
            'month': 'month',
            'day': 'day',
            'hour': 'hour',
            'minute': 'minute',
            'second': 'second',
            'dayofweek': 'dayofweek',
            'dayofyear': 'dayofyear',
            'weekofyear': 'isocalendar().week',  # Changed in pandas 1.1+
            'week': 'isocalendar().week',
            'quarter': 'quarter'
        }

        part_lower = node.part.lower()
        dt_property = part_map.get(part_lower, part_lower)

        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_{part_lower}'] = {node.source_alias}['{node.column}'].dt.{dt_property}\n"
        code += f"print(f'Extracted {part_lower} from {node.column}')"

        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_DateDiffNode(self, node):
        """Generate: df['diff'] = (df['end'] - df['start']).dt.days"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        if node.unit == "days":
            code += f"{node.new_alias}['date_diff'] = ({node.source_alias}['{node.end_column}'] - {node.source_alias}['{node.start_column}']).dt.days\n"
        elif node.unit == "hours":
            code += f"{node.new_alias}['date_diff'] = ({node.source_alias}['{node.end_column}'] - {node.source_alias}['{node.start_column}']).dt.total_seconds() / 3600\n"
        elif node.unit == "minutes":
            code += f"{node.new_alias}['date_diff'] = ({node.source_alias}['{node.end_column}'] - {node.source_alias}['{node.start_column}']).dt.total_seconds() / 60\n"
        else:
            code += f"{node.new_alias}['date_diff'] = ({node.source_alias}['{node.end_column}'] - {node.source_alias}['{node.start_column}']).dt.days\n"
        code += f"print(f'Computed date difference in {node.unit}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # Phase 4D: Type Operations
    def visit_AsTypeNode(self, node):
        """Generate: df['col'] = df['col'].astype(dtype)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}'] = {node.source_alias}['{node.column}'].astype({self._format_value(node.dtype)})\n"
        code += f"print(f'Converted {node.column} to {node.dtype}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ToNumericNode(self, node):
        """Generate: df['col'] = pd.to_numeric(df['col'], errors=...)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}'] = pd.to_numeric({node.source_alias}['{node.column}'], errors={self._format_value(node.errors)})\n"
        code += f"print(f'Converted {node.column} to numeric (errors={node.errors})')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # Phase 4E: Encoding Operations
    def visit_OneHotEncodeNode(self, node):
        """Generate: df = pd.get_dummies(df, columns=[col])"""
        code = f"{node.new_alias} = pd.get_dummies({node.source_alias}, columns=['{node.column}'])\n"
        code += f"print(f'One-hot encoded {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_LabelEncodeNode(self, node):
        """Generate: df['col'] = LabelEncoder().fit_transform(df['col'])"""
        self.imports.add("from sklearn.preprocessing import LabelEncoder")
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"le = LabelEncoder()\n"
        code += f"{node.new_alias}['{node.column}'] = le.fit_transform({node.source_alias}['{node.column}'])\n"
        code += f"print(f'Label encoded {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # Phase 4F: Scaling Operations
    def visit_StandardScaleNode(self, node):
        """Generate: df['col'] = StandardScaler().fit_transform(df[['col']])"""
        self.imports.add("from sklearn.preprocessing import StandardScaler")
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"scaler = StandardScaler()\n"
        code += f"{node.new_alias}['{node.column}'] = scaler.fit_transform({node.source_alias}[['{node.column}']])\n"
        code += f"print(f'Standard scaled {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_MinMaxScaleNode(self, node):
        """Generate: df['col'] = MinMaxScaler().fit_transform(df[['col']])"""
        self.imports.add("from sklearn.preprocessing import MinMaxScaler")
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"scaler = MinMaxScaler()\n"
        code += f"{node.new_alias}['{node.column}'] = scaler.fit_transform({node.source_alias}[['{node.column}']])\n"
        code += f"print(f'Min-Max scaled {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # ============================================================
    # PHASE 5: CLEANING OPERATIONS - CODE GENERATORS
    # ============================================================

    def visit_IsNullNode(self, node):
        """Generate: df['col_null'] = df['col'].isnull()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_isnull'] = {node.source_alias}['{node.column}'].isnull()\n"
        code += f"print(f'Created null mask for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_NotNullNode(self, node):
        """Generate: df['col_notnull'] = df['col'].notnull()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_notnull'] = {node.source_alias}['{node.column}'].notnull()\n"
        code += f"print(f'Created not-null mask for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_CountNANode(self, node):
        """Generate: print(df.isnull().sum())"""
        code = f"print('Missing values count:')\n"
        code += f"print({node.source_alias}.isnull().sum())"
        self.code_lines.append(code)

    def visit_FillForwardNode(self, node):
        """Generate: df = df.ffill() or df['col'] = df['col'].ffill()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        if node.column:
            code += f"{node.new_alias}['{node.column}'] = {node.source_alias}['{node.column}'].ffill()\n"
            code += f"print(f'Forward filled {node.column}')"
        else:
            code += f"{node.new_alias} = {node.source_alias}.ffill()\n"
            code += f"print(f'Forward filled all columns')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_FillBackwardNode(self, node):
        """Generate: df = df.bfill() or df['col'] = df['col'].bfill()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        if node.column:
            code += f"{node.new_alias}['{node.column}'] = {node.source_alias}['{node.column}'].bfill()\n"
            code += f"print(f'Backward filled {node.column}')"
        else:
            code += f"{node.new_alias} = {node.source_alias}.bfill()\n"
            code += f"print(f'Backward filled all columns')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_FillMeanNode(self, node):
        """Generate: df['col'] = df['col'].fillna(df['col'].mean())"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}'] = {node.source_alias}['{node.column}'].fillna({node.source_alias}['{node.column}'].mean())\n"
        code += f"print(f'Filled {node.column} with mean value')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_FillMedianNode(self, node):
        """Generate: df['col'] = df['col'].fillna(df['col'].median())"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}'] = {node.source_alias}['{node.column}'].fillna({node.source_alias}['{node.column}'].median())\n"
        code += f"print(f'Filled {node.column} with median value')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_InterpolateNode(self, node):
        """Generate: df = df.interpolate(method=...) or df['col'] = df['col'].interpolate(method=...)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        if node.column:
            code += f"{node.new_alias}['{node.column}'] = {node.source_alias}['{node.column}'].interpolate(method={self._format_value(node.method)})\n"
            code += f"print(f'Interpolated {node.column} using {node.method} method')"
        else:
            code += f"{node.new_alias} = {node.source_alias}.interpolate(method={self._format_value(node.method)})\n"
            code += f"print(f'Interpolated all columns using {node.method} method')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_DuplicatedNode(self, node):
        """Generate: df['is_dup'] = df.duplicated(subset=[...], keep=...)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        if node.columns:
            code += f"{node.new_alias}['is_duplicate'] = {node.source_alias}.duplicated(subset={node.columns}, keep={self._format_value(node.keep)})\n"
        else:
            code += f"{node.new_alias}['is_duplicate'] = {node.source_alias}.duplicated(keep={self._format_value(node.keep)})\n"
        code += f"print(f'Marked duplicate rows')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_CountDuplicatesNode(self, node):
        """Generate: print(df.duplicated(subset=[...]).sum())"""
        code = f"print('Duplicate count:')\n"
        if node.columns:
            code += f"print({node.source_alias}.duplicated(subset={node.columns}).sum())"
        else:
            code += f"print({node.source_alias}.duplicated().sum())"
        self.code_lines.append(code)

    def visit_DropDuplicatesNode(self, node):
        """Generate: df = df.drop_duplicates(subset=[...], keep='first')"""
        if node.subset:
            code = f"{node.new_alias} = {node.source_alias}.drop_duplicates(subset={node.subset}, keep='{node.keep}').reset_index(drop=True)\n"
        else:
            code = f"{node.new_alias} = {node.source_alias}.drop_duplicates(keep='{node.keep}').reset_index(drop=True)\n"
        code += f"print(f'Removed duplicates: {{len({node.source_alias}) - len({node.new_alias})}} rows removed')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_FillModeNode(self, node):
        """Generate: df['col'] = df['col'].fillna(df['col'].mode()[0])"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}'] = {node.source_alias}['{node.column}'].fillna({node.source_alias}['{node.column}'].mode()[0])\n"
        code += f"print(f'Filled {node.column} with mode value')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_QcutNode(self, node):
        """Generate: df['col_qcut'] = pd.qcut(df['col'], q=4, labels=[...])"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        if node.labels:
            code += f"{node.new_alias}['{node.column}_qcut'] = pd.qcut({node.source_alias}['{node.column}'], q={node.q}, labels={node.labels})\n"
        else:
            code += f"{node.new_alias}['{node.column}_qcut'] = pd.qcut({node.source_alias}['{node.column}'], q={node.q})\n"
        code += f"print(f'Created {node.q} quantile bins for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # ============================================================
    # PHASE 6: DATA ORDERING OPERATIONS - CODE GENERATORS
    # ============================================================

    def visit_SortIndexNode(self, node):
        """Generate: df = df.sort_index(ascending=True)"""
        code = f"{node.new_alias} = {node.source_alias}.sort_index(ascending={node.ascending})\n"
        code += f"print(f'Sorted index {'ascending' if node.ascending else 'descending'}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_RankNode(self, node):
        """Generate: df['col_rank'] = df['col'].rank(method='average', ascending=True, pct=False)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_rank'] = {node.source_alias}['{node.column}'].rank(method='{node.method}', ascending={node.ascending}, pct={node.pct})\n"
        code += f"print(f'Ranked {node.column} using {node.method} method')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # ============================================================
    # PHASE 7: AGGREGATION & GROUPING OPERATIONS - CODE GENERATORS
    # ============================================================

    def visit_FilterGroupsNode(self, node):
        """Generate: df = df.groupby(...).filter(lambda x: condition)"""
        # Parse the condition to create a proper lambda
        condition = node.condition
        # Simple parsing: replace aggregate functions with group references
        condition_code = condition.replace("count", "len(x)").replace("sum", "x.sum()").replace("mean", "x.mean()")
        
        code = f"{node.new_alias} = {node.source_alias}.groupby({node.group_columns}).filter(lambda x: {condition_code})\n"
        code += f"print(f'Filtered groups where {node.condition}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_GroupTransformNode(self, node):
        """Generate: df['col_transformed'] = df.groupby(...)['col'].transform('mean')"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_{node.function}'] = {node.source_alias}.groupby({node.group_columns})['{node.column}'].transform('{node.function}')\n"
        code += f"print(f'Applied {node.function} transform to {node.column} within groups')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_WindowRankNode(self, node):
        """Generate: df['col_rank'] = df.groupby(...)['col'].rank(...)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        
        if node.partition_by:
            code += f"{node.new_alias}['{node.column}_rank'] = {node.source_alias}.groupby({node.partition_by})['{node.column}'].rank(method='{node.method}', ascending={node.ascending})\n"
        else:
            code += f"{node.new_alias}['{node.column}_rank'] = {node.source_alias}['{node.column}'].rank(method='{node.method}', ascending={node.ascending})\n"
        
        code += f"print(f'Computed window rank for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_WindowLagNode(self, node):
        """Generate: df['col_lag'] = df.groupby(...)['col'].shift(n)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        
        if node.partition_by:
            code += f"{node.new_alias}['{node.column}_lag{node.periods}'] = {node.source_alias}.groupby({node.partition_by})['{node.column}'].shift({node.periods})\n"
        else:
            code += f"{node.new_alias}['{node.column}_lag{node.periods}'] = {node.source_alias}['{node.column}'].shift({node.periods})\n"
        
        if node.fill_value is not None:
            code += f"{node.new_alias}['{node.column}_lag{node.periods}'] = {node.new_alias}['{node.column}_lag{node.periods}'].fillna({self._format_value(node.fill_value)})\n"
        
        code += f"print(f'Created lag of {node.periods} periods for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_WindowLeadNode(self, node):
        """Generate: df['col_lead'] = df.groupby(...)['col'].shift(-n)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        
        if node.partition_by:
            code += f"{node.new_alias}['{node.column}_lead{node.periods}'] = {node.source_alias}.groupby({node.partition_by})['{node.column}'].shift(-{node.periods})\n"
        else:
            code += f"{node.new_alias}['{node.column}_lead{node.periods}'] = {node.source_alias}['{node.column}'].shift(-{node.periods})\n"
        
        if node.fill_value is not None:
            code += f"{node.new_alias}['{node.column}_lead{node.periods}'] = {node.new_alias}['{node.column}_lead{node.periods}'].fillna({self._format_value(node.fill_value)})\n"
        
        code += f"print(f'Created lead of {node.periods} periods for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_RollingMeanNode(self, node):
        """Generate: df['col_rolling_mean'] = df['col'].rolling(window=n).mean()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_rolling_mean'] = {node.source_alias}['{node.column}'].rolling(window={node.window}, min_periods={node.min_periods}).mean()\n"
        code += f"print(f'Computed rolling mean with window {node.window} for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_RollingSumNode(self, node):
        """Generate: df['col_rolling_sum'] = df['col'].rolling(window=n).sum()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_rolling_sum'] = {node.source_alias}['{node.column}'].rolling(window={node.window}, min_periods={node.min_periods}).sum()\n"
        code += f"print(f'Computed rolling sum with window {node.window} for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_RollingStdNode(self, node):
        """Generate: df['col_rolling_std'] = df['col'].rolling(window=n).std()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_rolling_std'] = {node.source_alias}['{node.column}'].rolling(window={node.window}, min_periods={node.min_periods}).std()\n"
        code += f"print(f'Computed rolling std with window {node.window} for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_RollingMinNode(self, node):
        """Generate: df['col_rolling_min'] = df['col'].rolling(window=n).min()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_rolling_min'] = {node.source_alias}['{node.column}'].rolling(window={node.window}, min_periods={node.min_periods}).min()\n"
        code += f"print(f'Computed rolling min with window {node.window} for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_RollingMaxNode(self, node):
        """Generate: df['col_rolling_max'] = df['col'].rolling(window=n).max()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_rolling_max'] = {node.source_alias}['{node.column}'].rolling(window={node.window}, min_periods={node.min_periods}).max()\n"
        code += f"print(f'Computed rolling max with window {node.window} for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ExpandingMeanNode(self, node):
        """Generate: df['col_expanding_mean'] = df['col'].expanding(min_periods=n).mean()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_expanding_mean'] = {node.source_alias}['{node.column}'].expanding(min_periods={node.min_periods}).mean()\n"
        code += f"print(f'Computed expanding mean for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ExpandingSumNode(self, node):
        """Generate: df['col_expanding_sum'] = df['col'].expanding(min_periods=n).sum()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_expanding_sum'] = {node.source_alias}['{node.column}'].expanding(min_periods={node.min_periods}).sum()\n"
        code += f"print(f'Computed expanding sum for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ExpandingMinNode(self, node):
        """Generate: df['col_expanding_min'] = df['col'].expanding(min_periods=n).min()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_expanding_min'] = {node.source_alias}['{node.column}'].expanding(min_periods={node.min_periods}).min()\n"
        code += f"print(f'Computed expanding min for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ExpandingMaxNode(self, node):
        """Generate: df['col_expanding_max'] = df['col'].expanding(min_periods=n).max()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_expanding_max'] = {node.source_alias}['{node.column}'].expanding(min_periods={node.min_periods}).max()\n"
        code += f"print(f'Computed expanding max for {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # ============================================================
    # PHASE 8: DATA RESHAPING OPERATIONS - CODE GENERATORS
    # ============================================================

    def visit_PivotNode(self, node):
        """Generate: df = df.pivot(index='idx', columns='col', values='val')"""
        code = f"{node.new_alias} = {node.source_alias}.pivot(index='{node.index}', columns='{node.columns}', values='{node.values}')\n"
        code += f"print(f'Pivoted data with {node.index} as index, {node.columns} as columns')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_PivotTableNode(self, node):
        """Generate: df = pd.pivot_table(df, index='idx', columns='col', values='val', aggfunc='mean')"""
        fill_val = f", fill_value={self._format_value(node.fill_value)}" if node.fill_value is not None else ""
        code = f"{node.new_alias} = pd.pivot_table({node.source_alias}, index='{node.index}', columns='{node.columns}', values='{node.values}', aggfunc='{node.aggfunc}'{fill_val})\n"
        code += f"print(f'Created pivot table with {node.aggfunc} aggregation')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_MeltNode(self, node):
        """Generate: df = pd.melt(df, id_vars=['id'], value_vars=['v1'], var_name='var', value_name='val')"""
        value_vars_str = f", value_vars={node.value_vars}" if node.value_vars else ""
        code = f"{node.new_alias} = pd.melt({node.source_alias}, id_vars={node.id_vars}{value_vars_str}, var_name='{node.var_name}', value_name='{node.value_name}')\n"
        code += f"print(f'Melted data with {len(node.id_vars)} id columns')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_StackNode(self, node):
        """Generate: df = df.stack(level=-1)"""
        code = f"{node.new_alias} = {node.source_alias}.stack(level={node.level})\n"
        code += f"print(f'Stacked data at level {node.level}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_UnstackNode(self, node):
        """Generate: df = df.unstack(level=-1, fill_value=None)"""
        fill_val = f", fill_value={self._format_value(node.fill_value)}" if node.fill_value is not None else ""
        code = f"{node.new_alias} = {node.source_alias}.unstack(level={node.level}{fill_val})\n"
        code += f"print(f'Unstacked data at level {node.level}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_TransposeNode(self, node):
        """Generate: df = df.T"""
        code = f"{node.new_alias} = {node.source_alias}.T.copy()\n"
        code += f"print(f'Transposed data')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_CrosstabNode(self, node):
        """Generate: df = pd.crosstab(df['row'], df['col'], values=df['val'], aggfunc='count')"""
        if node.values:
            code = f"{node.new_alias} = pd.crosstab({node.source_alias}['{node.row_column}'], {node.source_alias}['{node.col_column}'], values={node.source_alias}['{node.values}'], aggfunc='{node.aggfunc}')\n"
        else:
            code = f"{node.new_alias} = pd.crosstab({node.source_alias}['{node.row_column}'], {node.source_alias}['{node.col_column}'])\n"
        code += f"print(f'Created crosstab of {node.row_column} by {node.col_column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # ============================================================
    # PHASE 9: DATA COMBINING OPERATIONS - CODE GENERATORS
    # ============================================================

    def visit_MergeNode(self, node):
        """Generate: df = pd.merge(left, right, on='col', how='inner')"""
        on_clause = ""
        if node.on:
            on_clause = f"on='{node.on}'"
        elif node.left_on and node.right_on:
            on_clause = f"left_on='{node.left_on}', right_on='{node.right_on}'"
        
        suffixes_str = f", suffixes={node.suffixes}" if node.suffixes != ("_x", "_y") else ""
        
        code = f"{node.new_alias} = pd.merge({node.left_alias}, {node.right_alias}, {on_clause}, how='{node.how}'{suffixes_str})\n"
        code += f"print(f'Merged data using {node.how} join: {{len({node.new_alias})}} rows')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ConcatVerticalNode(self, node):
        """Generate: df = pd.concat([df1, df2], axis=0, ignore_index=True)"""
        sources_str = "[" + ", ".join(node.sources) + "]"
        code = f"{node.new_alias} = pd.concat({sources_str}, axis=0, ignore_index={node.ignore_index})\n"
        code += f"print(f'Concatenated {len(node.sources)} dataframes vertically: {{len({node.new_alias})}} rows')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ConcatHorizontalNode(self, node):
        """Generate: df = pd.concat([df1, df2], axis=1)"""
        sources_str = "[" + ", ".join(node.sources) + "]"
        code = f"{node.new_alias} = pd.concat({sources_str}, axis=1, ignore_index={node.ignore_index})\n"
        code += f"print(f'Concatenated {len(node.sources)} dataframes horizontally: {{len({node.new_alias}.columns)}} columns')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_UnionNode(self, node):
        """Generate: df = pd.concat([df1, df2]).drop_duplicates()"""
        code = f"{node.new_alias} = pd.concat([{node.left_alias}, {node.right_alias}]).drop_duplicates().reset_index(drop=True)\n"
        code += f"print(f'Union of {node.left_alias} and {node.right_alias}: {{len({node.new_alias})}} unique rows')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_IntersectionNode(self, node):
        """Generate: df = pd.merge(df1, df2, how='inner')"""
        code = f"{node.new_alias} = pd.merge({node.left_alias}, {node.right_alias}, how='inner').drop_duplicates().reset_index(drop=True)\n"
        code += f"print(f'Intersection of {node.left_alias} and {node.right_alias}: {{len({node.new_alias})}} common rows')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_DifferenceNode(self, node):
        """Generate: df = df1[~df1.isin(df2).all(axis=1)]"""
        code = f"_merged = {node.left_alias}.merge({node.right_alias}, how='outer', indicator=True)\n"
        code += f"{node.new_alias} = _merged[_merged['_merge'] == 'left_only'].drop('_merge', axis=1).reset_index(drop=True)\n"
        code += f"print(f'Difference of {node.left_alias} minus {node.right_alias}: {{len({node.new_alias})}} rows')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # ============================================================
    # PHASE 10: ADVANCED OPERATIONS - CODE GENERATORS
    # ============================================================

    def visit_SetIndexNode(self, node):
        """Generate: df = df.set_index('col', drop=True)"""
        code = f"{node.new_alias} = {node.source_alias}.set_index('{node.column}', drop={node.drop})\n"
        code += f"print(f'Set {node.column} as index')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ResetIndexNode(self, node):
        """Generate: df = df.reset_index(drop=False)"""
        code = f"{node.new_alias} = {node.source_alias}.reset_index(drop={node.drop})\n"
        code += f"print(f'Reset index')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ApplyRowNode(self, node):
        """Generate: df['result'] = df.apply(func, axis=1)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['applied_result'] = {node.source_alias}.apply({node.function_expr}, axis=1)\n"
        code += f"print(f'Applied function to each row')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ApplyColumnNode(self, node):
        """Generate: df['col_transformed'] = df['col'].apply(func)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_applied'] = {node.source_alias}['{node.column}'].apply({node.function_expr})\n"
        code += f"print(f'Applied function to column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ResampleNode(self, node):
        """Generate: df = df.resample('D')['col'].agg()"""
        code = f"{node.new_alias} = {node.source_alias}.resample('{node.rule}')['{node.column}'].{node.aggfunc}().reset_index()\n"
        code += f"print(f'Resampled data with rule {node.rule} using {node.aggfunc}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_AssignNode(self, node):
        """Generate: df['col'] = value"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}'] = {self._format_value(node.value)}\n"
        code += f"print(f'Assigned value to column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # ========================================================================
    # HIGH-PRIORITY MISSING OPERATIONS (Phase 11) - Code Generator Methods
    # ========================================================================

    # Cumulative Operations
    def visit_CumSumNode(self, node):
        """Generate: df['col_cumsum'] = df['col'].cumsum()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_cumsum'] = {node.source_alias}['{node.column}'].cumsum()\n"
        code += f"print(f'Computed cumulative sum for column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_CumMaxNode(self, node):
        """Generate: df['col_cummax'] = df['col'].cummax()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_cummax'] = {node.source_alias}['{node.column}'].cummax()\n"
        code += f"print(f'Computed cumulative maximum for column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_CumMinNode(self, node):
        """Generate: df['col_cummin'] = df['col'].cummin()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_cummin'] = {node.source_alias}['{node.column}'].cummin()\n"
        code += f"print(f'Computed cumulative minimum for column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_CumProdNode(self, node):
        """Generate: df['col_cumprod'] = df['col'].cumprod()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_cumprod'] = {node.source_alias}['{node.column}'].cumprod()\n"
        code += f"print(f'Computed cumulative product for column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # Time Series Operations
    def visit_PctChangeNode(self, node):
        """Generate: df['col_pct_change'] = df['col'].pct_change(periods=n)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_pct_change'] = {node.source_alias}['{node.column}'].pct_change(periods={node.periods})\n"
        code += f"print(f'Computed percentage change for column {node.column} with periods={node.periods}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_DiffNode(self, node):
        """Generate: df['col_diff'] = df['col'].diff(periods=n)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_diff'] = {node.source_alias}['{node.column}'].diff(periods={node.periods})\n"
        code += f"print(f'Computed difference for column {node.column} with periods={node.periods}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ShiftNode(self, node):
        """Generate: df['col_shifted'] = df['col'].shift(periods=n, fill_value=val)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        if node.fill_value is not None:
            code += f"{node.new_alias}['{node.column}_shifted'] = {node.source_alias}['{node.column}'].shift(periods={node.periods}, fill_value={self._format_value(node.fill_value)})\n"
        else:
            code += f"{node.new_alias}['{node.column}_shifted'] = {node.source_alias}['{node.column}'].shift(periods={node.periods})\n"
        code += f"print(f'Shifted column {node.column} by {node.periods} periods')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # Apply/Map Operations
    def visit_ApplyMapNode(self, node):
        """Generate: df = df.map(function) or df.applymap(function) for older pandas"""
        self.imports.add("import pandas as pd")
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        # Use map() for pandas 2.1+ (applymap is deprecated)
        code += f"# Using map for element-wise function application\n"
        code += f"{node.new_alias} = {node.source_alias}.map({node.function_expr})\n"
        code += f"print(f'Applied function element-wise to dataframe')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_MapValuesNode(self, node):
        """Generate: df['col_mapped'] = df['col'].map(mapping_dict)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_mapped'] = {node.source_alias}['{node.column}'].map({node.mapping})\n"
        code += f"print(f'Mapped values in column {node.column} using dictionary')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # Additional Date/Time Extraction Operations
    def visit_ExtractHourNode(self, node):
        """Generate: df['hour'] = df['col'].dt.hour"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_hour'] = {node.source_alias}['{node.column}'].dt.hour\n"
        code += f"print(f'Extracted hour from column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ExtractMinuteNode(self, node):
        """Generate: df['minute'] = df['col'].dt.minute"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_minute'] = {node.source_alias}['{node.column}'].dt.minute\n"
        code += f"print(f'Extracted minute from column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ExtractSecondNode(self, node):
        """Generate: df['second'] = df['col'].dt.second"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_second'] = {node.source_alias}['{node.column}'].dt.second\n"
        code += f"print(f'Extracted second from column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ExtractDayOfWeekNode(self, node):
        """Generate: df['day_of_week'] = df['col'].dt.dayofweek"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_dayofweek'] = {node.source_alias}['{node.column}'].dt.dayofweek\n"
        code += f"print(f'Extracted day of week from column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ExtractDayOfYearNode(self, node):
        """Generate: df['day_of_year'] = df['col'].dt.dayofyear"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_dayofyear'] = {node.source_alias}['{node.column}'].dt.dayofyear\n"
        code += f"print(f'Extracted day of year from column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ExtractWeekOfYearNode(self, node):
        """Generate: df['week_of_year'] = df['col'].dt.isocalendar().week"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_weekofyear'] = {node.source_alias}['{node.column}'].dt.isocalendar().week\n"
        code += f"print(f'Extracted week of year from column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_ExtractQuarterNode(self, node):
        """Generate: df['quarter'] = df['col'].dt.quarter"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_quarter'] = {node.source_alias}['{node.column}'].dt.quarter\n"
        code += f"print(f'Extracted quarter from column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # Date Arithmetic Operations
    def visit_DateAddNode(self, node):
        """Generate: df['date_future'] = df['col'] + pd.Timedelta(**{unit: value})"""
        self.imports.add("import pandas as pd")
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_plus_{node.value}{node.unit}'] = {node.source_alias}['{node.column}'] + pd.Timedelta(**{{'{node.unit}': {node.value}}})\n"
        code += f"print(f'Added {node.value} {node.unit} to column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_DateSubtractNode(self, node):
        """Generate: df['date_past'] = df['col'] - pd.Timedelta(**{unit: value})"""
        self.imports.add("import pandas as pd")
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_minus_{node.value}{node.unit}'] = {node.source_alias}['{node.column}'] - pd.Timedelta(**{{'{node.unit}': {node.value}}})\n"
        code += f"print(f'Subtracted {node.value} {node.unit} from column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_FormatDateTimeNode(self, node):
        """Generate: df['date_formatted'] = df['col'].dt.strftime(format)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_formatted'] = {node.source_alias}['{node.column}'].dt.strftime('{node.format_string}')\n"
        code += f"print(f'Formatted datetime column {node.column} with format {node.format_string}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # Advanced String Operations
    def visit_ExtractRegexNode(self, node):
        """Generate: df['extracted'] = df['col'].str.extract(pattern, expand=False)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        # Add capture group if pattern doesn't have one
        pattern = node.pattern if '(' in node.pattern else f'({node.pattern})'
        code += f"{node.new_alias}['{node.column}_extracted'] = {node.source_alias}['{node.column}'].str.extract(r'{pattern}', expand=False)\n"
        if node.group > 0:
            code += f"# Note: Extracting group {node.group}\n"
        code += f"print(f'Extracted regex pattern from column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_TitleNode(self, node):
        """Generate: df['col_title'] = df['col'].str.title()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_title'] = {node.source_alias}['{node.column}'].str.title()\n"
        code += f"print(f'Converted column {node.column} to title case')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_CapitalizeNode(self, node):
        """Generate: df['col_capitalized'] = df['col'].str.capitalize()"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_capitalized'] = {node.source_alias}['{node.column}'].str.capitalize()\n"
        code += f"print(f'Capitalized column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_LStripNode(self, node):
        """Generate: df['col_lstripped'] = df['col'].str.lstrip(chars)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        if node.chars:
            code += f"{node.new_alias}['{node.column}_lstripped'] = {node.source_alias}['{node.column}'].str.lstrip('{node.chars}')\n"
        else:
            code += f"{node.new_alias}['{node.column}_lstripped'] = {node.source_alias}['{node.column}'].str.lstrip()\n"
        code += f"print(f'Left-stripped column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_RStripNode(self, node):
        """Generate: df['col_rstripped'] = df['col'].str.rstrip(chars)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        if node.chars:
            code += f"{node.new_alias}['{node.column}_rstripped'] = {node.source_alias}['{node.column}'].str.rstrip('{node.chars}')\n"
        else:
            code += f"{node.new_alias}['{node.column}_rstripped'] = {node.source_alias}['{node.column}'].str.rstrip()\n"
        code += f"print(f'Right-stripped column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_FindNode(self, node):
        """Generate: df['col_position'] = df['col'].str.find(substring)"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"{node.new_alias}['{node.column}_position'] = {node.source_alias}['{node.column}'].str.find('{node.substring}')\n"
        code += f"print(f'Found position of substring in column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # Binning with Explicit Boundaries
    def visit_CutNode(self, node):
        """Generate: df['col_binned'] = pd.cut(df['col'], bins=bins, labels=labels)"""
        self.imports.add("import pandas as pd")
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        if node.labels:
            code += f"{node.new_alias}['{node.column}_binned'] = pd.cut({node.source_alias}['{node.column}'], bins={node.bins}, labels={node.labels}, include_lowest={node.include_lowest})\n"
        else:
            code += f"{node.new_alias}['{node.column}_binned'] = pd.cut({node.source_alias}['{node.column}'], bins={node.bins}, include_lowest={node.include_lowest})\n"
        code += f"print(f'Binned column {node.column} with explicit boundaries')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # ===== PHASE 12: MEDIUM PRIORITY OPERATIONS =====

    # Scaling & Normalization Operations
    def visit_RobustScaleNode(self, node):
        """Generate: from sklearn.preprocessing import RobustScaler; df['col_robust'] = RobustScaler().fit_transform(df[['col']])"""
        self.imports.add("from sklearn.preprocessing import RobustScaler")
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"scaler = RobustScaler()\n"
        code += f"{node.new_alias}['{node.column}_robust'] = scaler.fit_transform({node.source_alias}[['{node.column}']])\n"
        code += f"print(f'Applied robust scaling to column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_MaxAbsScaleNode(self, node):
        """Generate: from sklearn.preprocessing import MaxAbsScaler; df['col_maxabs'] = MaxAbsScaler().fit_transform(df[['col']])"""
        self.imports.add("from sklearn.preprocessing import MaxAbsScaler")
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"scaler = MaxAbsScaler()\n"
        code += f"{node.new_alias}['{node.column}_maxabs'] = scaler.fit_transform({node.source_alias}[['{node.column}']])\n"
        code += f"print(f'Applied max abs scaling to column {node.column}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # Advanced Encoding Operations
    def visit_OrdinalEncodeNode(self, node):
        """Generate: df['col_encoded'] = df['col'].map({'S': 1, 'M': 2, 'L': 3, 'XL': 4})"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        # Create mapping dict from order list
        mapping = {val: idx + 1 for idx, val in enumerate(node.order)}
        code += f"{node.new_alias}['{node.column}_encoded'] = {node.source_alias}['{node.column}'].map({mapping})\n"
        code += f"print(f'Ordinal encoded column {node.column} with order {node.order}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_TargetEncodeNode(self, node):
        """Generate: df['col_encoded'] = df['col'].map(df.groupby('col')['target'].mean())"""
        code = f"{node.new_alias} = {node.source_alias}.copy()\n"
        code += f"target_means = {node.source_alias}.groupby('{node.column}')['{node.target}'].mean()\n"
        code += f"{node.new_alias}['{node.column}_target_encoded'] = {node.source_alias}['{node.column}'].map(target_means)\n"
        code += f"print(f'Target encoded column {node.column} using target {node.target}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # Data Validation Operations
    def visit_AssertUniqueNode(self, node):
        """Generate: assert df['col'].is_unique, 'Duplicate values found in column'"""
        code = f"assert {node.source_alias}['{node.column}'].is_unique, f'Duplicate values found in column {node.column}'\n"
        code += f"print(f'Assertion passed: Column {node.column} has unique values')"
        self.code_lines.append(code)

    def visit_AssertNoNullsNode(self, node):
        """Generate: assert not df['col'].isnull().any(), 'Null values found in column'"""
        code = f"assert not {node.source_alias}['{node.column}'].isnull().any(), f'Null values found in column {node.column}'\n"
        code += f"print(f'Assertion passed: Column {node.column} has no null values')"
        self.code_lines.append(code)

    def visit_AssertRangeNode(self, node):
        """Generate: assert df['col'].between(min, max).all(), 'Values outside range'"""
        code = ""
        if node.min_value is not None and node.max_value is not None:
            code += f"assert {node.source_alias}['{node.column}'].between({node.min_value}, {node.max_value}).all(), f'Values outside range [{node.min_value}, {node.max_value}] in column {node.column}'\n"
            code += f"print(f'Assertion passed: Column {node.column} values are within range [{node.min_value}, {node.max_value}]')"
        elif node.min_value is not None:
            code += f"assert ({node.source_alias}['{node.column}'] >= {node.min_value}).all(), f'Values below minimum {node.min_value} in column {node.column}'\n"
            code += f"print(f'Assertion passed: Column {node.column} values are >= {node.min_value}')"
        elif node.max_value is not None:
            code += f"assert ({node.source_alias}['{node.column}'] <= {node.max_value}).all(), f'Values above maximum {node.max_value} in column {node.column}'\n"
            code += f"print(f'Assertion passed: Column {node.column} values are <= {node.max_value}')"
        self.code_lines.append(code)

    # Advanced Index Operations
    def visit_ReindexNode(self, node):
        """Generate: df_reindexed = df.reindex([0, 1, 2, 3])"""
        code = f"{node.new_alias} = {node.source_alias}.reindex({node.index})\n"
        code += f"print(f'Reindexed dataframe with new index {node.index}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    def visit_SetMultiIndexNode(self, node):
        """Generate: df_hierarchical = df.set_index(['category', 'subcategory'])"""
        # Convert list to proper Python list representation
        columns_repr = repr(node.columns)
        columns_str = ', '.join(node.columns)
        code = f"{node.new_alias} = {node.source_alias}.set_index({columns_repr})\n"
        code += f"print(f'Set multi-index using columns: {columns_str}')"
        self.code_lines.append(code)
        self.symbol_table[node.new_alias] = True

    # Boolean Operations
    def visit_AnyNode(self, node):
        """Generate: result = df['col'].any()"""
        code = f"result = {node.source_alias}['{node.column}'].any()\n"
        code += f"print(f'Any True in column {node.column}: {{result}}')"
        self.code_lines.append(code)

    def visit_AllNode(self, node):
        """Generate: result = df['col'].all()"""
        code = f"result = {node.source_alias}['{node.column}'].all()\n"
        code += f"print(f'All True in column {node.column}: {{result}}')"
        self.code_lines.append(code)

    def visit_CountTrueNode(self, node):
        """Generate: result = df['col'].sum()"""
        code = f"result = {node.source_alias}['{node.column}'].sum()\n"
        code += f"print(f'Count of True values in column {node.column}: {{result}}')"
        self.code_lines.append(code)

    def visit_CompareNode(self, node):
        """Generate: comparison = df1.compare(df2)"""
        code = f"comparison = {node.left_alias}.compare({node.right_alias})\n"
        code += f"print(f'Compared {node.left_alias} with {node.right_alias}')\n"
        code += f"print(comparison)"
        self.code_lines.append(code)
