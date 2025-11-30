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
        # Determine file type and use appropriate pandas function
        if node.file_path.endswith('.csv'):
            code = f"{node.alias} = pd.read_csv('{node.file_path}')"
        elif node.file_path.endswith('.json'):
            code = f"{node.alias} = pd.read_json('{node.file_path}')"
        elif node.file_path.endswith(('.xlsx', '.xls')):
            code = f"{node.alias} = pd.read_excel('{node.file_path}')"
        else:
            code = f"{node.alias} = pd.read_csv('{node.file_path}')"  # Default to CSV

        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Loaded {node.file_path} as {node.alias}: {{len({node.alias})}} rows, {{len({node.alias}.columns)}} columns')")

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
        code = f"{node.new_alias} = {node.source_alias}.head({node.n_rows})"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Selected first {node.n_rows} rows from {node.source_alias}')")
        self.symbol_table[node.new_alias] = True

    def visit_TailNode(self, node: TailNode):
        """Generate code for getting last N rows"""
        code = f"{node.new_alias} = {node.source_alias}.tail({node.n_rows})"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Selected last {node.n_rows} rows from {node.source_alias}')")
        self.symbol_table[node.new_alias] = True

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
        if node.is_random:
            code = f"{node.new_alias} = {node.source_alias}.sample(n={node.sample_size}, random_state=42).copy()"
        else:
            code = f"{node.new_alias} = {node.source_alias}.head({node.sample_size}).copy()"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Sampled {node.sample_size} rows from {node.source_alias}')")
    
    def visit_DropNANode(self, node: DropNANode):
        if node.columns:
            code = f"{node.new_alias} = {node.source_alias}.dropna(subset={node.columns}).copy()"
        else:
            code = f"{node.new_alias} = {node.source_alias}.dropna().copy()"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Dropped NA values: {{len({node.source_alias}) - len({node.new_alias})}} rows removed')")
    
    def visit_FillNANode(self, node: FillNANode):
        if node.columns:
            code = f"{node.new_alias} = {node.source_alias}.copy()\n"
            for col in node.columns:
                if isinstance(node.fill_value, str):
                    code += f"{node.new_alias}['{col}'] = {node.new_alias}['{col}'].fillna('{node.fill_value}')\n"
                else:
                    code += f"{node.new_alias}['{col}'] = {node.new_alias}['{col}'].fillna({node.fill_value})\n"
        else:
            if isinstance(node.fill_value, str):
                code = f"{node.new_alias} = {node.source_alias}.fillna('{node.fill_value}').copy()"
            else:
                code = f"{node.new_alias} = {node.source_alias}.fillna({node.fill_value}).copy()"
        
        self.code_lines.append(code)
        self.code_lines.append(f"print('Filled NA values in {node.source_alias}')")
    
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
        if node.format_type == 'parquet':
            code = f"{node.source_alias}.to_parquet('{node.file_path}', index=False)\n"
        elif node.format_type == 'json':
            code = f"{node.source_alias}.to_json('{node.file_path}', orient='records', indent=2)\n"
        else:  # Default to CSV
            code = f"{node.source_alias}.to_csv('{node.file_path}', index=False)\n"
        
        code += f"print(f'Saved {node.source_alias} to {node.file_path}')"
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
