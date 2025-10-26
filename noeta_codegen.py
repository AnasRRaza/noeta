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
        self.code_lines.append(f"print('Sorted {node.source_alias} by {columns}')")
    
    def visit_JoinNode(self, node: JoinNode):
        code = f"{node.new_alias} = pd.merge({node.alias1}, {node.alias2}, "
        code += f"on='{node.join_column}', how='inner')"
        self.code_lines.append(code)
        self.code_lines.append(f"print(f'Joined {node.alias1} and {node.alias2}: {{len({node.new_alias})}} rows')")
    
    def visit_GroupByNode(self, node: GroupByNode):
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
        code += f"{node.source_alias}[{node.columns}].boxplot()\n"
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
