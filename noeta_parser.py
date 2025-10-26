"""
Noeta Parser - Builds AST from tokens
"""
from typing import List, Optional
from noeta_lexer import Token, TokenType
from noeta_ast import *

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
    
    def current_token(self) -> Optional[Token]:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]
    
    def peek_token(self, offset=1) -> Optional[Token]:
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return None
        return self.tokens[pos]
    
    def advance(self):
        self.pos += 1
    
    def expect(self, token_type: TokenType) -> Token:
        token = self.current_token()
        if not token or token.type != token_type:
            raise SyntaxError(f"Expected {token_type}, got {token.type if token else 'EOF'}")
        self.advance()
        return token
    
    def match(self, *token_types: TokenType) -> bool:
        token = self.current_token()
        return token and token.type in token_types
    
    def parse(self) -> ProgramNode:
        statements = []
        while self.current_token() and self.current_token().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        return ProgramNode(statements)
    
    def parse_statement(self) -> Optional[ASTNode]:
        token = self.current_token()
        if not token:
            return None
        
        # Data manipulation statements
        if token.type == TokenType.LOAD:
            return self.parse_load()
        elif token.type == TokenType.SELECT:
            return self.parse_select()
        elif token.type == TokenType.FILTER:
            return self.parse_filter()
        elif token.type == TokenType.SORT:
            return self.parse_sort()
        elif token.type == TokenType.JOIN:
            return self.parse_join()
        elif token.type == TokenType.GROUPBY:
            return self.parse_groupby()
        elif token.type == TokenType.SAMPLE:
            return self.parse_sample()
        elif token.type == TokenType.DROPNA:
            return self.parse_dropna()
        elif token.type == TokenType.FILLNA:
            return self.parse_fillna()
        elif token.type == TokenType.MUTATE:
            return self.parse_mutate()
        elif token.type == TokenType.APPLY:
            return self.parse_apply()
        
        # Analysis statements
        elif token.type == TokenType.DESCRIBE:
            return self.parse_describe()
        elif token.type == TokenType.SUMMARY:
            return self.parse_summary()
        elif token.type == TokenType.INFO:
            return self.parse_info()
        elif token.type == TokenType.OUTLIERS:
            return self.parse_outliers()
        elif token.type == TokenType.QUANTILE:
            return self.parse_quantile()
        elif token.type == TokenType.NORMALIZE:
            return self.parse_normalize()
        elif token.type == TokenType.BINNING:
            return self.parse_binning()
        elif token.type == TokenType.ROLLING:
            return self.parse_rolling()
        elif token.type == TokenType.HYPOTHESIS:
            return self.parse_hypothesis()
        
        # Visualization statements
        elif token.type == TokenType.BOXPLOT:
            return self.parse_boxplot()
        elif token.type == TokenType.HEATMAP:
            return self.parse_heatmap()
        elif token.type == TokenType.PAIRPLOT:
            return self.parse_pairplot()
        elif token.type == TokenType.TIMESERIES:
            return self.parse_timeseries()
        elif token.type == TokenType.PIE:
            return self.parse_pie()
        
        # File operations
        elif token.type == TokenType.SAVE:
            return self.parse_save()
        elif token.type == TokenType.EXPORT_PLOT:
            return self.parse_export_plot()
        
        else:
            raise SyntaxError(f"Unexpected token: {token.type}")
    
    def parse_load(self) -> LoadNode:
        self.expect(TokenType.LOAD)
        file_path = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        alias = self.expect(TokenType.IDENTIFIER).value
        return LoadNode(file_path, alias)
    
    def parse_select(self) -> SelectNode:
        self.expect(TokenType.SELECT)
        source = self.expect(TokenType.IDENTIFIER).value
        columns = self.parse_column_list()
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return SelectNode(source, columns, new_alias)
    
    def parse_filter(self) -> FilterNode:
        self.expect(TokenType.FILTER)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.LBRACKET)
        condition = self.parse_condition()
        self.expect(TokenType.RBRACKET)
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FilterNode(source, condition, new_alias)
    
    def parse_sort(self) -> SortNode:
        self.expect(TokenType.SORT)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.BY)
        self.expect(TokenType.COLON)
        sort_specs = self.parse_sort_specs()
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return SortNode(source, sort_specs, new_alias)
    
    def parse_join(self) -> JoinNode:
        self.expect(TokenType.JOIN)
        alias1 = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.COLON)
        alias2 = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ON)
        self.expect(TokenType.COLON)
        join_column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return JoinNode(alias1, alias2, join_column, new_alias)
    
    def parse_groupby(self) -> GroupByNode:
        self.expect(TokenType.GROUPBY)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.BY)
        self.expect(TokenType.COLON)
        group_columns = self.parse_column_list()
        self.expect(TokenType.AGG)
        self.expect(TokenType.COLON)
        aggregations = self.parse_aggregations()
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return GroupByNode(source, group_columns, aggregations, new_alias)
    
    def parse_sample(self) -> SampleNode:
        self.expect(TokenType.SAMPLE)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.N)
        self.expect(TokenType.COLON)
        size = int(self.expect(TokenType.NUMERIC_LITERAL).value)
        is_random = False
        if self.match(TokenType.RANDOM):
            is_random = True
            self.advance()
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return SampleNode(source, size, is_random, new_alias)
    
    def parse_dropna(self) -> DropNANode:
        self.expect(TokenType.DROPNA)
        source = self.expect(TokenType.IDENTIFIER).value
        columns = None
        if self.match(TokenType.COLUMNS):
            self.advance()
            self.expect(TokenType.COLON)
            columns = self.parse_column_list()
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return DropNANode(source, columns, new_alias)
    
    def parse_fillna(self) -> FillNANode:
        self.expect(TokenType.FILLNA)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.VALUE)
        self.expect(TokenType.COLON)
        
        # Parse fill value (can be string or numeric)
        token = self.current_token()
        if token.type == TokenType.STRING_LITERAL:
            fill_value = token.value
        elif token.type == TokenType.NUMERIC_LITERAL:
            fill_value = token.value
        else:
            raise SyntaxError(f"Expected literal value, got {token.type}")
        self.advance()
        
        columns = None
        if self.match(TokenType.COLUMNS):
            self.advance()
            self.expect(TokenType.COLON)
            columns = self.parse_column_list()
        
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FillNANode(source, fill_value, columns, new_alias)
    
    def parse_mutate(self) -> MutateNode:
        self.expect(TokenType.MUTATE)
        source = self.expect(TokenType.IDENTIFIER).value
        mutations = self.parse_mutations()
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return MutateNode(source, mutations, new_alias)
    
    def parse_apply(self) -> ApplyNode:
        self.expect(TokenType.APPLY)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMNS)
        self.expect(TokenType.COLON)
        columns = self.parse_column_list()
        self.expect(TokenType.FUNCTION)
        self.expect(TokenType.COLON)
        function_expr = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ApplyNode(source, columns, function_expr, new_alias)
    
    def parse_describe(self) -> DescribeNode:
        self.expect(TokenType.DESCRIBE)
        source = self.expect(TokenType.IDENTIFIER).value
        columns = None
        if self.match(TokenType.COLUMNS):
            self.advance()
            self.expect(TokenType.COLON)
            columns = self.parse_column_list()
        return DescribeNode(source, columns)
    
    def parse_summary(self) -> SummaryNode:
        self.expect(TokenType.SUMMARY)
        source = self.expect(TokenType.IDENTIFIER).value
        return SummaryNode(source)
    
    def parse_info(self) -> InfoNode:
        self.expect(TokenType.INFO)
        source = self.expect(TokenType.IDENTIFIER).value
        return InfoNode(source)
    
    def parse_outliers(self) -> OutliersNode:
        self.expect(TokenType.OUTLIERS)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.METHOD)
        self.expect(TokenType.COLON)
        method = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMNS)
        self.expect(TokenType.COLON)
        columns = self.parse_column_list()
        return OutliersNode(source, method, columns)
    
    def parse_quantile(self) -> QuantileNode:
        self.expect(TokenType.QUANTILE)
        source = self.expect(TokenType.IDENTIFIER).value
        # Accept either 'column:' or 'columns:'
        if self.match(TokenType.COLUMN, TokenType.COLUMNS):
            self.advance()
            self.expect(TokenType.COLON)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.Q)
        self.expect(TokenType.COLON)
        q_value = float(self.expect(TokenType.NUMERIC_LITERAL).value)
        return QuantileNode(source, column, q_value)
    
    def parse_normalize(self) -> NormalizeNode:
        self.expect(TokenType.NORMALIZE)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMNS)
        self.expect(TokenType.COLON)
        columns = self.parse_column_list()
        self.expect(TokenType.METHOD)
        self.expect(TokenType.COLON)
        method = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return NormalizeNode(source, columns, method, new_alias)
    
    def parse_binning(self) -> BinningNode:
        self.expect(TokenType.BINNING)
        source = self.expect(TokenType.IDENTIFIER).value
        # Accept either 'column:' or 'columns:'
        if self.match(TokenType.COLUMN, TokenType.COLUMNS):
            self.advance()
            self.expect(TokenType.COLON)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.BINS)
        self.expect(TokenType.COLON)
        num_bins = int(self.expect(TokenType.NUMERIC_LITERAL).value)
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return BinningNode(source, column, num_bins, new_alias)
    
    def parse_rolling(self) -> RollingNode:
        self.expect(TokenType.ROLLING)
        source = self.expect(TokenType.IDENTIFIER).value
        # Accept either 'column:' or 'columns:'
        if self.match(TokenType.COLUMN, TokenType.COLUMNS):
            self.advance()
            self.expect(TokenType.COLON)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WINDOW)
        self.expect(TokenType.COLON)
        window = int(self.expect(TokenType.NUMERIC_LITERAL).value)
        self.expect(TokenType.FUNCTION)
        self.expect(TokenType.COLON)
        function = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return RollingNode(source, column, window, function, new_alias)
    
    def parse_hypothesis(self) -> HypothesisNode:
        self.expect(TokenType.HYPOTHESIS)
        alias1 = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.VS)
        self.expect(TokenType.COLON)
        alias2 = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMNS)
        self.expect(TokenType.COLON)
        columns = self.parse_column_list()
        self.expect(TokenType.TEST)
        self.expect(TokenType.COLON)
        test_type = self.expect(TokenType.IDENTIFIER).value
        return HypothesisNode(alias1, alias2, columns, test_type)
    
    def parse_boxplot(self) -> BoxPlotNode:
        self.expect(TokenType.BOXPLOT)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMNS)
        self.expect(TokenType.COLON)
        columns = self.parse_column_list()
        return BoxPlotNode(source, columns)
    
    def parse_heatmap(self) -> HeatmapNode:
        self.expect(TokenType.HEATMAP)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMNS)
        self.expect(TokenType.COLON)
        columns = self.parse_column_list()
        return HeatmapNode(source, columns)
    
    def parse_pairplot(self) -> PairPlotNode:
        self.expect(TokenType.PAIRPLOT)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMNS)
        self.expect(TokenType.COLON)
        columns = self.parse_column_list()
        return PairPlotNode(source, columns)
    
    def parse_timeseries(self) -> TimeSeriesNode:
        self.expect(TokenType.TIMESERIES)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.X)
        self.expect(TokenType.COLON)
        x_column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.Y)
        self.expect(TokenType.COLON)
        y_column = self.expect(TokenType.IDENTIFIER).value
        return TimeSeriesNode(source, x_column, y_column)
    
    def parse_pie(self) -> PieChartNode:
        self.expect(TokenType.PIE)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.VALUES)
        self.expect(TokenType.COLON)
        values_column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.LABELS)
        self.expect(TokenType.COLON)
        labels_column = self.expect(TokenType.IDENTIFIER).value
        return PieChartNode(source, values_column, labels_column)
    
    def parse_save(self) -> SaveNode:
        self.expect(TokenType.SAVE)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.TO)
        self.expect(TokenType.COLON)
        file_path = self.expect(TokenType.STRING_LITERAL).value
        format_type = None
        if self.match(TokenType.FORMAT):
            self.advance()
            self.expect(TokenType.COLON)
            format_type = self.expect(TokenType.IDENTIFIER).value
        return SaveNode(source, file_path, format_type)
    
    def parse_export_plot(self) -> ExportPlotNode:
        self.expect(TokenType.EXPORT_PLOT)
        self.expect(TokenType.FILENAME)
        self.expect(TokenType.COLON)
        file_name = self.expect(TokenType.STRING_LITERAL).value
        width = None
        height = None
        if self.match(TokenType.WIDTH):
            self.advance()
            self.expect(TokenType.COLON)
            width = int(self.expect(TokenType.NUMERIC_LITERAL).value)
        if self.match(TokenType.HEIGHT):
            self.advance()
            self.expect(TokenType.COLON)
            height = int(self.expect(TokenType.NUMERIC_LITERAL).value)
        return ExportPlotNode(file_name, width, height)
    
    # Helper parsing methods
    def parse_column_list(self) -> List[str]:
        self.expect(TokenType.LBRACE)
        columns = []
        columns.append(self.expect(TokenType.IDENTIFIER).value)
        while self.match(TokenType.COMMA):
            self.advance()
            columns.append(self.expect(TokenType.IDENTIFIER).value)
        self.expect(TokenType.RBRACE)
        return columns
    
    def parse_condition(self) -> ConditionNode:
        left = self.expect(TokenType.IDENTIFIER).value
        
        # Parse operator
        op_token = self.current_token()
        if op_token.type in [TokenType.EQ, TokenType.NEQ, TokenType.LT, 
                             TokenType.GT, TokenType.LTE, TokenType.GTE]:
            operator = op_token.value
            self.advance()
        else:
            raise SyntaxError(f"Expected comparison operator, got {op_token.type}")
        
        # Parse right operand (can be identifier, string, or number)
        right_token = self.current_token()
        if right_token.type == TokenType.IDENTIFIER:
            right = right_token.value
        elif right_token.type == TokenType.STRING_LITERAL:
            right = right_token.value
        elif right_token.type == TokenType.NUMERIC_LITERAL:
            right = right_token.value
        else:
            raise SyntaxError(f"Expected identifier or literal, got {right_token.type}")
        self.advance()
        
        return ConditionNode(left, operator, right)
    
    def parse_sort_specs(self) -> List[SortSpecNode]:
        specs = []
        # Parse first sort spec
        column = self.expect(TokenType.IDENTIFIER).value
        direction = 'ASC'
        if self.match(TokenType.DESC):
            direction = 'DESC'
            self.advance()
        specs.append(SortSpecNode(column, direction))
        
        # Parse additional sort specs
        while self.match(TokenType.COMMA):
            self.advance()
            column = self.expect(TokenType.IDENTIFIER).value
            direction = 'ASC'
            if self.match(TokenType.DESC):
                direction = 'DESC'
                self.advance()
            specs.append(SortSpecNode(column, direction))
        
        return specs
    
    def parse_aggregations(self) -> List[AggregationNode]:
        self.expect(TokenType.LBRACE)
        aggregations = []
        
        # Parse first aggregation
        func_name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLON)
        column_name = self.expect(TokenType.IDENTIFIER).value
        aggregations.append(AggregationNode(func_name, column_name))
        
        # Parse additional aggregations
        while self.match(TokenType.COMMA):
            self.advance()
            func_name = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.COLON)
            column_name = self.expect(TokenType.IDENTIFIER).value
            aggregations.append(AggregationNode(func_name, column_name))
        
        self.expect(TokenType.RBRACE)
        return aggregations
    
    def parse_mutations(self) -> List[MutationNode]:
        self.expect(TokenType.LBRACE)
        mutations = []
        
        # Parse first mutation
        new_column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLON)
        expression = self.expect(TokenType.STRING_LITERAL).value
        mutations.append(MutationNode(new_column, expression))
        
        # Parse additional mutations
        while self.match(TokenType.COMMA):
            self.advance()
            new_column = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.COLON)
            expression = self.expect(TokenType.STRING_LITERAL).value
            mutations.append(MutationNode(new_column, expression))
        
        self.expect(TokenType.RBRACE)
        return mutations
