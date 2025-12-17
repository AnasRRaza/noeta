"""
Noeta Parser - Builds AST from tokens
"""
from typing import List, Optional
from noeta_lexer import Token, TokenType
from noeta_ast import *
from noeta_errors import (
    NoetaError, ErrorCategory, ErrorContext,
    get_token_type_description, suggest_similar
)

class Parser:
    def __init__(self, tokens: List[Token], source_code: str = ""):
        self.tokens = tokens
        self.pos = 0
        self.source_code = source_code
        self.source_lines = source_code.split('\n') if source_code else []
    
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
    
    def expect(self, token_type: TokenType, context: str = "") -> Token:
        """
        Expect specific token with rich error on mismatch.

        Args:
            token_type: Expected token type
            context: Optional context description (e.g., "select statement")

        Returns:
            The expected token

        Raises:
            NoetaError: If token doesn't match expected type
        """
        token = self.current_token()

        if not token or token.type == TokenType.EOF:
            # Hit EOF unexpectedly
            context_msg = f" in {context}" if context else ""
            message = f"Unexpected end of file{context_msg}. Expected {self._friendly_token_name(token_type)}"

            # For EOF, use the last token's position if available
            error_context = None
            if self.pos > 0 and self.pos - 1 < len(self.tokens):
                last_token = self.tokens[self.pos - 1]
                error_context = ErrorContext(
                    line=last_token.line,
                    column=last_token.column + (len(last_token.value) if hasattr(last_token, 'value') else 1),
                    length=1,
                    source_line=self._get_source_line(last_token.line)
                )

            raise NoetaError(
                message=message,
                category=ErrorCategory.SYNTAX,
                context=error_context,
                hint=f"The file ended before the {context or 'statement'} was complete"
            )

        if token.type != token_type:
            # Wrong token type
            context_msg = f" in {context}" if context else ""
            message = f"Expected {self._friendly_token_name(token_type)}, got {self._friendly_token_name(token.type)}{context_msg}"

            raise NoetaError(
                message=message,
                category=ErrorCategory.SYNTAX,
                context=self._create_error_context(token),
                hint=f"Check the syntax for {context or 'this operation'}"
            )

        self.advance()
        return token
    
    def match(self, *token_types: TokenType) -> bool:
        token = self.current_token()
        return token and token.type in token_types

    def _get_source_line(self, line_num: int) -> str:
        """Get specific line from source code."""
        if not self.source_lines or line_num < 1 or line_num > len(self.source_lines):
            return ""
        return self.source_lines[line_num - 1]

    def _create_error_context(self, token: Optional[Token] = None) -> Optional[ErrorContext]:
        """
        Create error context from current position.

        Args:
            token: Optional token to use for position (uses current token if None)

        Returns:
            ErrorContext or None if no position info available
        """
        if token is None:
            token = self.current_token()

        if not token:
            return None

        source_line = self._get_source_line(token.line)
        # Calculate length based on token value if available
        length = len(token.value) if hasattr(token, 'value') and token.value else 1

        return ErrorContext(
            line=token.line,
            column=token.column,
            length=length,
            source_line=source_line
        )

    def _friendly_token_name(self, token_type: TokenType) -> str:
        """Convert TokenType to human-friendly description."""
        return get_token_type_description(token_type.name)
    
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
            # Enhanced load dispatching based on next token
            return self.parse_load_enhanced()
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

        # Phase 2: Selection & Projection operations
        elif token.type == TokenType.SELECT_BY_TYPE:
            return self.parse_select_by_type()
        elif token.type == TokenType.HEAD:
            return self.parse_head()
        elif token.type == TokenType.TAIL:
            return self.parse_tail()
        elif token.type == TokenType.ILOC:
            return self.parse_iloc()
        elif token.type == TokenType.LOC:
            return self.parse_loc()
        elif token.type == TokenType.RENAME:
            return self.parse_rename_columns()
        elif token.type == TokenType.REORDER:
            return self.parse_reorder_columns()

        # Phase 3: Filtering operations
        elif token.type == TokenType.FILTER_BETWEEN:
            return self.parse_filter_between()
        elif token.type == TokenType.FILTER_ISIN:
            return self.parse_filter_isin()
        elif token.type == TokenType.FILTER_CONTAINS:
            return self.parse_filter_contains()
        elif token.type == TokenType.FILTER_STARTSWITH:
            return self.parse_filter_startswith()
        elif token.type == TokenType.FILTER_ENDSWITH:
            return self.parse_filter_endswith()
        elif token.type == TokenType.FILTER_REGEX:
            return self.parse_filter_regex()
        elif token.type == TokenType.FILTER_NULL:
            return self.parse_filter_null()
        elif token.type == TokenType.FILTER_NOTNULL:
            return self.parse_filter_notnull()
        elif token.type == TokenType.FILTER_DUPLICATES:
            return self.parse_filter_duplicates()

        # Phase 4: Transformation operations
        # Math operations
        elif token.type == TokenType.ROUND:
            return self.parse_round()
        elif token.type == TokenType.ABS:
            return self.parse_abs()
        elif token.type == TokenType.SQRT:
            return self.parse_sqrt()
        elif token.type == TokenType.POWER:
            return self.parse_power()
        elif token.type == TokenType.LOG:
            return self.parse_log()
        elif token.type == TokenType.CEIL:
            return self.parse_ceil()
        elif token.type == TokenType.FLOOR:
            return self.parse_floor()

        # String operations
        elif token.type == TokenType.UPPER:
            return self.parse_upper()
        elif token.type == TokenType.LOWER:
            return self.parse_lower()
        elif token.type == TokenType.STRIP:
            return self.parse_strip()
        elif token.type == TokenType.REPLACE:
            return self.parse_replace()
        elif token.type == TokenType.SPLIT:
            return self.parse_split()
        elif token.type == TokenType.CONCAT:
            return self.parse_concat()
        elif token.type == TokenType.SUBSTRING:
            return self.parse_substring()
        elif token.type == TokenType.LENGTH:
            return self.parse_length()

        # Date operations
        elif token.type == TokenType.PARSE_DATETIME:
            return self.parse_parse_datetime()
        elif token.type == TokenType.EXTRACT:
            return self.parse_extract()
        elif token.type == TokenType.EXTRACT_YEAR:
            return self.parse_extract_year()
        elif token.type == TokenType.EXTRACT_MONTH:
            return self.parse_extract_month()
        elif token.type == TokenType.EXTRACT_DAY:
            return self.parse_extract_day()
        elif token.type == TokenType.DATE_DIFF:
            return self.parse_date_diff()

        # Type operations
        elif token.type == TokenType.ASTYPE:
            return self.parse_astype()
        elif token.type == TokenType.TO_NUMERIC:
            return self.parse_to_numeric()

        # Encoding operations
        elif token.type == TokenType.ONE_HOT_ENCODE:
            return self.parse_one_hot_encode()
        elif token.type == TokenType.LABEL_ENCODE:
            return self.parse_label_encode()

        # Scaling operations
        elif token.type == TokenType.STANDARD_SCALE:
            return self.parse_standard_scale()
        elif token.type == TokenType.MINMAX_SCALE:
            return self.parse_minmax_scale()

        # Phase 5: Cleaning operations
        elif token.type == TokenType.ISNULL:
            return self.parse_isnull()
        elif token.type == TokenType.NOTNULL:
            return self.parse_notnull()
        elif token.type == TokenType.COUNT_NA:
            return self.parse_count_na()
        elif token.type == TokenType.FILL_FORWARD:
            return self.parse_fill_forward()
        elif token.type == TokenType.FILL_BACKWARD:
            return self.parse_fill_backward()
        elif token.type == TokenType.FILL_MEAN:
            return self.parse_fill_mean()
        elif token.type == TokenType.FILL_MEDIAN:
            return self.parse_fill_median()
        elif token.type == TokenType.INTERPOLATE:
            return self.parse_interpolate()
        elif token.type == TokenType.DUPLICATED:
            return self.parse_duplicated()
        elif token.type == TokenType.COUNT_DUPLICATES:
            return self.parse_count_duplicates()
        elif token.type == TokenType.DROP_DUPLICATES:
            return self.parse_drop_duplicates()
        elif token.type == TokenType.FILL_MODE:
            return self.parse_fill_mode()
        elif token.type == TokenType.QCUT:
            return self.parse_qcut()

        # Phase 6: Data Ordering operations
        elif token.type == TokenType.SORT_INDEX:
            return self.parse_sort_index()
        elif token.type == TokenType.RANK:
            return self.parse_rank()

        # Phase 7: Aggregation & Grouping operations
        elif token.type == TokenType.FILTER_GROUPS:
            return self.parse_filter_groups()
        elif token.type == TokenType.GROUP_TRANSFORM:
            return self.parse_group_transform()
        elif token.type == TokenType.WINDOW_RANK:
            return self.parse_window_rank()
        elif token.type == TokenType.WINDOW_LAG:
            return self.parse_window_lag()
        elif token.type == TokenType.WINDOW_LEAD:
            return self.parse_window_lead()
        elif token.type == TokenType.ROLLING_MEAN:
            return self.parse_rolling_mean()
        elif token.type == TokenType.ROLLING_SUM:
            return self.parse_rolling_sum()
        elif token.type == TokenType.ROLLING_STD:
            return self.parse_rolling_std()
        elif token.type == TokenType.ROLLING_MIN:
            return self.parse_rolling_min()
        elif token.type == TokenType.ROLLING_MAX:
            return self.parse_rolling_max()
        elif token.type == TokenType.EXPANDING_MEAN:
            return self.parse_expanding_mean()
        elif token.type == TokenType.EXPANDING_SUM:
            return self.parse_expanding_sum()
        elif token.type == TokenType.EXPANDING_MIN:
            return self.parse_expanding_min()
        elif token.type == TokenType.EXPANDING_MAX:
            return self.parse_expanding_max()

        # Phase 8: Data Reshaping operations
        elif token.type == TokenType.PIVOT:
            return self.parse_pivot()
        elif token.type == TokenType.PIVOT_TABLE:
            return self.parse_pivot_table()
        elif token.type == TokenType.MELT:
            return self.parse_melt()
        elif token.type == TokenType.STACK:
            return self.parse_stack()
        elif token.type == TokenType.UNSTACK:
            return self.parse_unstack()
        elif token.type == TokenType.TRANSPOSE:
            return self.parse_transpose()
        elif token.type == TokenType.CROSSTAB:
            return self.parse_crosstab()

        # Phase 9: Data Combining operations
        elif token.type == TokenType.MERGE:
            return self.parse_merge()
        elif token.type == TokenType.CONCAT_VERTICAL:
            return self.parse_concat_vertical()
        elif token.type == TokenType.CONCAT_HORIZONTAL:
            return self.parse_concat_horizontal()
        elif token.type == TokenType.UNION:
            return self.parse_union()
        elif token.type == TokenType.INTERSECTION:
            return self.parse_intersection()
        elif token.type == TokenType.DIFFERENCE:
            return self.parse_difference()

        # Phase 10: Advanced Operations
        elif token.type == TokenType.SET_INDEX:
            return self.parse_set_index()
        elif token.type == TokenType.RESET_INDEX:
            return self.parse_reset_index()
        elif token.type == TokenType.APPLY_ROW:
            return self.parse_apply_row()
        elif token.type == TokenType.APPLY_COLUMN:
            return self.parse_apply_column()
        elif token.type == TokenType.RESAMPLE:
            return self.parse_resample()
        elif token.type == TokenType.ASSIGN_CONST:
            return self.parse_assign()

        # Phase 11: High-Priority Missing Operations
        # Cumulative operations
        elif token.type == TokenType.CUMSUM:
            return self.parse_cumsum()
        elif token.type == TokenType.CUMMAX:
            return self.parse_cummax()
        elif token.type == TokenType.CUMMIN:
            return self.parse_cummin()
        elif token.type == TokenType.CUMPROD:
            return self.parse_cumprod()

        # Time series operations
        elif token.type == TokenType.PCT_CHANGE:
            return self.parse_pct_change()
        elif token.type == TokenType.DIFF:
            return self.parse_diff()
        elif token.type == TokenType.SHIFT:
            return self.parse_shift()

        # Apply/Map operations
        elif token.type == TokenType.APPLYMAP:
            return self.parse_applymap()
        elif token.type == TokenType.MAP_VALUES:
            return self.parse_map_values()

        # Additional date/time extractions
        elif token.type == TokenType.EXTRACT_HOUR:
            return self.parse_extract_hour()
        elif token.type == TokenType.EXTRACT_MINUTE:
            return self.parse_extract_minute()
        elif token.type == TokenType.EXTRACT_SECOND:
            return self.parse_extract_second()
        elif token.type == TokenType.EXTRACT_DAYOFWEEK:
            return self.parse_extract_dayofweek()
        elif token.type == TokenType.EXTRACT_DAYOFYEAR:
            return self.parse_extract_dayofyear()
        elif token.type == TokenType.EXTRACT_WEEKOFYEAR:
            return self.parse_extract_weekofyear()
        elif token.type == TokenType.EXTRACT_QUARTER:
            return self.parse_extract_quarter()

        # Date arithmetic
        elif token.type == TokenType.DATE_ADD:
            return self.parse_date_add()
        elif token.type == TokenType.DATE_SUBTRACT:
            return self.parse_date_subtract()
        elif token.type == TokenType.FORMAT_DATETIME:
            return self.parse_format_datetime()

        # Advanced string operations
        elif token.type == TokenType.EXTRACT_REGEX:
            return self.parse_extract_regex()
        elif token.type == TokenType.TITLE:
            return self.parse_title()
        elif token.type == TokenType.CAPITALIZE:
            return self.parse_capitalize()
        elif token.type == TokenType.LSTRIP:
            return self.parse_lstrip()
        elif token.type == TokenType.RSTRIP:
            return self.parse_rstrip()
        elif token.type == TokenType.FIND:
            return self.parse_find()

        # Binning with explicit boundaries
        elif token.type == TokenType.CUT:
            return self.parse_cut()

        # Phase 12: Medium Priority Operations
        elif token.type == TokenType.ROBUST_SCALE:
            return self.parse_robust_scale()
        elif token.type == TokenType.MAXABS_SCALE:
            return self.parse_maxabs_scale()
        elif token.type == TokenType.ORDINAL_ENCODE:
            return self.parse_ordinal_encode()
        elif token.type == TokenType.TARGET_ENCODE:
            return self.parse_target_encode()
        elif token.type == TokenType.ASSERT_UNIQUE:
            return self.parse_assert_unique()
        elif token.type == TokenType.ASSERT_NO_NULLS:
            return self.parse_assert_no_nulls()
        elif token.type == TokenType.ASSERT_RANGE:
            return self.parse_assert_range()
        elif token.type == TokenType.REINDEX:
            return self.parse_reindex()
        elif token.type == TokenType.SET_MULTIINDEX:
            return self.parse_set_multiindex()
        elif token.type == TokenType.ANY:
            return self.parse_any()
        elif token.type == TokenType.ALL:
            return self.parse_all()
        elif token.type == TokenType.COUNT_TRUE:
            return self.parse_count_true()
        elif token.type == TokenType.COMPARE:
            return self.parse_compare()

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
        elif token.type == TokenType.UNIQUE:
            return self.parse_unique()
        elif token.type == TokenType.VALUE_COUNTS:
            return self.parse_value_counts()
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
            return self.parse_save_enhanced()
        elif token.type == TokenType.EXPORT_PLOT:
            return self.parse_export_plot()
        
        else:
            raise SyntaxError(f"Unexpected token: {token.type}")
    
    def parse_load(self) -> LoadNode:
        """Parse: load <file> [with format=<fmt> <params>] as <alias>"""
        self.expect(TokenType.LOAD)
        filepath = self.expect(TokenType.STRING_LITERAL).value

        # Parse optional WITH clause for format and parameters
        format_type = None
        params = None

        if self.match(TokenType.WITH):
            self.advance()
            params = {}

            # Parse parameters until we hit AS
            while not self.match(TokenType.AS):
                if not self.current_token():
                    raise SyntaxError("Expected AS after load parameters")

                if not self.match(TokenType.IDENTIFIER):
                    raise SyntaxError(f"Expected parameter name in WITH clause")

                param_name = self.current_token().value
                self.advance()
                self.expect(TokenType.ASSIGN)
                param_value = self.parse_value()

                # Special handling for 'format' parameter
                if param_name == 'format':
                    format_type = param_value
                else:
                    params[param_name] = param_value

        # Auto-detect format from file extension if not explicitly specified
        if format_type is None:
            filepath_lower = filepath.lower()
            if filepath_lower.endswith('.csv'):
                format_type = 'csv'
            elif filepath_lower.endswith('.json'):
                format_type = 'json'
            elif filepath_lower.endswith(('.xlsx', '.xls')):
                format_type = 'excel'
            elif filepath_lower.endswith('.parquet'):
                format_type = 'parquet'
            # For SQL, format must be explicitly specified

        self.expect(TokenType.AS)
        alias = self.expect(TokenType.IDENTIFIER).value

        return LoadNode(filepath, alias, format_type, params)

    def parse_load_enhanced(self):
        """
        Enhanced load dispatcher supporting:
        - load csv "file.csv" as alias
        - load csv "file.csv" with params as alias
        - load json "file.json" as alias
        - load excel "file.xlsx" as alias
        - load parquet "file.parquet" as alias
        - load sql "query" from "connection" as alias
        """
        self.expect(TokenType.LOAD)

        # Check for format keyword
        if self.match(TokenType.CSV):
            return self.parse_load_csv()
        elif self.match(TokenType.JSON):
            return self.parse_load_json()
        elif self.match(TokenType.EXCEL):
            return self.parse_load_excel()
        elif self.match(TokenType.PARQUET):
            return self.parse_load_parquet()
        elif self.match(TokenType.SQL):
            return self.parse_load_sql()
        elif self.match(TokenType.STRING_LITERAL):
            # Fallback to old simple load
            file_path = self.expect(TokenType.STRING_LITERAL).value
            self.expect(TokenType.AS)
            alias = self.expect(TokenType.IDENTIFIER).value
            return LoadNode(file_path, alias)
        else:
            raise SyntaxError(f"Expected file format (csv, json, excel, parquet, sql) or file path after 'load'")

    def parse_load_csv(self) -> LoadCSVNode:
        """Parse: load csv "file.csv" [with params] as alias"""
        self.advance()  # consume CSV token
        filepath = self.expect(TokenType.STRING_LITERAL).value

        # Parse optional parameters
        params = {}
        if self.match(TokenType.WITH):
            self.advance()
            params = self.parse_params()

        self.expect(TokenType.AS)
        alias = self.expect(TokenType.IDENTIFIER).value
        return LoadCSVNode(filepath, params, alias)

    def parse_load_json(self) -> LoadJSONNode:
        """Parse: load json "file.json" [with params] as alias"""
        self.advance()  # consume JSON token
        filepath = self.expect(TokenType.STRING_LITERAL).value

        params = {}
        if self.match(TokenType.WITH):
            self.advance()
            params = self.parse_params()

        self.expect(TokenType.AS)
        alias = self.expect(TokenType.IDENTIFIER).value
        return LoadJSONNode(filepath, params, alias)

    def parse_load_excel(self) -> LoadExcelNode:
        """Parse: load excel "file.xlsx" [with params] as alias"""
        self.advance()  # consume EXCEL token
        filepath = self.expect(TokenType.STRING_LITERAL).value

        params = {}
        if self.match(TokenType.WITH):
            self.advance()
            params = self.parse_params()

        self.expect(TokenType.AS)
        alias = self.expect(TokenType.IDENTIFIER).value
        return LoadExcelNode(filepath, params, alias)

    def parse_load_parquet(self) -> LoadParquetNode:
        """Parse: load parquet "file.parquet" [with params] as alias"""
        self.advance()  # consume PARQUET token
        filepath = self.expect(TokenType.STRING_LITERAL).value

        params = {}
        if self.match(TokenType.WITH):
            self.advance()
            params = self.parse_params()

        self.expect(TokenType.AS)
        alias = self.expect(TokenType.IDENTIFIER).value
        return LoadParquetNode(filepath, params, alias)

    def parse_load_sql(self) -> LoadSQLNode:
        """Parse: load sql "query" from "connection" [with params] as alias"""
        self.advance()  # consume SQL token
        query = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.FROM)
        connection = self.expect(TokenType.STRING_LITERAL).value

        params = {}
        if self.match(TokenType.WITH):
            self.advance()
            params = self.parse_params()

        self.expect(TokenType.AS)
        alias = self.expect(TokenType.IDENTIFIER).value
        return LoadSQLNode(query, connection, params, alias)

    def parse_save_enhanced(self):
        """
        Enhanced save dispatcher supporting:
        - save data to "file.csv"
        - save data to "file.csv" with params
        """
        self.expect(TokenType.SAVE)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.TO)
        filepath = self.expect(TokenType.STRING_LITERAL).value

        # Parse optional parameters
        params = {}
        if self.match(TokenType.WITH):
            self.advance()
            params = self.parse_params()

        # Determine format from extension or params
        ext = filepath.lower().split('.')[-1]
        if ext == 'csv' or 'csv' in filepath.lower():
            return SaveCSVNode(source, filepath, params)
        elif ext == 'json':
            return SaveJSONNode(source, filepath, params)
        elif ext in ['xlsx', 'xls']:
            return SaveExcelNode(source, filepath, params)
        elif ext == 'parquet':
            return SaveParquetNode(source, filepath, params)
        else:
            # Default to CSV
            return SaveCSVNode(source, filepath, params)

    # Phase 2: Selection & Projection Parser Methods

    def parse_select_by_type(self) -> SelectByTypeNode:
        """Parse: select_by_type data with type="numeric" as alias"""
        self.advance()  # consume SELECT_BY_TYPE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.TYPE)
        self.expect(TokenType.ASSIGN)
        dtype = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return SelectByTypeNode(source, dtype, new_alias)

    def parse_head(self) -> HeadNode:
        """Parse: head data [with n=10] [as alias]"""
        self.advance()  # consume HEAD
        source = self.expect(TokenType.IDENTIFIER).value

        # Default to 5 rows if no 'with' clause
        n_rows = 5
        if self.match(TokenType.WITH):
            self.advance()
            self.expect(TokenType.N)
            self.expect(TokenType.ASSIGN)
            n_rows = self.expect(TokenType.NUMERIC_LITERAL).value

        # Make 'as' optional
        new_alias = None
        if self.match(TokenType.AS):
            self.advance()
            new_alias = self.expect(TokenType.IDENTIFIER).value

        return HeadNode(source, int(n_rows), new_alias)

    def parse_tail(self) -> TailNode:
        """Parse: tail data [with n=10] [as alias]"""
        self.advance()  # consume TAIL
        source = self.expect(TokenType.IDENTIFIER).value

        # Default to 5 rows if no 'with' clause
        n_rows = 5
        if self.match(TokenType.WITH):
            self.advance()
            self.expect(TokenType.N)
            self.expect(TokenType.ASSIGN)
            n_rows = self.expect(TokenType.NUMERIC_LITERAL).value

        # Make 'as' optional
        new_alias = None
        if self.match(TokenType.AS):
            self.advance()
            new_alias = self.expect(TokenType.IDENTIFIER).value

        return TailNode(source, int(n_rows), new_alias)

    def parse_iloc(self) -> ILocNode:
        """Parse: iloc data with rows=[0,10] as alias OR iloc data with rows=[0,10] columns=[0,3] as alias"""
        self.advance()  # consume ILOC
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)

        row_slice = None
        col_slice = None

        # Parse rows parameter
        if self.match(TokenType.ROWS):
            self.advance()
            self.expect(TokenType.ASSIGN)
            row_slice = self.parse_slice_value()

        # Parse optional columns parameter
        if self.match(TokenType.COLUMNS):
            self.advance()
            self.expect(TokenType.ASSIGN)
            col_slice = self.parse_slice_value()

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ILocNode(source, row_slice, col_slice, new_alias)

    def parse_loc(self) -> LocNode:
        """Parse: loc data with rows=["label1", "label2"] as alias"""
        self.advance()  # consume LOC
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)

        row_labels = None
        col_labels = None

        # Parse rows parameter
        if self.match(TokenType.ROWS):
            self.advance()
            self.expect(TokenType.ASSIGN)
            row_labels = self.parse_value()

        # Parse optional columns parameter
        if self.match(TokenType.COLUMNS):
            self.advance()
            self.expect(TokenType.ASSIGN)
            col_labels = self.parse_value()

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return LocNode(source, row_labels, col_labels, new_alias)

    def parse_rename_columns(self) -> RenameColumnsNode:
        """Parse: rename data with mapping={"old": "new", "old2": "new2"} as alias"""
        self.advance()  # consume RENAME
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.MAPPING)
        self.expect(TokenType.ASSIGN)
        mapping = self.parse_dict_value()
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return RenameColumnsNode(source, mapping, new_alias)

    def parse_reorder_columns(self) -> ReorderColumnsNode:
        """Parse: reorder data with order=["col1", "col2", "col3"] as alias"""
        self.advance()  # consume REORDER
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.ORDER)
        self.expect(TokenType.ASSIGN)
        column_order = self.parse_list_value()
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ReorderColumnsNode(source, column_order, new_alias)

    def parse_slice_value(self):
        """Parse slice notation: [start, end] or single value"""
        if self.match(TokenType.LBRACKET):
            self.advance()
            start = self.expect(TokenType.NUMERIC_LITERAL).value
            self.expect(TokenType.COMMA)
            end = self.expect(TokenType.NUMERIC_LITERAL).value
            self.expect(TokenType.RBRACKET)
            return (int(start), int(end))
        else:
            value = self.expect(TokenType.NUMERIC_LITERAL).value
            return int(value)

    # Phase 3: Filtering Parser Methods

    def parse_filter_between(self) -> FilterBetweenNode:
        """Parse: filter_between data with column="price" min=10 max=100 as alias"""
        self.advance()  # consume FILTER_BETWEEN
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.COLUMN)
        self.expect(TokenType.ASSIGN)
        column = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.MIN)
        self.expect(TokenType.ASSIGN)
        min_value = self.parse_value()
        self.expect(TokenType.MAX)
        self.expect(TokenType.ASSIGN)
        max_value = self.parse_value()
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FilterBetweenNode(source, column, min_value, max_value, new_alias)

    def parse_filter_isin(self) -> FilterIsInNode:
        """Parse: filter_isin data with column="category" values=["A", "B", "C"] as alias"""
        self.advance()  # consume FILTER_ISIN
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.COLUMN)
        self.expect(TokenType.ASSIGN)
        column = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.VALUES)
        self.expect(TokenType.ASSIGN)
        values = self.parse_list_value()
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FilterIsInNode(source, column, values, new_alias)

    def parse_filter_contains(self) -> FilterContainsNode:
        """Parse: filter_contains data with column="product" pattern="laptop" as alias"""
        self.advance()  # consume FILTER_CONTAINS
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.COLUMN)
        self.expect(TokenType.ASSIGN)
        column = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.PATTERN)
        self.expect(TokenType.ASSIGN)
        pattern = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FilterContainsNode(source, column, pattern, new_alias)

    def parse_filter_startswith(self) -> FilterStartsWithNode:
        """Parse: filter_startswith data with column="product" pattern="ABC" as alias"""
        self.advance()  # consume FILTER_STARTSWITH
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.COLUMN)
        self.expect(TokenType.ASSIGN)
        column = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.PATTERN)
        self.expect(TokenType.ASSIGN)
        pattern = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FilterStartsWithNode(source, column, pattern, new_alias)

    def parse_filter_endswith(self) -> FilterEndsWithNode:
        """Parse: filter_endswith data with column="product" pattern=".pdf" as alias"""
        self.advance()  # consume FILTER_ENDSWITH
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.COLUMN)
        self.expect(TokenType.ASSIGN)
        column = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.PATTERN)
        self.expect(TokenType.ASSIGN)
        pattern = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FilterEndsWithNode(source, column, pattern, new_alias)

    def parse_filter_regex(self) -> FilterRegexNode:
        """Parse: filter_regex data with column="email" pattern=".*@gmail\\.com" as alias"""
        self.advance()  # consume FILTER_REGEX
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.COLUMN)
        self.expect(TokenType.ASSIGN)
        column = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.PATTERN)
        self.expect(TokenType.ASSIGN)
        pattern = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FilterRegexNode(source, column, pattern, new_alias)

    def parse_filter_null(self) -> FilterNullNode:
        """Parse: filter_null data with column="discount" as alias"""
        self.advance()  # consume FILTER_NULL
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.COLUMN)
        self.expect(TokenType.ASSIGN)
        column = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FilterNullNode(source, column, new_alias)

    def parse_filter_notnull(self) -> FilterNotNullNode:
        """Parse: filter_notnull data with column="discount" as alias"""
        self.advance()  # consume FILTER_NOTNULL
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.COLUMN)
        self.expect(TokenType.ASSIGN)
        column = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FilterNotNullNode(source, column, new_alias)

    def parse_filter_duplicates(self) -> FilterDuplicatesNode:
        """Parse: filter_duplicates data with keep="first" as alias OR filter_duplicates data with subset=["col1"] keep="first" as alias"""
        self.advance()  # consume FILTER_DUPLICATES
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)

        subset = None
        keep = "first"  # default

        # Parse optional subset parameter
        if self.match(TokenType.SUBSET):
            self.advance()
            self.expect(TokenType.ASSIGN)
            subset = self.parse_list_value()

        # Parse keep parameter
        if self.match(TokenType.KEEP):
            self.advance()
            self.expect(TokenType.ASSIGN)
            keep = self.expect(TokenType.STRING_LITERAL).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FilterDuplicatesNode(source, subset, keep, new_alias)

    def parse_select(self) -> SelectNode:
        """
        Supports:
        - Classic: select df {col1, col2} as alias
        - Natural: select df with col1, col2 as alias
        """
        self.expect(TokenType.SELECT)
        source = self.expect(TokenType.IDENTIFIER).value

        # Detect syntax variant
        if self.match(TokenType.WITH):
            # Natural syntax
            self.advance()
            columns = self.parse_column_list_natural()
        elif self.match(TokenType.LBRACE):
            # Classic syntax
            columns = self.parse_column_list()
        else:
            raise SyntaxError(f"Expected 'with' or '{{' after select source")

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return SelectNode(source, columns, new_alias)
    
    def parse_filter(self) -> UpdatedFilterNode:
        """Parse: filter <source> where <condition> as <alias>"""
        self.expect(TokenType.FILTER)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WHERE)

        # Parse rich where clause (supports all filtering modes)
        condition = self.parse_where_clause()

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return UpdatedFilterNode(source, condition, new_alias)

    def parse_where_clause(self) -> CompoundConditionNode:
        """
        Parse WHERE clause with support for:
        - Simple comparisons: price > 100
        - Between: price between 50 and 200
        - In: category in ["A", "B", "C"]
        - String matching: description contains "premium"
        - Null checks: discount is null
        - Complex conditions: price > 100 and quantity < 10
        """
        return self.parse_or_condition()

    def parse_or_condition(self) -> CompoundConditionNode:
        """Parse OR conditions (lowest precedence)"""
        left = self.parse_and_condition()

        while self.match(TokenType.OR):
            self.advance()
            right = self.parse_and_condition()
            left = BinaryConditionNode(left, 'or', right)

        return left

    def parse_and_condition(self) -> CompoundConditionNode:
        """Parse AND conditions (higher precedence than OR)"""
        left = self.parse_not_condition()

        while self.match(TokenType.AND):
            self.advance()
            right = self.parse_not_condition()
            left = BinaryConditionNode(left, 'and', right)

        return left

    def parse_not_condition(self) -> CompoundConditionNode:
        """Parse NOT conditions (highest precedence)"""
        if self.match(TokenType.NOT):
            self.advance()
            condition = self.parse_not_condition()  # Allow chaining: not not condition
            return NotConditionNode(condition)

        return self.parse_primary_condition()

    def parse_primary_condition(self) -> CompoundConditionNode:
        """Parse primary (atomic) conditions"""
        # Check for parentheses
        if self.match(TokenType.LPAREN):
            self.advance()
            condition = self.parse_where_clause()
            self.expect(TokenType.RPAREN)
            return condition

        # Must start with a column name
        if not self.match(TokenType.IDENTIFIER):
            raise SyntaxError(f"Expected column name in condition, got {self.current_token()}")

        column = self.expect(TokenType.IDENTIFIER).value

        # Check what kind of condition this is
        if self.match(TokenType.BETWEEN):
            # BETWEEN condition: column between min and max
            self.advance()
            min_value = self.parse_value()
            self.expect(TokenType.AND)
            max_value = self.parse_value()
            return BetweenNode(column, min_value, max_value)

        elif self.match(TokenType.IN):
            # IN condition: column in [values]
            self.advance()
            values = self.parse_list_value()
            return InNode(column, values)

        elif self.match(TokenType.CONTAINS):
            # String matching: column contains "pattern"
            self.advance()
            pattern = self.expect(TokenType.STRING_LITERAL).value
            return StringMatchNode(column, 'contains', pattern)

        elif self.match(TokenType.STARTS_WITH):
            # String matching: column starts_with "pattern"
            self.advance()
            pattern = self.expect(TokenType.STRING_LITERAL).value
            return StringMatchNode(column, 'starts_with', pattern)

        elif self.match(TokenType.ENDS_WITH):
            # String matching: column ends_with "pattern"
            self.advance()
            pattern = self.expect(TokenType.STRING_LITERAL).value
            return StringMatchNode(column, 'ends_with', pattern)

        elif self.match(TokenType.MATCHES):
            # Regex matching: column matches "regex"
            self.advance()
            pattern = self.expect(TokenType.STRING_LITERAL).value
            return StringMatchNode(column, 'matches', pattern)

        elif self.match(TokenType.IS):
            # Null check: column is [not] null
            self.advance()
            is_not = False
            if self.match(TokenType.NOT):
                is_not = True
                self.advance()
            self.expect(TokenType.NULL)
            return NullCheckNode(column, is_not)

        elif self.match(TokenType.EQ, TokenType.NEQ, TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE):
            # Comparison: column op value
            op_token = self.current_token()
            self.advance()
            value = self.parse_value()

            # Map token types to operator strings
            op_map = {
                TokenType.EQ: '==',
                TokenType.NEQ: '!=',
                TokenType.LT: '<',
                TokenType.GT: '>',
                TokenType.LTE: '<=',
                TokenType.GTE: '>='
            }
            operator = op_map[op_token.type]
            return ComparisonNode(column, operator, value)

        else:
            raise SyntaxError(f"Expected comparison operator or keyword after column '{column}', got {self.current_token()}")
    
    def parse_sort(self) -> SortNode:
        """Parse: sort <source> by <column> [desc|asc] as <alias>"""
        self.expect(TokenType.SORT)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.BY)
        sort_specs = self.parse_sort_specs()
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return SortNode(source, sort_specs, new_alias)
    
    def parse_join(self) -> JoinNode:
        """Parse: join <df1> with <df2> on <column> as <alias>"""
        self.expect(TokenType.JOIN)
        alias1 = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        alias2 = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ON)
        join_column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return JoinNode(alias1, alias2, join_column, new_alias)
    
    def parse_groupby(self) -> GroupByNode:
        """Parse: groupby <source> by {cols} compute {funcs} as <alias>"""
        self.expect(TokenType.GROUPBY)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.BY)

        # Parse group columns (with braces or natural)
        if self.match(TokenType.LBRACE):
            group_columns = self.parse_column_list()
        else:
            # Natural syntax: by col or by col1, col2
            group_columns = self.parse_column_list_natural()

        # Parse aggregations (optional)
        aggregations = []
        if self.match(TokenType.COMPUTE):
            self.advance()
            aggregations = self.parse_aggregations()

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return GroupByNode(source, group_columns, aggregations, new_alias)
    
    def parse_sample(self) -> SampleNode:
        """Parse: sample <source> with n=<num> [random] [as <alias>]"""
        self.expect(TokenType.SAMPLE)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.N)
        self.expect(TokenType.ASSIGN)
        size = int(self.parse_value())

        # Check for random flag
        is_random = False
        if self.match(TokenType.RANDOM):
            is_random = True
            self.advance()

        # Make 'as' optional
        new_alias = None
        if self.match(TokenType.AS):
            self.advance()
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
        """Parse: fillna <source> column <col> with value=<val>|method=<method> as <alias>"""
        self.expect(TokenType.FILLNA)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        # Parse WITH clause for value or method
        self.expect(TokenType.WITH)

        fill_value = None
        method = None

        # Check if it's value= or method= (both are keywords in lexer)
        if self.match(TokenType.VALUE):
            # value= syntax
            self.advance()
            self.expect(TokenType.ASSIGN)
            fill_value = self.parse_value()
        elif self.match(TokenType.METHOD):
            # method= syntax
            self.advance()
            self.expect(TokenType.ASSIGN)
            method = self.parse_value()
        else:
            raise SyntaxError("Expected 'value' or 'method' after 'with'")

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value

        return FillNANode(source, column, new_alias, fill_value, method)
    
    def parse_mutate(self) -> MutateNode:
        self.expect(TokenType.MUTATE)
        source = self.expect(TokenType.IDENTIFIER).value

        # Check if using WITH syntax or brace syntax
        if self.match(TokenType.WITH):
            mutations = self.parse_mutations_with_syntax()
        else:
            mutations = self.parse_mutations()

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return MutateNode(source, mutations, new_alias)
    
    def parse_apply(self) -> ApplyNode:
        """Parse: apply <source> columns {cols} with function=<expr> as <alias>"""
        self.expect(TokenType.APPLY)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMNS)
        columns = self.parse_column_list()
        self.expect(TokenType.WITH)
        self.expect(TokenType.FUNCTION)
        self.expect(TokenType.ASSIGN)
        function_expr = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ApplyNode(source, columns, function_expr, new_alias)
    
    def parse_describe(self) -> DescribeNode:
        """Parse: describe <source> [columns {cols}]"""
        self.expect(TokenType.DESCRIBE)
        source = self.expect(TokenType.IDENTIFIER).value
        columns = None
        if self.match(TokenType.COLUMNS):
            self.advance()
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

    def parse_unique(self) -> 'UniqueNode':
        """Parse: unique <source> column <column>"""
        from noeta_ast import UniqueNode
        self.expect(TokenType.UNIQUE)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        return UniqueNode(source, column)

    def parse_value_counts(self) -> 'ValueCountsNode':
        """Parse: value_counts <source> column <column> [normalize] [ascending]"""
        from noeta_ast import ValueCountsNode
        self.expect(TokenType.VALUE_COUNTS)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        # Optional flags
        normalize = False
        ascending = False

        if self.match(TokenType.NORMALIZE):
            normalize = True
            self.advance()

        if self.match(TokenType.ASCENDING):
            ascending = True
            self.advance()

        return ValueCountsNode(source, column, normalize, ascending)

    def parse_outliers(self) -> OutliersNode:
        """Parse: outliers <source> with method=<method> columns {cols}"""
        self.expect(TokenType.OUTLIERS)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.METHOD)
        self.expect(TokenType.ASSIGN)
        method = self.parse_value()
        self.expect(TokenType.COLUMNS)
        columns = self.parse_column_list()
        return OutliersNode(source, method, columns)
    
    def parse_quantile(self) -> QuantileNode:
        """Parse: quantile <source> column <col> with q=<value>"""
        self.expect(TokenType.QUANTILE)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.Q)
        self.expect(TokenType.ASSIGN)
        q_value = float(self.parse_value())
        return QuantileNode(source, column, q_value)
    
    def parse_normalize(self) -> NormalizeNode:
        """Parse: normalize <source> columns {cols} with method=<method> as <alias>"""
        self.expect(TokenType.NORMALIZE)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMNS)
        columns = self.parse_column_list()
        self.expect(TokenType.WITH)
        self.expect(TokenType.METHOD)
        self.expect(TokenType.ASSIGN)
        method = self.parse_value()
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return NormalizeNode(source, columns, method, new_alias)
    
    def parse_binning(self) -> BinningNode:
        """Parse: binning <source> column <col> with bins=<num> as <alias>"""
        self.expect(TokenType.BINNING)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.BINS)
        self.expect(TokenType.ASSIGN)
        num_bins = int(self.parse_value())
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return BinningNode(source, column, num_bins, new_alias)
    
    def parse_rolling(self) -> RollingNode:
        """Parse: rolling <source> column <col> with window=<num> function=<func> as <alias>"""
        self.expect(TokenType.ROLLING)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.WINDOW)
        self.expect(TokenType.ASSIGN)
        window = int(self.parse_value())
        self.expect(TokenType.FUNCTION)
        self.expect(TokenType.ASSIGN)
        function = self.parse_value()
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
        """
        Parse: boxplot <source> columns {cols} OR boxplot <source> with <col> by <group_col>
        """
        self.expect(TokenType.BOXPLOT)
        source = self.expect(TokenType.IDENTIFIER).value

        columns = None
        value_column = None
        group_column = None

        # Detect syntax variant
        if self.match(TokenType.WITH):
            # Natural syntax: boxplot df with Age by Pclass
            self.advance()
            value_column = self.expect(TokenType.IDENTIFIER).value

            # Optional BY clause
            if self.match(TokenType.BY):
                self.advance()
                group_column = self.expect(TokenType.IDENTIFIER).value
        elif self.match(TokenType.COLUMNS):
            # Unified syntax: boxplot df columns {cols}
            self.advance()
            columns = self.parse_column_list()
        else:
            raise SyntaxError(f"Expected 'with' or 'columns' after boxplot source")

        return BoxPlotNode(source, columns, value_column, group_column)
    
    def parse_heatmap(self) -> HeatmapNode:
        """Parse: heatmap <source> columns {cols}"""
        self.expect(TokenType.HEATMAP)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMNS)
        columns = self.parse_column_list()
        return HeatmapNode(source, columns)

    def parse_pairplot(self) -> PairPlotNode:
        """Parse: pairplot <source> columns {cols}"""
        self.expect(TokenType.PAIRPLOT)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMNS)
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
        """Parse: pie <source> with values=<col> labels=<col>"""
        self.expect(TokenType.PIE)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.VALUES)
        self.expect(TokenType.ASSIGN)
        values_column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.LABELS)
        self.expect(TokenType.ASSIGN)
        labels_column = self.expect(TokenType.IDENTIFIER).value
        return PieChartNode(source, values_column, labels_column)
    
    def parse_save(self) -> SaveNode:
        """Parse: save <source> to <file> [with format=<fmt> <params>]"""
        self.expect(TokenType.SAVE)
        source_alias = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.TO)
        filepath = self.expect(TokenType.STRING_LITERAL).value

        # Parse optional WITH clause for format and parameters
        format_type = None
        params = None

        if self.match(TokenType.WITH):
            self.advance()
            params = {}

            # Parse parameters
            while self.current_token() and not self.match(TokenType.EOF):
                if not self.match(TokenType.IDENTIFIER):
                    break

                param_name = self.current_token().value
                self.advance()
                self.expect(TokenType.ASSIGN)
                param_value = self.parse_value()

                # Special handling for 'format' parameter
                if param_name == 'format':
                    format_type = param_value
                else:
                    params[param_name] = param_value

                # Break if we've consumed all parameters
                if not self.match(TokenType.IDENTIFIER):
                    break

        # Auto-detect format from file extension if not explicitly specified
        if format_type is None:
            filepath_lower = filepath.lower()
            if filepath_lower.endswith('.csv'):
                format_type = 'csv'
            elif filepath_lower.endswith('.json'):
                format_type = 'json'
            elif filepath_lower.endswith(('.xlsx', '.xls')):
                format_type = 'excel'
            elif filepath_lower.endswith('.parquet'):
                format_type = 'parquet'

        return SaveNode(source_alias, filepath, format_type, params)
    
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
    def parse_identifier_or_keyword(self) -> str:
        """Parse identifier or allow reserved keywords as column names."""
        token = self.current_token()
        if token.type == TokenType.IDENTIFIER:
            self.advance()
            return token.value
        # Allow common reserved keywords as column names
        elif token.type in [TokenType.TARGET, TokenType.INDEX, TokenType.COLUMN,
                            TokenType.VALUES, TokenType.N, TokenType.MIN,
                            TokenType.MAX, TokenType.METHOD, TokenType.SUBSET]:
            self.advance()
            # Get keyword text from token
            return token.value if hasattr(token, 'value') else token.type.name.lower()
        else:
            raise SyntaxError(f"Expected identifier or column name, got {token.type}")

    def parse_column_list(self) -> List[str]:
        """Parse: {col1, col2, col3} - allows reserved keywords as column names"""
        self.expect(TokenType.LBRACE)
        columns = []
        columns.append(self.parse_identifier_or_keyword())
        while self.match(TokenType.COMMA):
            self.advance()
            columns.append(self.parse_identifier_or_keyword())
        self.expect(TokenType.RBRACE)
        return columns

    def parse_column_list_natural(self) -> List[str]:
        """Parse comma-separated columns without braces - allows reserved keywords as column names"""
        columns = []
        columns.append(self.parse_identifier_or_keyword())

        while self.match(TokenType.COMMA):
            self.advance()
            columns.append(self.parse_identifier_or_keyword())

        return columns

    def parse_condition_natural(self) -> ConditionNode:
        """Parse condition without brackets, accepting = or =="""
        left = self.expect(TokenType.IDENTIFIER).value

        # Accept both ASSIGN (=) and comparison operators
        op_token = self.current_token()
        if op_token.type == TokenType.ASSIGN:
            operator = '=='  # Convert single = to ==
            self.advance()
        elif op_token.type in [TokenType.EQ, TokenType.NEQ, TokenType.LT,
                               TokenType.GT, TokenType.LTE, TokenType.GTE]:
            operator = op_token.value
            self.advance()
        else:
            raise SyntaxError(f"Expected comparison operator, got {op_token.type}")

        # Parse right operand
        right_token = self.current_token()
        if right_token.type == TokenType.IDENTIFIER:
            right = right_token.value
        elif right_token.type == TokenType.STRING_LITERAL:
            right = right_token.value
        elif right_token.type == TokenType.NUMERIC_LITERAL:
            right = right_token.value
        else:
            raise SyntaxError(f"Expected identifier or literal")
        self.advance()

        return ConditionNode(left, operator, right)

    def parse_condition(self) -> ConditionNode:
        left = self.expect(TokenType.IDENTIFIER).value

        # Parse operator - accept both = and ==
        op_token = self.current_token()
        if op_token.type == TokenType.ASSIGN:
            operator = '=='  # Convert single = to ==
            self.advance()
        elif op_token.type in [TokenType.EQ, TokenType.NEQ, TokenType.LT,
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
        elif self.match(TokenType.ASC):
            direction = 'ASC'
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
            elif self.match(TokenType.ASC):
                direction = 'ASC'
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

    def parse_mutations_with_syntax(self) -> List[MutationNode]:
        """Parse mutations using WITH keyword syntax: WITH col = expr [WITH col = expr ...]"""
        mutations = []

        # Parse first mutation: WITH col = expr
        self.expect(TokenType.WITH)
        new_column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ASSIGN)
        expression = self.parse_expression()
        mutations.append(MutationNode(new_column, expression))

        # Parse additional mutations: WITH col = expr
        while self.match(TokenType.WITH):
            self.advance()
            new_column = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.ASSIGN)
            expression = self.parse_expression()
            mutations.append(MutationNode(new_column, expression))

        return mutations

    def parse_expression(self) -> str:
        """Parse an expression and return it as a string for pandas eval()"""
        expr_tokens = []

        # Parse the expression until we hit AS keyword or WITH keyword
        while not self.match(TokenType.AS, TokenType.WITH) and self.pos < len(self.tokens):
            token = self.current_token()

            if token.type == TokenType.IDENTIFIER:
                expr_tokens.append(token.value)
                self.advance()
            elif token.type == TokenType.NUMERIC_LITERAL:
                expr_tokens.append(str(token.value))
                self.advance()
            elif token.type == TokenType.STRING_LITERAL:
                expr_tokens.append(f'"{token.value}"')
                self.advance()
            elif token.type in [TokenType.PLUS, TokenType.MINUS, TokenType.STAR,
                               TokenType.SLASH, TokenType.PERCENT]:
                expr_tokens.append(token.value)
                self.advance()
            elif token.type in [TokenType.EQ, TokenType.NEQ, TokenType.LT,
                               TokenType.GT, TokenType.LTE, TokenType.GTE]:
                expr_tokens.append(token.value)
                self.advance()
            else:
                break

        # Join tokens with spaces for readability
        return ' '.join(expr_tokens)

    def parse_params(self) -> dict:
        """
        Parse parameter list: param1=value1 param2=value2 ...
        Stops when it hits 'as' or end of tokens
        """
        params = {}

        while self.current_token() and not self.match(TokenType.AS, TokenType.EOF):
            # Parameter name (identifier or keyword)
            param_token = self.current_token()
            if not param_token:
                break

            # Check if this is a parameter keyword
            if param_token.type == TokenType.IDENTIFIER:
                param_name = param_token.value
            elif param_token.type in [TokenType.DELIMITER, TokenType.ENCODING, TokenType.HEADER,
                                     TokenType.NAMES, TokenType.USECOLS, TokenType.DTYPE,
                                     TokenType.SKIPROWS, TokenType.NROWS, TokenType.NA_VALUES,
                                     TokenType.THOUSANDS, TokenType.DECIMAL, TokenType.COMMENT,
                                     TokenType.SKIP_BLANK_LINES, TokenType.PARSE_DATES,
                                     TokenType.DATE_FORMAT, TokenType.CHUNKSIZE, TokenType.COMPRESSION,
                                     TokenType.LOW_MEMORY, TokenType.MEMORY_MAP, TokenType.ORIENT,
                                     TokenType.TYP, TokenType.CONVERT_AXES, TokenType.CONVERT_DATES,
                                     TokenType.PRECISE_FLOAT, TokenType.DATE_UNIT, TokenType.LINES,
                                     TokenType.SHEET, TokenType.SHEET_NAME, TokenType.INDEX_COL,
                                     TokenType.ENGINE, TokenType.CONVERTERS, TokenType.SKIPFOOTER,
                                     TokenType.FILTERS, TokenType.USE_NULLABLE_DTYPES,
                                     TokenType.STORAGE_OPTIONS, TokenType.PARAMS, TokenType.COERCE_FLOAT,
                                     TokenType.INDEX, TokenType.INDEX_LABEL, TokenType.NA_REP,
                                     TokenType.MODE, TokenType.QUOTING, TokenType.QUOTECHAR,
                                     TokenType.ESCAPECHAR, TokenType.LINETERMINATOR, TokenType.FLOAT_FORMAT]:
                param_name = param_token.value
            else:
                # Not a parameter, stop parsing
                break

            self.advance()

            # Expect '='
            if not self.match(TokenType.ASSIGN):
                raise SyntaxError(f"Expected '=' after parameter '{param_name}'")
            self.advance()

            # Parse value
            value = self.parse_value()
            params[param_name] = value

        return params

    def parse_value(self):
        """
        Parse a value: string, number, boolean, list, dict, or identifier
        """
        token = self.current_token()
        if not token:
            raise SyntaxError("Expected value")

        # String literal
        if token.type == TokenType.STRING_LITERAL:
            value = token.value
            self.advance()
            return value

        # Numeric literal
        elif token.type == TokenType.NUMERIC_LITERAL:
            value = token.value
            self.advance()
            return value

        # Boolean
        elif token.type == TokenType.IDENTIFIER and token.value.lower() in ['true', 'false']:
            value = token.value.lower() == 'true'
            self.advance()
            return value

        # None/null
        elif token.type == TokenType.IDENTIFIER and token.value.lower() in ['none', 'null']:
            self.advance()
            return None

        # List
        elif token.type == TokenType.LBRACKET:
            return self.parse_list_value()

        # Dict
        elif token.type == TokenType.LBRACE:
            return self.parse_dict_value()

        # Identifier (for column names, etc.)
        elif token.type == TokenType.IDENTIFIER:
            value = token.value
            self.advance()
            return value

        else:
            raise SyntaxError(f"Unexpected token type for value: {token.type}")

    def parse_list_value(self) -> list:
        """Parse a list: [val1, val2, val3]"""
        self.expect(TokenType.LBRACKET)
        values = []

        # Handle empty list
        if self.match(TokenType.RBRACKET):
            self.advance()
            return values

        # Parse first value
        values.append(self.parse_value())

        # Parse additional values
        while self.match(TokenType.COMMA):
            self.advance()
            # Allow trailing comma
            if self.match(TokenType.RBRACKET):
                break
            values.append(self.parse_value())

        self.expect(TokenType.RBRACKET)
        return values

    def parse_dict_value(self) -> dict:
        """Parse a dictionary: {key1: val1, key2: val2}"""
        self.expect(TokenType.LBRACE)
        result = {}

        # Handle empty dict
        if self.match(TokenType.RBRACE):
            self.advance()
            return result

        # Parse first key-value pair
        key = self.expect(TokenType.STRING_LITERAL).value if self.match(TokenType.STRING_LITERAL) else self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLON)
        value = self.parse_value()
        result[key] = value

        # Parse additional key-value pairs
        while self.match(TokenType.COMMA):
            self.advance()
            # Allow trailing comma
            if self.match(TokenType.RBRACE):
                break
            key = self.expect(TokenType.STRING_LITERAL).value if self.match(TokenType.STRING_LITERAL) else self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.COLON)
            value = self.parse_value()
            result[key] = value

        self.expect(TokenType.RBRACE)
        return result

    # ============================================================
    # EXPRESSION AND PARAMETER PARSERS
    # ============================================================

    def parse_expression(self) -> 'ExpressionNode':
        """
        Parse DSL expressions for apply/map operations.
        Supports:
        - Arithmetic: value * 1.1, price + tax
        - Logical: value > 100 and value < 200
        - Function calls: is_numeric(value), round(value, 2)
        - Conditionals: value * 1.1 where is_numeric(value) else value

        Grammar (with precedence from lowest to highest):
        expression     := conditional
        conditional    := logical_or ( WHERE logical_or ELSE logical_or )?
        logical_or     := logical_and ( OR logical_and )*
        logical_and    := comparison ( AND comparison )*
        comparison     := additive ( (== | != | < | > | <= | >=) additive )?
        additive       := multiplicative ( (+ | -) multiplicative )*
        multiplicative := power ( (* | / | %) power )*
        power          := unary ( ** unary )*
        unary          := (- | NOT) unary | primary
        primary        := NUMBER | STRING | IDENTIFIER | function_call | ( expression )
        function_call  := IDENTIFIER LPAREN ( expression ( COMMA expression )* )? RPAREN
        """
        return self.parse_conditional()

    def parse_conditional(self) -> 'ExpressionNode':
        """Parse conditional: expr where condition else expr"""
        from noeta_ast import ConditionalExprNode

        expr = self.parse_logical_or()

        if self.match(TokenType.WHERE):
            self.advance()
            condition = self.parse_logical_or()
            self.expect(TokenType.ELSE)
            else_expr = self.parse_logical_or()
            return ConditionalExprNode(condition, expr, else_expr)

        return expr

    def parse_logical_or(self) -> 'ExpressionNode':
        """Parse logical OR: expr or expr or ..."""
        from noeta_ast import BinaryOpNode

        left = self.parse_logical_and()

        while self.match(TokenType.OR):
            self.advance()
            right = self.parse_logical_and()
            left = BinaryOpNode(left, 'or', right)

        return left

    def parse_logical_and(self) -> 'ExpressionNode':
        """Parse logical AND: expr and expr and ..."""
        from noeta_ast import BinaryOpNode

        left = self.parse_comparison()

        while self.match(TokenType.AND):
            self.advance()
            right = self.parse_comparison()
            left = BinaryOpNode(left, 'and', right)

        return left

    def parse_comparison(self) -> 'ExpressionNode':
        """Parse comparison: expr == expr, expr < expr, etc."""
        from noeta_ast import BinaryOpNode

        left = self.parse_additive()

        if self.match(TokenType.EQ, TokenType.NEQ, TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE):
            op_token = self.current_token()
            self.advance()
            right = self.parse_additive()

            # Map token types to operator strings
            op_map = {
                TokenType.EQ: '==',
                TokenType.NEQ: '!=',
                TokenType.LT: '<',
                TokenType.GT: '>',
                TokenType.LTE: '<=',
                TokenType.GTE: '>='
            }
            op = op_map.get(op_token.type, '==')
            return BinaryOpNode(left, op, right)

        return left

    def parse_additive(self) -> 'ExpressionNode':
        """Parse addition/subtraction: expr + expr, expr - expr"""
        from noeta_ast import BinaryOpNode

        left = self.parse_multiplicative()

        while self.match(TokenType.PLUS, TokenType.MINUS):
            op_token = self.current_token()
            self.advance()
            right = self.parse_multiplicative()
            op = '+' if op_token.type == TokenType.PLUS else '-'
            left = BinaryOpNode(left, op, right)

        return left

    def parse_multiplicative(self) -> 'ExpressionNode':
        """Parse multiplication/division/modulo: expr * expr, expr / expr, expr % expr"""
        from noeta_ast import BinaryOpNode

        left = self.parse_power()

        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO):
            op_token = self.current_token()
            self.advance()
            right = self.parse_power()

            op_map = {
                TokenType.MULTIPLY: '*',
                TokenType.DIVIDE: '/',
                TokenType.MODULO: '%'
            }
            op = op_map.get(op_token.type, '*')
            left = BinaryOpNode(left, op, right)

        return left

    def parse_power(self) -> 'ExpressionNode':
        """Parse exponentiation: expr ** expr"""
        from noeta_ast import BinaryOpNode

        left = self.parse_unary()

        if self.match(TokenType.EXPONENT):
            self.advance()
            right = self.parse_power()  # Right-associative
            return BinaryOpNode(left, '**', right)

        return left

    def parse_unary(self) -> 'ExpressionNode':
        """Parse unary operators: -expr, not expr"""
        from noeta_ast import UnaryOpNode

        if self.match(TokenType.MINUS):
            self.advance()
            expr = self.parse_unary()
            return UnaryOpNode('-', expr)

        if self.match(TokenType.NOT):
            self.advance()
            expr = self.parse_unary()
            return UnaryOpNode('not', expr)

        return self.parse_primary()

    def parse_primary(self) -> 'ExpressionNode':
        """Parse primary expressions: literals, identifiers, function calls, parenthesized expressions"""
        from noeta_ast import LiteralNode, IdentifierNode, FunctionCallNode

        token = self.current_token()

        # Numeric literal
        if token.type == TokenType.NUMERIC_LITERAL:
            self.advance()
            return LiteralNode(token.value)

        # String literal
        if token.type == TokenType.STRING_LITERAL:
            self.advance()
            return LiteralNode(token.value)

        # Boolean literal
        if token.type == TokenType.BOOLEAN_LITERAL:
            self.advance()
            return LiteralNode(token.value.lower() == 'true')

        # Null literal
        if token.type == TokenType.NULL:
            self.advance()
            return LiteralNode(None)

        # Identifier or function call
        if token.type == TokenType.IDENTIFIER:
            name = token.value
            self.advance()

            # Function call
            if self.match(TokenType.LPAREN):
                self.advance()
                args = []

                # Parse arguments
                if not self.match(TokenType.RPAREN):
                    args.append(self.parse_expression())

                    while self.match(TokenType.COMMA):
                        self.advance()
                        args.append(self.parse_expression())

                self.expect(TokenType.RPAREN)
                return FunctionCallNode(name, args)

            # Plain identifier
            return IdentifierNode(name)

        # Parenthesized expression
        if token.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr

        raise SyntaxError(f"Unexpected token in expression: {token}")

    def parse_with_clause(self) -> dict:
        """
        Parse unified WITH clause for parameters.
        Examples:
        - with n=100
        - with method="mean"
        - with decimals=2 fill_value=0
        - with transform value * 1.1
        - with mapping {"A": 1, "B": 2}

        Returns dict of parameter name -> value
        """
        params = {}

        if not self.match(TokenType.WITH):
            return params

        self.advance()  # consume WITH

        # Special case: transform parameter (for apply/map)
        if self.match(TokenType.TRANSFORM):
            self.advance()
            params['transform'] = self.parse_expression()
            return params

        # Standard key=value parameters
        while True:
            # Check if we've reached the end (AS, WHERE, etc.)
            if self.match(TokenType.AS, TokenType.WHERE, TokenType.BY, TokenType.COMPUTE):
                break

            # Check for end of tokens
            if not self.current_token():
                break

            # Expect parameter name
            if not self.match(TokenType.IDENTIFIER):
                break

            param_name = self.current_token().value
            self.advance()

            # Expect equals sign
            if not self.match(TokenType.ASSIGN):
                raise SyntaxError(f"Expected '=' after parameter name '{param_name}'")
            self.advance()

            # Parse parameter value
            params[param_name] = self.parse_value()

            # If no more parameters, break
            if not self.match(TokenType.IDENTIFIER):
                break

        return params

    # ============================================================
    # PHASE 4: TRANSFORMATION OPERATIONS - PARSERS
    # ============================================================

    # Phase 4A: Math Operations
    def parse_round(self) -> 'RoundNode':
        """Parse: round data column price decimals=2 as rounded"""
        from noeta_ast import RoundNode
        self.advance()  # consume ROUND
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        # Optional decimals parameter
        decimals = 0
        if self.match(TokenType.DECIMALS):
            self.advance()
            self.expect(TokenType.ASSIGN)
            decimals = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return RoundNode(source, column, new_alias, decimals)

    def parse_abs(self) -> 'AbsNode':
        """Parse: abs data column delta as absolute"""
        from noeta_ast import AbsNode
        self.advance()  # consume ABS
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return AbsNode(source, column, new_alias)

    def parse_sqrt(self) -> 'SqrtNode':
        """Parse: sqrt data column area as sqrt_area"""
        from noeta_ast import SqrtNode
        self.advance()  # consume SQRT
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return SqrtNode(source, column, new_alias)

    def parse_power(self) -> 'PowerNode':
        """Parse: power data column value exponent=2 as squared"""
        from noeta_ast import PowerNode
        self.advance()  # consume POWER
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.EXPONENT)
        self.expect(TokenType.ASSIGN)
        exponent = float(self.expect(TokenType.NUMERIC_LITERAL).value)
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return PowerNode(source, column, new_alias, exponent)

    def parse_log(self) -> 'LogNode':
        """Parse: log data column value base=10 as log_values"""
        from noeta_ast import LogNode
        self.advance()  # consume LOG
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        # Optional base parameter (default "e")
        base = "e"
        if self.match(TokenType.BASE):
            self.advance()
            self.expect(TokenType.ASSIGN)
            if self.match(TokenType.NUMERIC_LITERAL):
                base = str(int(self.expect(TokenType.NUMERIC_LITERAL).value))
            elif self.match(TokenType.IDENTIFIER):
                base = self.expect(TokenType.IDENTIFIER).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return LogNode(source, column, new_alias, base)

    def parse_ceil(self) -> 'CeilNode':
        """Parse: ceil data column price as rounded_up"""
        from noeta_ast import CeilNode
        self.advance()  # consume CEIL
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return CeilNode(source, column, new_alias)

    def parse_floor(self) -> 'FloorNode':
        """Parse: floor data column price as rounded_down"""
        from noeta_ast import FloorNode
        self.advance()  # consume FLOOR
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FloorNode(source, column, new_alias)

    # Phase 4B: String Operations
    def parse_upper(self) -> 'UpperNode':
        """Parse: upper data column name as uppercase"""
        from noeta_ast import UpperNode
        self.advance()  # consume UPPER
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return UpperNode(source, column, new_alias)

    def parse_lower(self) -> 'LowerNode':
        """Parse: lower data column email as lowercase"""
        from noeta_ast import LowerNode
        self.advance()  # consume LOWER
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return LowerNode(source, column, new_alias)

    def parse_strip(self) -> 'StripNode':
        """Parse: strip data column text as trimmed"""
        from noeta_ast import StripNode
        self.advance()  # consume STRIP
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return StripNode(source, column, new_alias)

    def parse_replace(self) -> 'ReplaceNode':
        """Parse: replace data column name old="Mr." new="Mr" as cleaned"""
        from noeta_ast import ReplaceNode
        self.advance()  # consume REPLACE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.OLD)
        self.expect(TokenType.ASSIGN)
        old = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.NEW)
        self.expect(TokenType.ASSIGN)
        new = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ReplaceNode(source, column, new_alias, old, new)

    def parse_split(self) -> 'SplitNode':
        """Parse: split data column fullname delimiter=" " as name_parts"""
        from noeta_ast import SplitNode
        self.advance()  # consume SPLIT
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        # Optional delimiter parameter (default " ")
        delimiter = " "
        if self.match(TokenType.DELIMITER):
            self.advance()
            self.expect(TokenType.ASSIGN)
            delimiter = self.expect(TokenType.STRING_LITERAL).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return SplitNode(source, column, new_alias, delimiter)

    def parse_concat(self) -> 'ConcatNode':
        """Parse: concat data columns ["first", "last"] separator=" " as fullname"""
        from noeta_ast import ConcatNode
        self.advance()  # consume CONCAT
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMNS)
        columns = self.parse_list_value()

        # Optional separator
        separator = ""
        if self.match(TokenType.SEPARATOR):
            self.advance()
            self.expect(TokenType.ASSIGN)
            separator = self.expect(TokenType.STRING_LITERAL).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ConcatNode(source, columns, new_alias, separator)

    def parse_substring(self) -> 'SubstringNode':
        """Parse: substring data column text start=0 end=10 as substring"""
        from noeta_ast import SubstringNode
        self.advance()  # consume SUBSTRING
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.START)
        self.expect(TokenType.ASSIGN)
        start = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        # Optional end parameter
        end = None
        if self.match(TokenType.END):
            self.advance()
            self.expect(TokenType.ASSIGN)
            end = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return SubstringNode(source, column, new_alias, start, end)

    def parse_length(self) -> 'LengthNode':
        """Parse: length data column text as text_length"""
        from noeta_ast import LengthNode
        self.advance()  # consume LENGTH
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return LengthNode(source, column, new_alias)

    # Phase 4C: Date Operations
    def parse_parse_datetime(self) -> 'ParseDatetimeNode':
        """Parse: parse_datetime data column date_string format="%Y-%m-%d" as parsed"""
        from noeta_ast import ParseDatetimeNode
        self.advance()  # consume PARSE_DATETIME
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        # Optional format parameter
        format_str = None
        if self.match(TokenType.FORMAT):
            self.advance()
            self.expect(TokenType.ASSIGN)
            format_str = self.expect(TokenType.STRING_LITERAL).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ParseDatetimeNode(source, column, new_alias, format_str)

    def parse_extract_year(self) -> 'ExtractYearNode':
        """Parse: extract_year data column timestamp as year"""
        from noeta_ast import ExtractYearNode
        self.advance()  # consume EXTRACT_YEAR
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ExtractYearNode(source, column, new_alias)

    def parse_extract_month(self) -> 'ExtractMonthNode':
        """Parse: extract_month data column timestamp as month"""
        from noeta_ast import ExtractMonthNode
        self.advance()  # consume EXTRACT_MONTH
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ExtractMonthNode(source, column, new_alias)

    def parse_extract_day(self) -> 'ExtractDayNode':
        """Parse: extract_day data column timestamp as day"""
        from noeta_ast import ExtractDayNode
        self.advance()  # consume EXTRACT_DAY
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ExtractDayNode(source, column, new_alias)

    def parse_date_diff(self) -> 'DateDiffNode':
        """Parse: date_diff data start=start_date end=end_date unit="days" as duration"""
        from noeta_ast import DateDiffNode
        self.advance()  # consume DATE_DIFF
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.START)
        self.expect(TokenType.ASSIGN)
        start_column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.END)
        self.expect(TokenType.ASSIGN)
        end_column = self.expect(TokenType.IDENTIFIER).value

        # Optional unit parameter (default "days")
        unit = "days"
        if self.match(TokenType.UNIT):
            self.advance()
            self.expect(TokenType.ASSIGN)
            unit = self.expect(TokenType.STRING_LITERAL).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return DateDiffNode(source, start_column, end_column, new_alias, unit)

    # Phase 4D: Type Operations
    def parse_astype(self) -> 'AsTypeNode':
        """Parse: astype data column age dtype="int32" as converted"""
        from noeta_ast import AsTypeNode
        self.advance()  # consume ASTYPE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        # Optional dtype parameter (default "str")
        dtype = "str"
        if self.match(TokenType.DTYPE):
            self.advance()
            self.expect(TokenType.ASSIGN)
            dtype = self.expect(TokenType.STRING_LITERAL).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return AsTypeNode(source, column, new_alias, dtype)

    def parse_to_numeric(self) -> 'ToNumericNode':
        """Parse: to_numeric data column value errors="coerce" as numeric"""
        from noeta_ast import ToNumericNode
        self.advance()  # consume TO_NUMERIC
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        # Optional errors parameter
        errors = "raise"
        if self.match(TokenType.ERRORS):
            self.advance()
            self.expect(TokenType.ASSIGN)
            errors = self.expect(TokenType.STRING_LITERAL).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ToNumericNode(source, column, new_alias, errors)

    # Phase 4E: Encoding Operations
    def parse_one_hot_encode(self) -> 'OneHotEncodeNode':
        """Parse: one_hot_encode data column category as encoded"""
        from noeta_ast import OneHotEncodeNode
        self.advance()  # consume ONE_HOT_ENCODE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return OneHotEncodeNode(source, column, new_alias)

    def parse_label_encode(self) -> 'LabelEncodeNode':
        """Parse: label_encode data column status as encoded"""
        from noeta_ast import LabelEncodeNode
        self.advance()  # consume LABEL_ENCODE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return LabelEncodeNode(source, column, new_alias)

    # Phase 4F: Scaling Operations
    def parse_standard_scale(self) -> 'StandardScaleNode':
        """Parse: standard_scale data column price as scaled"""
        from noeta_ast import StandardScaleNode
        self.advance()  # consume STANDARD_SCALE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return StandardScaleNode(source, column, new_alias)

    def parse_minmax_scale(self) -> 'MinMaxScaleNode':
        """Parse: minmax_scale data column score as normalized"""
        from noeta_ast import MinMaxScaleNode
        self.advance()  # consume MINMAX_SCALE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return MinMaxScaleNode(source, column, new_alias)

    # ============================================================
    # PHASE 5: CLEANING OPERATIONS - PARSERS
    # ============================================================

    def parse_isnull(self) -> 'IsNullNode':
        """Parse: isnull data column age as missing_mask"""
        from noeta_ast import IsNullNode
        self.advance()  # consume ISNULL
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return IsNullNode(source, column, new_alias)

    def parse_notnull(self) -> 'NotNullNode':
        """Parse: notnull data column age as has_value"""
        from noeta_ast import NotNullNode
        self.advance()  # consume NOTNULL
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return NotNullNode(source, column, new_alias)

    def parse_count_na(self) -> 'CountNANode':
        """Parse: count_na data"""
        from noeta_ast import CountNANode
        self.advance()  # consume COUNT_NA
        source = self.expect(TokenType.IDENTIFIER).value
        return CountNANode(source)

    def parse_fill_forward(self) -> 'FillForwardNode':
        """Parse: fill_forward data column value as filled"""
        from noeta_ast import FillForwardNode
        self.advance()  # consume FILL_FORWARD
        source = self.expect(TokenType.IDENTIFIER).value

        # Optional column parameter
        column = None
        if self.match(TokenType.COLUMN):
            self.advance()
            column = self.expect(TokenType.IDENTIFIER).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FillForwardNode(source, new_alias, column)

    def parse_fill_backward(self) -> 'FillBackwardNode':
        """Parse: fill_backward data column value as filled"""
        from noeta_ast import FillBackwardNode
        self.advance()  # consume FILL_BACKWARD
        source = self.expect(TokenType.IDENTIFIER).value

        # Optional column parameter
        column = None
        if self.match(TokenType.COLUMN):
            self.advance()
            column = self.expect(TokenType.IDENTIFIER).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FillBackwardNode(source, new_alias, column)

    def parse_fill_mean(self) -> 'FillMeanNode':
        """Parse: fill_mean data column age as filled"""
        from noeta_ast import FillMeanNode
        self.advance()  # consume FILL_MEAN
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FillMeanNode(source, column, new_alias)

    def parse_fill_median(self) -> 'FillMedianNode':
        """Parse: fill_median data column salary as filled"""
        from noeta_ast import FillMedianNode
        self.advance()  # consume FILL_MEDIAN
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FillMedianNode(source, column, new_alias)

    def parse_interpolate(self) -> 'InterpolateNode':
        """Parse: interpolate data column timeseries method="linear" as interpolated"""
        from noeta_ast import InterpolateNode
        self.advance()  # consume INTERPOLATE
        source = self.expect(TokenType.IDENTIFIER).value

        # Optional column parameter
        column = None
        if self.match(TokenType.COLUMN):
            self.advance()
            column = self.expect(TokenType.IDENTIFIER).value

        # Optional method parameter
        method = "linear"
        if self.match(TokenType.METHOD):
            self.advance()
            self.expect(TokenType.ASSIGN)
            method = self.expect(TokenType.STRING_LITERAL).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return InterpolateNode(source, new_alias, column, method)

    def parse_duplicated(self) -> 'DuplicatedNode':
        """Parse: duplicated data columns ["email"] keep="first" as is_dup"""
        from noeta_ast import DuplicatedNode
        self.advance()  # consume DUPLICATED
        source = self.expect(TokenType.IDENTIFIER).value

        # Optional columns parameter
        columns = None
        if self.match(TokenType.COLUMNS):
            self.advance()
            columns = self.parse_list_value()

        # Optional keep parameter
        keep = "first"
        if self.match(TokenType.KEEP):
            self.advance()
            self.expect(TokenType.ASSIGN)
            keep = self.expect(TokenType.STRING_LITERAL).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return DuplicatedNode(source, new_alias, columns, keep)

    def parse_count_duplicates(self) -> 'CountDuplicatesNode':
        """Parse: count_duplicates data columns ["email"]"""
        from noeta_ast import CountDuplicatesNode
        self.advance()  # consume COUNT_DUPLICATES
        source = self.expect(TokenType.IDENTIFIER).value

        # Optional columns parameter
        columns = None
        if self.match(TokenType.COLUMNS):
            self.advance()
            columns = self.parse_list_value()

        return CountDuplicatesNode(source, columns)

    def parse_drop_duplicates(self) -> 'DropDuplicatesNode':
        """Parse: drop_duplicates data subset=["col1", "col2"] keep="first" as deduped"""
        from noeta_ast import DropDuplicatesNode
        self.advance()  # consume DROP_DUPLICATES
        source = self.expect(TokenType.IDENTIFIER).value

        subset = None
        keep = "first"

        if self.match(TokenType.SUBSET):
            self.advance()
            self.expect(TokenType.ASSIGN)
            subset = self.parse_list_value()

        if self.match(TokenType.KEEP):
            self.advance()
            self.expect(TokenType.ASSIGN)
            keep = self.expect(TokenType.STRING_LITERAL).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return DropDuplicatesNode(source, new_alias, subset, keep)

    def parse_fill_mode(self) -> 'FillModeNode':
        """Parse: fill_mode data column category as filled"""
        from noeta_ast import FillModeNode
        self.advance()  # consume FILL_MODE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FillModeNode(source, column, new_alias)

    def parse_qcut(self) -> 'QcutNode':
        """Parse: qcut data column price q=4 labels=["Q1","Q2","Q3","Q4"] as quantiled"""
        from noeta_ast import QcutNode
        self.advance()  # consume QCUT
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.Q)
        self.expect(TokenType.ASSIGN)
        q = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        labels = None
        if self.match(TokenType.LABELS):
            self.advance()
            self.expect(TokenType.ASSIGN)
            labels = self.parse_list_value()

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return QcutNode(source, column, q, new_alias, labels)

    # ============================================================
    # PHASE 6: DATA ORDERING OPERATIONS - PARSERS
    # ============================================================

    def parse_sort_index(self) -> 'SortIndexNode':
        """Parse: sort_index data ascending=true as sorted"""
        from noeta_ast import SortIndexNode
        self.advance()  # consume SORT_INDEX
        source = self.expect(TokenType.IDENTIFIER).value

        # Optional ascending parameter
        ascending = True
        if self.match(TokenType.ASCENDING):
            self.advance()
            self.expect(TokenType.ASSIGN)
            asc_val = self.parse_value()
            ascending = asc_val if isinstance(asc_val, bool) else str(asc_val).lower() == 'true'

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return SortIndexNode(source, new_alias, ascending)

    def parse_rank(self) -> 'RankNode':
        """Parse: rank data column score method="dense" ascending=true pct=false as ranked"""
        from noeta_ast import RankNode
        self.advance()  # consume RANK
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        # Optional parameters
        method = "average"
        ascending = True
        pct = False

        if self.match(TokenType.METHOD):
            self.advance()
            self.expect(TokenType.ASSIGN)
            method = self.expect(TokenType.STRING_LITERAL).value

        if self.match(TokenType.ASCENDING):
            self.advance()
            self.expect(TokenType.ASSIGN)
            asc_val = self.parse_value()
            ascending = asc_val if isinstance(asc_val, bool) else str(asc_val).lower() == 'true'

        if self.match(TokenType.PCT):
            self.advance()
            self.expect(TokenType.ASSIGN)
            pct_val = self.parse_value()
            pct = pct_val if isinstance(pct_val, bool) else str(pct_val).lower() == 'true'

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return RankNode(source, column, new_alias, method, ascending, pct)

    # ============================================================
    # PHASE 7: AGGREGATION & GROUPING OPERATIONS - PARSERS
    # ============================================================

    def parse_filter_groups(self) -> 'FilterGroupsNode':
        """Parse: filter_groups data by ["category"] condition="count > 5" as filtered"""
        from noeta_ast import FilterGroupsNode
        self.advance()  # consume FILTER_GROUPS
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.BY)
        group_columns = self.parse_list_value()
        self.expect(TokenType.CONDITION)
        self.expect(TokenType.ASSIGN)
        condition = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FilterGroupsNode(source, group_columns, condition, new_alias)

    def parse_group_transform(self) -> 'GroupTransformNode':
        """Parse: group_transform data by ["category"] column value function="mean" as transformed"""
        from noeta_ast import GroupTransformNode
        self.advance()  # consume GROUP_TRANSFORM
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.BY)
        group_columns = self.parse_list_value()
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.FUNCTION)
        self.expect(TokenType.ASSIGN)
        function = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return GroupTransformNode(source, group_columns, column, function, new_alias)

    def parse_window_rank(self) -> 'WindowRankNode':
        """Parse: window_rank data column score by ["category"] method="rank" as ranked"""
        from noeta_ast import WindowRankNode
        self.advance()  # consume WINDOW_RANK
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        partition_by = None
        if self.match(TokenType.BY):
            self.advance()
            partition_by = self.parse_list_value()

        method = "rank"
        if self.match(TokenType.METHOD):
            self.advance()
            self.expect(TokenType.ASSIGN)
            method = self.expect(TokenType.STRING_LITERAL).value

        ascending = True
        if self.match(TokenType.ASCENDING):
            self.advance()
            self.expect(TokenType.ASSIGN)
            asc_val = self.parse_value()
            ascending = asc_val if isinstance(asc_val, bool) else str(asc_val).lower() == 'true'

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return WindowRankNode(source, column, partition_by, new_alias, method, ascending)

    def parse_window_lag(self) -> 'WindowLagNode':
        """Parse: window_lag data column value periods=1 by ["category"] as lagged"""
        from noeta_ast import WindowLagNode
        self.advance()  # consume WINDOW_LAG
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.PERIODS)
        self.expect(TokenType.ASSIGN)
        periods = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        partition_by = None
        if self.match(TokenType.BY):
            self.advance()
            partition_by = self.parse_list_value()

        fill_value = None
        if self.match(TokenType.FILL_VALUE):
            self.advance()
            self.expect(TokenType.ASSIGN)
            fill_value = self.parse_value()

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return WindowLagNode(source, column, periods, new_alias, partition_by, fill_value)

    def parse_window_lead(self) -> 'WindowLeadNode':
        """Parse: window_lead data column value periods=1 by ["category"] as lead"""
        from noeta_ast import WindowLeadNode
        self.advance()  # consume WINDOW_LEAD
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.PERIODS)
        self.expect(TokenType.ASSIGN)
        periods = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        partition_by = None
        if self.match(TokenType.BY):
            self.advance()
            partition_by = self.parse_list_value()

        fill_value = None
        if self.match(TokenType.FILL_VALUE):
            self.advance()
            self.expect(TokenType.ASSIGN)
            fill_value = self.parse_value()

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return WindowLeadNode(source, column, periods, new_alias, partition_by, fill_value)

    def parse_rolling_mean(self) -> 'RollingMeanNode':
        """Parse: rolling_mean data column value window=3 as rolling"""
        from noeta_ast import RollingMeanNode
        self.advance()  # consume ROLLING_MEAN
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WINDOW)
        self.expect(TokenType.ASSIGN)
        window = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        min_periods = 1
        if self.match(TokenType.MIN):
            self.advance()
            self.expect(TokenType.ASSIGN)
            min_periods = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return RollingMeanNode(source, column, window, new_alias, min_periods)

    def parse_rolling_sum(self) -> 'RollingSumNode':
        """Parse: rolling_sum data column value window=3 as rolling"""
        from noeta_ast import RollingSumNode
        self.advance()  # consume ROLLING_SUM
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WINDOW)
        self.expect(TokenType.ASSIGN)
        window = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        min_periods = 1
        if self.match(TokenType.MIN):
            self.advance()
            self.expect(TokenType.ASSIGN)
            min_periods = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return RollingSumNode(source, column, window, new_alias, min_periods)

    def parse_rolling_std(self) -> 'RollingStdNode':
        """Parse: rolling_std data column value window=3 as rolling"""
        from noeta_ast import RollingStdNode
        self.advance()  # consume ROLLING_STD
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WINDOW)
        self.expect(TokenType.ASSIGN)
        window = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        min_periods = 1
        if self.match(TokenType.MIN):
            self.advance()
            self.expect(TokenType.ASSIGN)
            min_periods = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return RollingStdNode(source, column, window, new_alias, min_periods)

    def parse_rolling_min(self) -> 'RollingMinNode':
        """Parse: rolling_min data column value window=3 as rolling"""
        from noeta_ast import RollingMinNode
        self.advance()  # consume ROLLING_MIN
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WINDOW)
        self.expect(TokenType.ASSIGN)
        window = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        min_periods = 1
        if self.match(TokenType.MIN):
            self.advance()
            self.expect(TokenType.ASSIGN)
            min_periods = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return RollingMinNode(source, column, window, new_alias, min_periods)

    def parse_rolling_max(self) -> 'RollingMaxNode':
        """Parse: rolling_max data column value window=3 as rolling"""
        from noeta_ast import RollingMaxNode
        self.advance()  # consume ROLLING_MAX
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WINDOW)
        self.expect(TokenType.ASSIGN)
        window = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        min_periods = 1
        if self.match(TokenType.MIN):
            self.advance()
            self.expect(TokenType.ASSIGN)
            min_periods = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return RollingMaxNode(source, column, window, new_alias, min_periods)

    def parse_expanding_mean(self) -> 'ExpandingMeanNode':
        """Parse: expanding_mean data column value as expanding"""
        from noeta_ast import ExpandingMeanNode
        self.advance()  # consume EXPANDING_MEAN
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        min_periods = 1
        if self.match(TokenType.MIN):
            self.advance()
            self.expect(TokenType.ASSIGN)
            min_periods = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ExpandingMeanNode(source, column, new_alias, min_periods)

    def parse_expanding_sum(self) -> 'ExpandingSumNode':
        """Parse: expanding_sum data column value as expanding"""
        from noeta_ast import ExpandingSumNode
        self.advance()  # consume EXPANDING_SUM
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        min_periods = 1
        if self.match(TokenType.MIN):
            self.advance()
            self.expect(TokenType.ASSIGN)
            min_periods = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ExpandingSumNode(source, column, new_alias, min_periods)

    def parse_expanding_min(self) -> 'ExpandingMinNode':
        """Parse: expanding_min data column value as expanding"""
        from noeta_ast import ExpandingMinNode
        self.advance()  # consume EXPANDING_MIN
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        min_periods = 1
        if self.match(TokenType.MIN):
            self.advance()
            self.expect(TokenType.ASSIGN)
            min_periods = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ExpandingMinNode(source, column, new_alias, min_periods)

    def parse_expanding_max(self) -> 'ExpandingMaxNode':
        """Parse: expanding_max data column value as expanding"""
        from noeta_ast import ExpandingMaxNode
        self.advance()  # consume EXPANDING_MAX
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        min_periods = 1
        if self.match(TokenType.MIN):
            self.advance()
            self.expect(TokenType.ASSIGN)
            min_periods = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ExpandingMaxNode(source, column, new_alias, min_periods)

    # ============================================================
    # PHASE 8: DATA RESHAPING OPERATIONS - PARSERS
    # ============================================================

    def parse_pivot(self) -> 'PivotNode':
        """Parse: pivot data index="date" columns="category" values="amount" as pivoted"""
        from noeta_ast import PivotNode
        self.advance()  # consume PIVOT
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.INDEX)
        self.expect(TokenType.ASSIGN)
        index = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.COLUMNS)
        self.expect(TokenType.ASSIGN)
        columns = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.VALUES)
        self.expect(TokenType.ASSIGN)
        values = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return PivotNode(source, index, columns, values, new_alias)

    def parse_pivot_table(self) -> 'PivotTableNode':
        """Parse: pivot_table data index="date" columns="category" values="amount" aggfunc="sum" as pivoted"""
        from noeta_ast import PivotTableNode
        self.advance()  # consume PIVOT_TABLE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.INDEX)
        self.expect(TokenType.ASSIGN)
        index = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.COLUMNS)
        self.expect(TokenType.ASSIGN)
        columns = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.VALUES)
        self.expect(TokenType.ASSIGN)
        values = self.expect(TokenType.STRING_LITERAL).value

        aggfunc = "mean"
        if self.match(TokenType.AGGFUNC):
            self.advance()
            self.expect(TokenType.ASSIGN)
            aggfunc = self.expect(TokenType.STRING_LITERAL).value

        fill_value = None
        if self.match(TokenType.FILL_VALUE):
            self.advance()
            self.expect(TokenType.ASSIGN)
            fill_value = self.parse_value()

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return PivotTableNode(source, index, columns, values, new_alias, aggfunc, fill_value)

    def parse_melt(self) -> 'MeltNode':
        """Parse: melt data id_vars=["id", "name"] value_vars=["jan", "feb"] var_name="month" value_name="sales" as melted"""
        from noeta_ast import MeltNode
        self.advance()  # consume MELT
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ID_VARS)
        self.expect(TokenType.ASSIGN)
        id_vars = self.parse_list_value()

        value_vars = None
        if self.match(TokenType.VALUE_VARS):
            self.advance()
            self.expect(TokenType.ASSIGN)
            value_vars = self.parse_list_value()

        var_name = "variable"
        if self.match(TokenType.VAR_NAME):
            self.advance()
            self.expect(TokenType.ASSIGN)
            var_name = self.expect(TokenType.STRING_LITERAL).value

        value_name = "value"
        if self.match(TokenType.VALUE_NAME):
            self.advance()
            self.expect(TokenType.ASSIGN)
            value_name = self.expect(TokenType.STRING_LITERAL).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return MeltNode(source, id_vars, value_vars, new_alias, var_name, value_name)

    def parse_stack(self) -> 'StackNode':
        """Parse: stack data level=-1 as stacked"""
        from noeta_ast import StackNode
        self.advance()  # consume STACK
        source = self.expect(TokenType.IDENTIFIER).value

        level = -1
        if self.match(TokenType.LEVEL):
            self.advance()
            self.expect(TokenType.ASSIGN)
            level = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return StackNode(source, new_alias, level)

    def parse_unstack(self) -> 'UnstackNode':
        """Parse: unstack data level=-1 fill_value=0 as unstacked"""
        from noeta_ast import UnstackNode
        self.advance()  # consume UNSTACK
        source = self.expect(TokenType.IDENTIFIER).value

        level = -1
        if self.match(TokenType.LEVEL):
            self.advance()
            self.expect(TokenType.ASSIGN)
            level = int(self.expect(TokenType.NUMERIC_LITERAL).value)

        fill_value = None
        if self.match(TokenType.FILL_VALUE):
            self.advance()
            self.expect(TokenType.ASSIGN)
            fill_value = self.parse_value()

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return UnstackNode(source, new_alias, level, fill_value)

    def parse_transpose(self) -> 'TransposeNode':
        """Parse: transpose data as transposed"""
        from noeta_ast import TransposeNode
        self.advance()  # consume TRANSPOSE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return TransposeNode(source, new_alias)

    def parse_crosstab(self) -> 'CrosstabNode':
        """Parse: crosstab data rows="gender" columns="status" values="count" aggfunc="count" as xtab"""
        from noeta_ast import CrosstabNode
        self.advance()  # consume CROSSTAB
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ROWS)
        self.expect(TokenType.ASSIGN)
        row_column = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.COLUMNS)
        self.expect(TokenType.ASSIGN)
        col_column = self.expect(TokenType.STRING_LITERAL).value

        values = None
        if self.match(TokenType.VALUES):
            self.advance()
            self.expect(TokenType.ASSIGN)
            values = self.expect(TokenType.STRING_LITERAL).value

        aggfunc = "count"
        if self.match(TokenType.AGGFUNC):
            self.advance()
            self.expect(TokenType.ASSIGN)
            aggfunc = self.expect(TokenType.STRING_LITERAL).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return CrosstabNode(source, row_column, col_column, new_alias, aggfunc, values)

    # ============================================================
    # PHASE 9: DATA COMBINING OPERATIONS - PARSERS
    # ============================================================

    def parse_merge(self) -> 'MergeNode':
        """Parse: merge left with right on="id" how="inner" as merged"""
        from noeta_ast import MergeNode
        self.advance()  # consume MERGE
        left_alias = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        right_alias = self.expect(TokenType.IDENTIFIER).value

        on = None
        left_on = None
        right_on = None
        how = "inner"
        suffixes = ("_x", "_y")

        if self.match(TokenType.ON):
            self.advance()
            self.expect(TokenType.ASSIGN)
            on = self.expect(TokenType.STRING_LITERAL).value

        if self.match(TokenType.LEFT_ON):
            self.advance()
            self.expect(TokenType.ASSIGN)
            left_on = self.expect(TokenType.STRING_LITERAL).value

        if self.match(TokenType.RIGHT_ON):
            self.advance()
            self.expect(TokenType.ASSIGN)
            right_on = self.expect(TokenType.STRING_LITERAL).value

        if self.match(TokenType.HOW):
            self.advance()
            self.expect(TokenType.ASSIGN)
            how = self.expect(TokenType.STRING_LITERAL).value

        if self.match(TokenType.SUFFIXES):
            self.advance()
            self.expect(TokenType.ASSIGN)
            suffixes_list = self.parse_list_value()
            suffixes = tuple(suffixes_list) if len(suffixes_list) >= 2 else ("_x", "_y")

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return MergeNode(left_alias, right_alias, new_alias, on, left_on, right_on, how, suffixes)

    def parse_concat_vertical(self) -> 'ConcatVerticalNode':
        """Parse: concat_vertical [df1, df2, df3] ignore_index=true as concatenated"""
        from noeta_ast import ConcatVerticalNode
        self.advance()  # consume CONCAT_VERTICAL
        sources = self.parse_list_value()

        ignore_index = True
        if self.match(TokenType.IGNORE_INDEX):
            self.advance()
            self.expect(TokenType.ASSIGN)
            idx_val = self.parse_value()
            ignore_index = idx_val if isinstance(idx_val, bool) else str(idx_val).lower() == 'true'

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ConcatVerticalNode(sources, new_alias, ignore_index)

    def parse_concat_horizontal(self) -> 'ConcatHorizontalNode':
        """Parse: concat_horizontal [df1, df2] as concatenated"""
        from noeta_ast import ConcatHorizontalNode
        self.advance()  # consume CONCAT_HORIZONTAL
        sources = self.parse_list_value()

        ignore_index = False
        if self.match(TokenType.IGNORE_INDEX):
            self.advance()
            self.expect(TokenType.ASSIGN)
            idx_val = self.parse_value()
            ignore_index = idx_val if isinstance(idx_val, bool) else str(idx_val).lower() == 'true'

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ConcatHorizontalNode(sources, new_alias, ignore_index)

    def parse_union(self) -> 'UnionNode':
        """Parse: union df1 with df2 as combined"""
        from noeta_ast import UnionNode
        self.advance()  # consume UNION
        left_alias = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        right_alias = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return UnionNode(left_alias, right_alias, new_alias)

    def parse_intersection(self) -> 'IntersectionNode':
        """Parse: intersection df1 with df2 as common"""
        from noeta_ast import IntersectionNode
        self.advance()  # consume INTERSECTION
        left_alias = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        right_alias = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return IntersectionNode(left_alias, right_alias, new_alias)

    def parse_difference(self) -> 'DifferenceNode':
        """Parse: difference df1 with df2 as diff"""
        from noeta_ast import DifferenceNode
        self.advance()  # consume DIFFERENCE
        left_alias = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        right_alias = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return DifferenceNode(left_alias, right_alias, new_alias)

    # ============================================================
    # PHASE 10: ADVANCED OPERATIONS - PARSERS
    # ============================================================

    def parse_set_index(self) -> 'SetIndexNode':
        """Parse: set_index data column id drop=true as indexed"""
        from noeta_ast import SetIndexNode
        self.advance()  # consume SET_INDEX
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        drop = True
        if self.match(TokenType.DROP):
            self.advance()
            self.expect(TokenType.ASSIGN)
            drop_val = self.parse_value()
            drop = drop_val if isinstance(drop_val, bool) else str(drop_val).lower() == 'true'

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return SetIndexNode(source, column, new_alias, drop)

    def parse_reset_index(self) -> 'ResetIndexNode':
        """Parse: reset_index data drop=false as reset"""
        from noeta_ast import ResetIndexNode
        self.advance()  # consume RESET_INDEX
        source = self.expect(TokenType.IDENTIFIER).value

        drop = False
        if self.match(TokenType.DROP):
            self.advance()
            self.expect(TokenType.ASSIGN)
            drop_val = self.parse_value()
            drop = drop_val if isinstance(drop_val, bool) else str(drop_val).lower() == 'true'

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ResetIndexNode(source, new_alias, drop)

    def parse_apply_row(self) -> 'ApplyRowNode':
        """Parse: apply_row data function="lambda x: x.sum()" as applied"""
        from noeta_ast import ApplyRowNode
        self.advance()  # consume APPLY_ROW
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.FUNCTION)
        self.expect(TokenType.ASSIGN)
        function_expr = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ApplyRowNode(source, function_expr, new_alias)

    def parse_apply_column(self) -> 'ApplyColumnNode':
        """Parse: apply_column data column value function="lambda x: x * 2" as applied"""
        from noeta_ast import ApplyColumnNode
        self.advance()  # consume APPLY_COLUMN
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.FUNCTION)
        self.expect(TokenType.ASSIGN)
        function_expr = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ApplyColumnNode(source, column, function_expr, new_alias)

    def parse_resample(self) -> 'ResampleNode':
        """Parse: resample data rule="D" column value aggfunc="sum" as resampled"""
        from noeta_ast import ResampleNode
        self.advance()  # consume RESAMPLE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.RULE)
        self.expect(TokenType.ASSIGN)
        rule = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AGGFUNC)
        self.expect(TokenType.ASSIGN)
        aggfunc = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ResampleNode(source, rule, column, aggfunc, new_alias)

    def parse_assign(self) -> 'AssignNode':
        """Parse: assign data column status value="active" as assigned"""
        from noeta_ast import AssignNode
        self.advance()  # consume ASSIGN
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.VALUE)
        self.expect(TokenType.ASSIGN)
        value = self.parse_value()
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return AssignNode(source, column, value, new_alias)

    # ========================================================================
    # HIGH-PRIORITY MISSING OPERATIONS (Phase 11) - Parser Methods
    # ========================================================================

    # Cumulative Operations
    def parse_cumsum(self) -> 'CumSumNode':
        """Parse: cumsum data column sales as cumulative_sales"""
        from noeta_ast import CumSumNode
        self.advance()  # consume CUMSUM
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return CumSumNode(source, column, new_alias)

    def parse_cummax(self) -> 'CumMaxNode':
        """Parse: cummax data column value as cumulative_max"""
        from noeta_ast import CumMaxNode
        self.advance()  # consume CUMMAX
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return CumMaxNode(source, column, new_alias)

    def parse_cummin(self) -> 'CumMinNode':
        """Parse: cummin data column value as cumulative_min"""
        from noeta_ast import CumMinNode
        self.advance()  # consume CUMMIN
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return CumMinNode(source, column, new_alias)

    def parse_cumprod(self) -> 'CumProdNode':
        """Parse: cumprod data column value as cumulative_product"""
        from noeta_ast import CumProdNode
        self.advance()  # consume CUMPROD
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return CumProdNode(source, column, new_alias)

    # Time Series Operations
    def parse_pct_change(self) -> 'PctChangeNode':
        """Parse: pct_change data column price with periods=1 as price_change"""
        from noeta_ast import PctChangeNode
        self.advance()  # consume PCT_CHANGE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        # Default period
        periods = 1
        if self.match(TokenType.WITH):
            self.advance()
            self.expect(TokenType.PERIODS)
            self.expect(TokenType.ASSIGN)
            periods = self.expect(TokenType.NUMERIC_LITERAL).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return PctChangeNode(source, column, periods, new_alias)

    def parse_diff(self) -> 'DiffNode':
        """Parse: diff data column value with periods=1 as value_diff"""
        from noeta_ast import DiffNode
        self.advance()  # consume DIFF
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        # Default period
        periods = 1
        if self.match(TokenType.WITH):
            self.advance()
            self.expect(TokenType.PERIODS)
            self.expect(TokenType.ASSIGN)
            periods = self.expect(TokenType.NUMERIC_LITERAL).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return DiffNode(source, column, periods, new_alias)

    def parse_shift(self) -> 'ShiftNode':
        """Parse: shift data column value with periods=1 fill_value=0 as shifted"""
        from noeta_ast import ShiftNode
        self.advance()  # consume SHIFT
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        # Default values
        periods = 1
        fill_value = None

        if self.match(TokenType.WITH):
            self.advance()
            # Parse parameters
            if self.match(TokenType.PERIODS):
                self.advance()
                self.expect(TokenType.ASSIGN)
                periods = self.expect(TokenType.NUMERIC_LITERAL).value

            if self.match(TokenType.FILL_VALUE):
                self.advance()
                self.expect(TokenType.ASSIGN)
                fill_value = self.parse_value()

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ShiftNode(source, column, periods, fill_value, new_alias)

    # Apply/Map Operations
    def parse_applymap(self) -> 'ApplyMapNode':
        """Parse: applymap data function="lambda x: x * 2" as doubled"""
        from noeta_ast import ApplyMapNode
        self.advance()  # consume APPLYMAP
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.FUNCTION)
        self.expect(TokenType.ASSIGN)
        function_expr = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ApplyMapNode(source, function_expr, new_alias)

    def parse_map_values(self) -> 'MapValuesNode':
        """Parse: map_values data column status mapping={"active": 1, "inactive": 0} as status_coded"""
        from noeta_ast import MapValuesNode
        self.advance()  # consume MAP_VALUES
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.MAPPING)
        self.expect(TokenType.ASSIGN)
        mapping = self.parse_dict_value()
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return MapValuesNode(source, column, mapping, new_alias)

    # Additional Date/Time Extraction Operations
    def parse_extract_hour(self) -> 'ExtractHourNode':
        """Parse: extract_hour data column timestamp as hour"""
        from noeta_ast import ExtractHourNode
        self.advance()  # consume EXTRACT_HOUR
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ExtractHourNode(source, column, new_alias)

    def parse_extract_minute(self) -> 'ExtractMinuteNode':
        """Parse: extract_minute data column timestamp as minute"""
        from noeta_ast import ExtractMinuteNode
        self.advance()  # consume EXTRACT_MINUTE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ExtractMinuteNode(source, column, new_alias)

    def parse_extract_second(self) -> 'ExtractSecondNode':
        """Parse: extract_second data column timestamp as second"""
        from noeta_ast import ExtractSecondNode
        self.advance()  # consume EXTRACT_SECOND
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ExtractSecondNode(source, column, new_alias)

    def parse_extract_dayofweek(self) -> 'ExtractDayOfWeekNode':
        """Parse: extract_dayofweek data column timestamp as day_of_week"""
        from noeta_ast import ExtractDayOfWeekNode
        self.advance()  # consume EXTRACT_DAYOFWEEK
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ExtractDayOfWeekNode(source, column, new_alias)

    def parse_extract_dayofyear(self) -> 'ExtractDayOfYearNode':
        """Parse: extract_dayofyear data column timestamp as day_of_year"""
        from noeta_ast import ExtractDayOfYearNode
        self.advance()  # consume EXTRACT_DAYOFYEAR
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ExtractDayOfYearNode(source, column, new_alias)

    def parse_extract_weekofyear(self) -> 'ExtractWeekOfYearNode':
        """Parse: extract_weekofyear data column timestamp as week_of_year"""
        from noeta_ast import ExtractWeekOfYearNode
        self.advance()  # consume EXTRACT_WEEKOFYEAR
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ExtractWeekOfYearNode(source, column, new_alias)

    def parse_extract_quarter(self) -> 'ExtractQuarterNode':
        """Parse: extract_quarter data column timestamp as quarter"""
        from noeta_ast import ExtractQuarterNode
        self.advance()  # consume EXTRACT_QUARTER
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ExtractQuarterNode(source, column, new_alias)

    def parse_extract(self) -> 'ExtractNode':
        """Parse: extract <source> column <col> with part=<part> as <alias>"""
        from noeta_ast import ExtractNode

        self.expect(TokenType.EXTRACT)
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        # Parse WITH clause for part parameter
        self.expect(TokenType.WITH)
        self.expect(TokenType.PART)
        self.expect(TokenType.ASSIGN)
        part = self.parse_value()  # String value like "year", "month", etc.

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value

        return ExtractNode(source, column, part, new_alias)

    # Date Arithmetic Operations
    def parse_date_add(self) -> 'DateAddNode':
        """Parse: date_add data column timestamp value=5 unit="days" as future_date"""
        from noeta_ast import DateAddNode
        self.advance()  # consume DATE_ADD
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.VALUE)
        self.expect(TokenType.ASSIGN)
        value = self.expect(TokenType.NUMERIC_LITERAL).value
        self.expect(TokenType.UNIT)
        self.expect(TokenType.ASSIGN)
        unit = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return DateAddNode(source, column, value, unit, new_alias)

    def parse_date_subtract(self) -> 'DateSubtractNode':
        """Parse: date_subtract data column timestamp value=5 unit="days" as past_date"""
        from noeta_ast import DateSubtractNode
        self.advance()  # consume DATE_SUBTRACT
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.VALUE)
        self.expect(TokenType.ASSIGN)
        value = self.expect(TokenType.NUMERIC_LITERAL).value
        self.expect(TokenType.UNIT)
        self.expect(TokenType.ASSIGN)
        unit = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return DateSubtractNode(source, column, value, unit, new_alias)

    def parse_format_datetime(self) -> 'FormatDateTimeNode':
        """Parse: format_datetime data column timestamp format="%Y-%m-%d" as formatted_date"""
        from noeta_ast import FormatDateTimeNode
        self.advance()  # consume FORMAT_DATETIME
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.FORMAT)
        self.expect(TokenType.ASSIGN)
        format_string = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FormatDateTimeNode(source, column, format_string, new_alias)

    # Advanced String Operations
    def parse_extract_regex(self) -> 'ExtractRegexNode':
        """Parse: extract_regex data column text pattern="[0-9]+" group=0 as numbers"""
        from noeta_ast import ExtractRegexNode
        self.advance()  # consume EXTRACT_REGEX
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.PATTERN)
        self.expect(TokenType.ASSIGN)
        pattern = self.expect(TokenType.STRING_LITERAL).value

        # Optional group parameter
        group = 0
        if self.match(TokenType.IDENTIFIER) and self.current_token().value == "group":
            self.advance()
            self.expect(TokenType.ASSIGN)
            group = self.expect(TokenType.NUMERIC_LITERAL).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ExtractRegexNode(source, column, pattern, group, new_alias)

    def parse_title(self) -> 'TitleNode':
        """Parse: title data column text as title_case"""
        from noeta_ast import TitleNode
        self.advance()  # consume TITLE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return TitleNode(source, column, new_alias)

    def parse_capitalize(self) -> 'CapitalizeNode':
        """Parse: capitalize data column text as capitalized"""
        from noeta_ast import CapitalizeNode
        self.advance()  # consume CAPITALIZE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return CapitalizeNode(source, column, new_alias)

    def parse_lstrip(self) -> 'LStripNode':
        """Parse: lstrip data column text with chars=" " as left_stripped"""
        from noeta_ast import LStripNode
        self.advance()  # consume LSTRIP
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        # Optional chars parameter
        chars = None
        if self.match(TokenType.WITH):
            self.advance()
            if self.match(TokenType.IDENTIFIER) and self.current_token().value == "chars":
                self.advance()
                self.expect(TokenType.ASSIGN)
                chars = self.expect(TokenType.STRING_LITERAL).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return LStripNode(source, column, chars, new_alias)

    def parse_rstrip(self) -> 'RStripNode':
        """Parse: rstrip data column text with chars=" " as right_stripped"""
        from noeta_ast import RStripNode
        self.advance()  # consume RSTRIP
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value

        # Optional chars parameter
        chars = None
        if self.match(TokenType.WITH):
            self.advance()
            if self.match(TokenType.IDENTIFIER) and self.current_token().value == "chars":
                self.advance()
                self.expect(TokenType.ASSIGN)
                chars = self.expect(TokenType.STRING_LITERAL).value

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return RStripNode(source, column, chars, new_alias)

    def parse_find(self) -> 'FindNode':
        """Parse: find data column text substring="hello" as position"""
        from noeta_ast import FindNode
        self.advance()  # consume FIND
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.SUBSTRING)
        self.expect(TokenType.ASSIGN)
        substring = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return FindNode(source, column, substring, new_alias)

    # Binning with Explicit Boundaries
    def parse_cut(self) -> 'CutNode':
        """Parse: cut data column age bins=[0, 18, 35, 50, 100] labels=["child", "young", "middle", "senior"] as age_group"""
        from noeta_ast import CutNode
        self.advance()  # consume CUT
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.BINS)
        self.expect(TokenType.ASSIGN)
        bins = self.parse_list_value()

        # Optional parameters
        labels = None
        include_lowest = False

        if self.match(TokenType.LABELS):
            self.advance()
            self.expect(TokenType.ASSIGN)
            labels = self.parse_list_value()

        if self.match(TokenType.IDENTIFIER) and self.current_token().value == "include_lowest":
            self.advance()
            self.expect(TokenType.ASSIGN)
            include_lowest = self.parse_value()

        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return CutNode(source, column, bins, labels, include_lowest, new_alias)

    # ===== PHASE 12: MEDIUM PRIORITY OPERATIONS =====

    # Scaling & Normalization Operations
    def parse_robust_scale(self) -> 'RobustScaleNode':
        """Parse: robust_scale data column price as price_robust"""
        from noeta_ast import RobustScaleNode
        self.advance()  # consume ROBUST_SCALE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return RobustScaleNode(source, column, new_alias)

    def parse_maxabs_scale(self) -> 'MaxAbsScaleNode':
        """Parse: maxabs_scale data column value as value_scaled"""
        from noeta_ast import MaxAbsScaleNode
        self.advance()  # consume MAXABS_SCALE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return MaxAbsScaleNode(source, column, new_alias)

    # Advanced Encoding Operations
    def parse_ordinal_encode(self) -> 'OrdinalEncodeNode':
        """Parse: ordinal_encode data column size order=["S", "M", "L", "XL"] as size_encoded"""
        from noeta_ast import OrdinalEncodeNode
        self.advance()  # consume ORDINAL_ENCODE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ORDER)
        self.expect(TokenType.ASSIGN)
        order = self.parse_list_value()
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return OrdinalEncodeNode(source, column, order, new_alias)

    def parse_target_encode(self) -> 'TargetEncodeNode':
        """Parse: target_encode data column category target="sales" as category_encoded"""
        from noeta_ast import TargetEncodeNode
        self.advance()  # consume TARGET_ENCODE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.TARGET)
        self.expect(TokenType.ASSIGN)
        target = self.expect(TokenType.STRING_LITERAL).value
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return TargetEncodeNode(source, column, target, new_alias)

    # Data Validation Operations
    def parse_assert_unique(self) -> 'AssertUniqueNode':
        """Parse: assert_unique data column id"""
        from noeta_ast import AssertUniqueNode
        self.advance()  # consume ASSERT_UNIQUE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        return AssertUniqueNode(source, column)

    def parse_assert_no_nulls(self) -> 'AssertNoNullsNode':
        """Parse: assert_no_nulls data column required_field"""
        from noeta_ast import AssertNoNullsNode
        self.advance()  # consume ASSERT_NO_NULLS
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        return AssertNoNullsNode(source, column)

    def parse_assert_range(self) -> 'AssertRangeNode':
        """Parse: assert_range data column age min=0 max=120"""
        from noeta_ast import AssertRangeNode
        self.advance()  # consume ASSERT_RANGE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        
        min_value = None
        max_value = None
        
        if self.match(TokenType.MIN):
            self.advance()
            self.expect(TokenType.ASSIGN)
            min_value = self.parse_value()
        
        if self.match(TokenType.MAX):
            self.advance()
            self.expect(TokenType.ASSIGN)
            max_value = self.parse_value()
        
        return AssertRangeNode(source, column, min_value, max_value)

    # Advanced Index Operations
    def parse_reindex(self) -> 'ReindexNode':
        """Parse: reindex data with index=[0, 1, 2, 3] as reindexed"""
        from noeta_ast import ReindexNode
        self.advance()  # consume REINDEX
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        self.expect(TokenType.INDEX)
        self.expect(TokenType.ASSIGN)
        index = self.parse_list_value()
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return ReindexNode(source, index, new_alias)

    def parse_set_multiindex(self) -> 'SetMultiIndexNode':
        """Parse: set_multiindex data columns ["category", "subcategory"] as hierarchical"""
        from noeta_ast import SetMultiIndexNode
        self.advance()  # consume SET_MULTIINDEX
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMNS)
        columns = self.parse_list_value()
        self.expect(TokenType.AS)
        new_alias = self.expect(TokenType.IDENTIFIER).value
        return SetMultiIndexNode(source, columns, new_alias)

    # Boolean Operations
    def parse_any(self) -> 'AnyNode':
        """Parse: any data column flag"""
        from noeta_ast import AnyNode
        self.advance()  # consume ANY
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        return AnyNode(source, column)

    def parse_all(self) -> 'AllNode':
        """Parse: all data column flag"""
        from noeta_ast import AllNode
        self.advance()  # consume ALL
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        return AllNode(source, column)

    def parse_count_true(self) -> 'CountTrueNode':
        """Parse: count_true data column flag"""
        from noeta_ast import CountTrueNode
        self.advance()  # consume COUNT_TRUE
        source = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLUMN)
        column = self.expect(TokenType.IDENTIFIER).value
        return CountTrueNode(source, column)

    def parse_compare(self) -> 'CompareNode':
        """Parse: compare df1 with df2"""
        from noeta_ast import CompareNode
        self.advance()  # consume COMPARE
        left_alias = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.WITH)
        right_alias = self.expect(TokenType.IDENTIFIER).value
        return CompareNode(left_alias, right_alias)
