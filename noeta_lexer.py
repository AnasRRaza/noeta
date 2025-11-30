"""
Noeta Lexer - Tokenizes Noeta DSL source code
"""
import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional

class TokenType(Enum):
    # Keywords - Operations
    LOAD = auto()
    SELECT = auto()
    FILTER = auto()
    SORT = auto()
    JOIN = auto()
    GROUPBY = auto()
    SAMPLE = auto()
    DROPNA = auto()
    FILLNA = auto()
    MUTATE = auto()
    APPLY = auto()
    DESCRIBE = auto()
    SUMMARY = auto()
    OUTLIERS = auto()
    QUANTILE = auto()
    NORMALIZE = auto()
    BINNING = auto()
    ROLLING = auto()
    HYPOTHESIS = auto()
    BOXPLOT = auto()
    HEATMAP = auto()
    PAIRPLOT = auto()
    TIMESERIES = auto()
    PIE = auto()
    SAVE = auto()
    EXPORT_PLOT = auto()
    INFO = auto()

    # Phase 2: Selection & Projection operations
    SELECT_BY_TYPE = auto()
    HEAD = auto()
    TAIL = auto()
    ILOC = auto()
    LOC = auto()
    RENAME = auto()
    REORDER = auto()

    # Phase 3: Filtering operations
    FILTER_BETWEEN = auto()
    FILTER_ISIN = auto()
    FILTER_CONTAINS = auto()
    FILTER_STARTSWITH = auto()
    FILTER_ENDSWITH = auto()
    FILTER_REGEX = auto()
    FILTER_NULL = auto()
    FILTER_NOTNULL = auto()
    FILTER_DUPLICATES = auto()

    # Phase 4: Transformation operations - Math
    ROUND = auto()
    ABS = auto()
    SQRT = auto()
    POWER = auto()
    LOG = auto()
    CEIL = auto()
    FLOOR = auto()

    # Phase 4: Transformation operations - String
    UPPER = auto()
    LOWER = auto()
    STRIP = auto()
    REPLACE = auto()
    SPLIT = auto()
    CONCAT = auto()
    SUBSTRING = auto()
    LENGTH = auto()

    # Phase 4: Transformation operations - Date
    PARSE_DATETIME = auto()
    EXTRACT_YEAR = auto()
    EXTRACT_MONTH = auto()
    EXTRACT_DAY = auto()
    DATE_DIFF = auto()

    # Phase 4: Transformation operations - Type/Encoding/Scaling
    ASTYPE = auto()
    TO_NUMERIC = auto()
    ONE_HOT_ENCODE = auto()
    LABEL_ENCODE = auto()
    STANDARD_SCALE = auto()
    MINMAX_SCALE = auto()

    # Phase 5: Cleaning operations
    ISNULL = auto()
    NOTNULL = auto()
    COUNT_NA = auto()
    FILL_FORWARD = auto()
    FILL_BACKWARD = auto()
    FILL_MEAN = auto()
    FILL_MEDIAN = auto()
    FILL_MODE = auto()
    INTERPOLATE = auto()
    DUPLICATED = auto()
    COUNT_DUPLICATES = auto()
    DROP_DUPLICATES = auto()

    # Phase 6: Data Ordering operations
    SORT_INDEX = auto()
    RANK = auto()

    # Phase 7: Aggregation & Grouping operations
    FILTER_GROUPS = auto()
    GROUP_TRANSFORM = auto()
    WINDOW_RANK = auto()
    WINDOW_LAG = auto()
    WINDOW_LEAD = auto()
    ROLLING_MEAN = auto()
    ROLLING_SUM = auto()
    ROLLING_STD = auto()
    ROLLING_MIN = auto()
    ROLLING_MAX = auto()
    EXPANDING_MEAN = auto()
    EXPANDING_SUM = auto()
    EXPANDING_MIN = auto()
    EXPANDING_MAX = auto()

    # Phase 8: Data Reshaping operations
    PIVOT = auto()
    PIVOT_TABLE = auto()
    MELT = auto()
    STACK = auto()
    UNSTACK = auto()
    TRANSPOSE = auto()
    CROSSTAB = auto()

    # Phase 9: Data Combining operations
    MERGE = auto()
    CONCAT_VERTICAL = auto()
    CONCAT_HORIZONTAL = auto()
    UNION = auto()
    INTERSECTION = auto()
    DIFFERENCE = auto()

    # Phase 10: Advanced Operations
    SET_INDEX = auto()
    RESET_INDEX = auto()
    APPLY_ROW = auto()
    APPLY_COLUMN = auto()
    RESAMPLE = auto()
    ASSIGN_CONST = auto()  # assign constant value operation

    # File format keywords
    CSV = auto()
    JSON = auto()
    EXCEL = auto()
    PARQUET = auto()
    SQL = auto()

    # Common keywords
    AS = auto()
    BY = auto()
    WITH = auto()
    ON = auto()
    FROM = auto()
    AGG = auto()
    COLUMN = auto()
    COLUMNS = auto()
    VALUE = auto()
    VALUES = auto()
    N = auto()
    RANDOM = auto()
    METHOD = auto()
    Q = auto()
    BINS = auto()
    WINDOW = auto()
    FUNCTION = auto()
    VS = auto()
    TEST = auto()
    X = auto()
    Y = auto()
    LABELS = auto()
    TO = auto()
    FORMAT = auto()
    FILENAME = auto()
    WIDTH = auto()
    HEIGHT = auto()
    DESC = auto()
    WHERE = auto()
    ASC = auto()
    TYPE = auto()
    ROWS = auto()
    MAPPING = auto()
    ORDER = auto()
    LIMIT = auto()
    OFFSET = auto()
    FIRST = auto()
    LAST = auto()
    MIN = auto()
    MAX = auto()
    PATTERN = auto()
    KEEP = auto()
    SUBSET = auto()
    AND = auto()
    OR = auto()
    IN = auto()
    DECIMALS = auto()
    EXPONENT = auto()
    BASE = auto()
    SEPARATOR = auto()
    UNIT = auto()
    OLD = auto()
    NEW = auto()
    DELIMITER_STR = auto()
    START = auto()
    END = auto()
    DTYPE_STR = auto()
    ERRORS = auto()
    STRATEGY = auto()
    AXIS = auto()
    
    # Additional parameter keywords
    ASCENDING = auto()
    PERIODS = auto()
    ID_VARS = auto()
    VALUE_VARS = auto()
    VAR_NAME = auto()
    VALUE_NAME = auto()
    LEFT = auto()
    RIGHT = auto()
    LEFT_ON = auto()
    RIGHT_ON = auto()
    SUFFIXES = auto()
    HOW = auto()
    AGGFUNC = auto()
    FILL_VALUE = auto()
    LEVEL = auto()
    RULE = auto()
    CONDITION = auto()
    PCT = auto()
    DROP = auto()
    IGNORE_INDEX = auto()

    # LOAD/SAVE parameters
    DELIMITER = auto()
    ENCODING = auto()
    HEADER = auto()
    NAMES = auto()
    USECOLS = auto()
    DTYPE = auto()
    SKIPROWS = auto()
    NROWS = auto()
    NA_VALUES = auto()
    THOUSANDS = auto()
    DECIMAL = auto()
    COMMENT_CHAR = auto()
    SKIP_BLANK_LINES = auto()
    PARSE_DATES = auto()
    DATE_FORMAT = auto()
    CHUNKSIZE = auto()
    COMPRESSION = auto()
    LOW_MEMORY = auto()
    MEMORY_MAP = auto()
    ORIENT = auto()
    TYP = auto()
    CONVERT_AXES = auto()
    CONVERT_DATES = auto()
    PRECISE_FLOAT = auto()
    DATE_UNIT = auto()
    LINES = auto()
    SHEET = auto()
    SHEET_NAME = auto()
    INDEX_COL = auto()
    ENGINE = auto()
    CONVERTERS = auto()
    SKIPFOOTER = auto()
    FILTERS = auto()
    USE_NULLABLE_DTYPES = auto()
    STORAGE_OPTIONS = auto()
    PARAMS = auto()
    COERCE_FLOAT = auto()
    INDEX = auto()
    INDEX_LABEL = auto()
    NA_REP = auto()
    MODE = auto()
    QUOTING = auto()
    QUOTECHAR = auto()
    ESCAPECHAR = auto()
    LINETERMINATOR = auto()
    FLOAT_FORMAT = auto()

    # Literals
    STRING_LITERAL = auto()
    NUMERIC_LITERAL = auto()
    IDENTIFIER = auto()
    
    # Operators
    EQ = auto()  # ==
    NEQ = auto()  # !=
    LT = auto()  # <
    GT = auto()  # >
    LTE = auto()  # <=
    GTE = auto()  # >=
    ASSIGN = auto()  # =
    PLUS = auto()  # +
    MINUS = auto()  # -
    STAR = auto()  # *
    SLASH = auto()  # /
    PERCENT = auto()  # %
    
    # Punctuation
    LBRACE = auto()  # {
    RBRACE = auto()  # }
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    COLON = auto()  # :
    COMMA = auto()  # ,
    
    # Special
    NEWLINE = auto()
    EOF = auto()
    COMMENT = auto()

@dataclass
class Token:
    type: TokenType
    value: any
    line: int
    column: int

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        
        # Keywords mapping (case-insensitive)
        self.keywords = {
            # Operations
            'load': TokenType.LOAD,
            'select': TokenType.SELECT,
            'filter': TokenType.FILTER,
            'sort': TokenType.SORT,
            'join': TokenType.JOIN,
            'groupby': TokenType.GROUPBY,
            'sample': TokenType.SAMPLE,
            'dropna': TokenType.DROPNA,
            'fillna': TokenType.FILLNA,
            'mutate': TokenType.MUTATE,
            'apply': TokenType.APPLY,
            'describe': TokenType.DESCRIBE,
            'summary': TokenType.SUMMARY,
            'outliers': TokenType.OUTLIERS,
            'quantile': TokenType.QUANTILE,
            'normalize': TokenType.NORMALIZE,
            'binning': TokenType.BINNING,
            'rolling': TokenType.ROLLING,
            'hypothesis': TokenType.HYPOTHESIS,
            'boxplot': TokenType.BOXPLOT,
            'heatmap': TokenType.HEATMAP,
            'pairplot': TokenType.PAIRPLOT,
            'timeseries': TokenType.TIMESERIES,
            'pie': TokenType.PIE,
            'save': TokenType.SAVE,
            'export_plot': TokenType.EXPORT_PLOT,
            'info': TokenType.INFO,

            # Phase 2: Selection & Projection
            'select_by_type': TokenType.SELECT_BY_TYPE,
            'head': TokenType.HEAD,
            'tail': TokenType.TAIL,
            'iloc': TokenType.ILOC,
            'loc': TokenType.LOC,
            'rename': TokenType.RENAME,
            'reorder': TokenType.REORDER,

            # Phase 3: Filtering operations
            'filter_between': TokenType.FILTER_BETWEEN,
            'filter_isin': TokenType.FILTER_ISIN,
            'filter_contains': TokenType.FILTER_CONTAINS,
            'filter_startswith': TokenType.FILTER_STARTSWITH,
            'filter_endswith': TokenType.FILTER_ENDSWITH,
            'filter_regex': TokenType.FILTER_REGEX,
            'filter_null': TokenType.FILTER_NULL,
            'filter_notnull': TokenType.FILTER_NOTNULL,
            'filter_duplicates': TokenType.FILTER_DUPLICATES,

            # Phase 4: Transformation operations
            'round': TokenType.ROUND,
            'abs': TokenType.ABS,
            'sqrt': TokenType.SQRT,
            'power': TokenType.POWER,
            'log': TokenType.LOG,
            'ceil': TokenType.CEIL,
            'floor': TokenType.FLOOR,
            'upper': TokenType.UPPER,
            'lower': TokenType.LOWER,
            'strip': TokenType.STRIP,
            'replace': TokenType.REPLACE,
            'split': TokenType.SPLIT,
            'concat': TokenType.CONCAT,
            'substring': TokenType.SUBSTRING,
            'length': TokenType.LENGTH,
            'parse_datetime': TokenType.PARSE_DATETIME,
            'extract_year': TokenType.EXTRACT_YEAR,
            'extract_month': TokenType.EXTRACT_MONTH,
            'extract_day': TokenType.EXTRACT_DAY,
            'date_diff': TokenType.DATE_DIFF,
            'astype': TokenType.ASTYPE,
            'to_numeric': TokenType.TO_NUMERIC,
            'one_hot_encode': TokenType.ONE_HOT_ENCODE,
            'label_encode': TokenType.LABEL_ENCODE,
            'standard_scale': TokenType.STANDARD_SCALE,
            'minmax_scale': TokenType.MINMAX_SCALE,

            # Phase 5: Cleaning operations
            'isnull': TokenType.ISNULL,
            'notnull': TokenType.NOTNULL,
            'count_na': TokenType.COUNT_NA,
            'fill_forward': TokenType.FILL_FORWARD,
            'fill_backward': TokenType.FILL_BACKWARD,
            'fill_mean': TokenType.FILL_MEAN,
            'fill_median': TokenType.FILL_MEDIAN,
            'fill_mode': TokenType.FILL_MODE,
            'interpolate': TokenType.INTERPOLATE,
            'duplicated': TokenType.DUPLICATED,
            'count_duplicates': TokenType.COUNT_DUPLICATES,
            'drop_duplicates': TokenType.DROP_DUPLICATES,

            # Phase 6: Data Ordering operations
            'sort_index': TokenType.SORT_INDEX,
            'rank': TokenType.RANK,

            # Phase 7: Aggregation & Grouping operations
            'filter_groups': TokenType.FILTER_GROUPS,
            'group_transform': TokenType.GROUP_TRANSFORM,
            'window_rank': TokenType.WINDOW_RANK,
            'window_lag': TokenType.WINDOW_LAG,
            'window_lead': TokenType.WINDOW_LEAD,
            'rolling_mean': TokenType.ROLLING_MEAN,
            'rolling_sum': TokenType.ROLLING_SUM,
            'rolling_std': TokenType.ROLLING_STD,
            'rolling_min': TokenType.ROLLING_MIN,
            'rolling_max': TokenType.ROLLING_MAX,
            'expanding_mean': TokenType.EXPANDING_MEAN,
            'expanding_sum': TokenType.EXPANDING_SUM,
            'expanding_min': TokenType.EXPANDING_MIN,
            'expanding_max': TokenType.EXPANDING_MAX,

            # Phase 8: Data Reshaping operations
            'pivot': TokenType.PIVOT,
            'pivot_table': TokenType.PIVOT_TABLE,
            'melt': TokenType.MELT,
            'stack': TokenType.STACK,
            'unstack': TokenType.UNSTACK,
            'transpose': TokenType.TRANSPOSE,
            'crosstab': TokenType.CROSSTAB,

            # Phase 9: Data Combining operations
            'merge': TokenType.MERGE,
            'concat_vertical': TokenType.CONCAT_VERTICAL,
            'concat_horizontal': TokenType.CONCAT_HORIZONTAL,
            'union': TokenType.UNION,
            'intersection': TokenType.INTERSECTION,
            'difference': TokenType.DIFFERENCE,

            # Phase 10: Advanced Operations
            'set_index': TokenType.SET_INDEX,
            'reset_index': TokenType.RESET_INDEX,
            'apply_row': TokenType.APPLY_ROW,
            'apply_column': TokenType.APPLY_COLUMN,
            'resample': TokenType.RESAMPLE,
            'assign': TokenType.ASSIGN_CONST,

            # File formats
            'csv': TokenType.CSV,
            'json': TokenType.JSON,
            'excel': TokenType.EXCEL,
            'parquet': TokenType.PARQUET,
            'sql': TokenType.SQL,

            # Common keywords
            'as': TokenType.AS,
            'by': TokenType.BY,
            'with': TokenType.WITH,
            'on': TokenType.ON,
            'from': TokenType.FROM,
            'agg': TokenType.AGG,
            'column': TokenType.COLUMN,
            'columns': TokenType.COLUMNS,
            'value': TokenType.VALUE,
            'values': TokenType.VALUES,
            'n': TokenType.N,
            'random': TokenType.RANDOM,
            'method': TokenType.METHOD,
            'q': TokenType.Q,
            'bins': TokenType.BINS,
            'window': TokenType.WINDOW,
            'function': TokenType.FUNCTION,
            'vs': TokenType.VS,
            'test': TokenType.TEST,
            'x': TokenType.X,
            'y': TokenType.Y,
            'labels': TokenType.LABELS,
            'to': TokenType.TO,
            'format': TokenType.FORMAT,
            'filename': TokenType.FILENAME,
            'width': TokenType.WIDTH,
            'height': TokenType.HEIGHT,
            'desc': TokenType.DESC,
            'where': TokenType.WHERE,
            'asc': TokenType.ASC,
            'type': TokenType.TYPE,
            'rows': TokenType.ROWS,
            'mapping': TokenType.MAPPING,
            'order': TokenType.ORDER,
            'limit': TokenType.LIMIT,
            'offset': TokenType.OFFSET,
            'first': TokenType.FIRST,
            'last': TokenType.LAST,
            'min': TokenType.MIN,
            'max': TokenType.MAX,
            'pattern': TokenType.PATTERN,
            'keep': TokenType.KEEP,
            'subset': TokenType.SUBSET,
            'and': TokenType.AND,
            'or': TokenType.OR,
            'in': TokenType.IN,
            'decimals': TokenType.DECIMALS,
            'exponent': TokenType.EXPONENT,
            'base': TokenType.BASE,
            'separator': TokenType.SEPARATOR,
            'unit': TokenType.UNIT,
            'old': TokenType.OLD,
            'new': TokenType.NEW,
            'delimiter_str': TokenType.DELIMITER_STR,
            'start': TokenType.START,
            'end': TokenType.END,
            'dtype_str': TokenType.DTYPE_STR,
            'errors': TokenType.ERRORS,
            'strategy': TokenType.STRATEGY,
            'axis': TokenType.AXIS,

            # Additional parameter keywords
            'ascending': TokenType.ASCENDING,
            'periods': TokenType.PERIODS,
            'id_vars': TokenType.ID_VARS,
            'value_vars': TokenType.VALUE_VARS,
            'var_name': TokenType.VAR_NAME,
            'value_name': TokenType.VALUE_NAME,
            'left': TokenType.LEFT,
            'right': TokenType.RIGHT,
            'left_on': TokenType.LEFT_ON,
            'right_on': TokenType.RIGHT_ON,
            'suffixes': TokenType.SUFFIXES,
            'how': TokenType.HOW,
            'aggfunc': TokenType.AGGFUNC,
            'fill_value': TokenType.FILL_VALUE,
            'level': TokenType.LEVEL,
            'rule': TokenType.RULE,
            'condition': TokenType.CONDITION,
            'pct': TokenType.PCT,
            'drop': TokenType.DROP,
            'ignore_index': TokenType.IGNORE_INDEX,

            # LOAD/SAVE parameters
            'delimiter': TokenType.DELIMITER,
            'encoding': TokenType.ENCODING,
            'header': TokenType.HEADER,
            'names': TokenType.NAMES,
            'usecols': TokenType.USECOLS,
            'dtype': TokenType.DTYPE,
            'skiprows': TokenType.SKIPROWS,
            'nrows': TokenType.NROWS,
            'na_values': TokenType.NA_VALUES,
            'thousands': TokenType.THOUSANDS,
            'decimal': TokenType.DECIMAL,
            'comment': TokenType.COMMENT_CHAR,
            'skip_blank_lines': TokenType.SKIP_BLANK_LINES,
            'parse_dates': TokenType.PARSE_DATES,
            'date_format': TokenType.DATE_FORMAT,
            'chunksize': TokenType.CHUNKSIZE,
            'compression': TokenType.COMPRESSION,
            'low_memory': TokenType.LOW_MEMORY,
            'memory_map': TokenType.MEMORY_MAP,
            'orient': TokenType.ORIENT,
            'typ': TokenType.TYP,
            'convert_axes': TokenType.CONVERT_AXES,
            'convert_dates': TokenType.CONVERT_DATES,
            'precise_float': TokenType.PRECISE_FLOAT,
            'date_unit': TokenType.DATE_UNIT,
            'lines': TokenType.LINES,
            'sheet': TokenType.SHEET,
            'sheet_name': TokenType.SHEET_NAME,
            'index_col': TokenType.INDEX_COL,
            'engine': TokenType.ENGINE,
            'converters': TokenType.CONVERTERS,
            'skipfooter': TokenType.SKIPFOOTER,
            'filters': TokenType.FILTERS,
            'use_nullable_dtypes': TokenType.USE_NULLABLE_DTYPES,
            'storage_options': TokenType.STORAGE_OPTIONS,
            'params': TokenType.PARAMS,
            'coerce_float': TokenType.COERCE_FLOAT,
            'index': TokenType.INDEX,
            'index_label': TokenType.INDEX_LABEL,
            'na_rep': TokenType.NA_REP,
            'mode': TokenType.MODE,
            'quoting': TokenType.QUOTING,
            'quotechar': TokenType.QUOTECHAR,
            'escapechar': TokenType.ESCAPECHAR,
            'lineterminator': TokenType.LINETERMINATOR,
            'float_format': TokenType.FLOAT_FORMAT,
        }
    
    def current_char(self) -> Optional[str]:
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]
    
    def peek_char(self, offset=1) -> Optional[str]:
        pos = self.pos + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]
    
    def advance(self):
        if self.pos < len(self.source) and self.source[self.pos] == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        self.pos += 1
    
    def skip_whitespace(self):
        while self.current_char() and self.current_char() in ' \t\r':
            self.advance()
    
    def skip_comment(self):
        if self.current_char() == '#':
            while self.current_char() and self.current_char() != '\n':
                self.advance()
    
    def read_string(self) -> str:
        # Skip opening quote
        self.advance()
        value = ''
        while self.current_char() and self.current_char() != '"':
            if self.current_char() == '\\' and self.peek_char() == '"':
                self.advance()  # Skip backslash
                value += '"'
                self.advance()
            else:
                value += self.current_char()
                self.advance()
        # Skip closing quote
        if self.current_char() == '"':
            self.advance()
        return value
    
    def read_number(self) -> float:
        value = ''
        while self.current_char() and (self.current_char().isdigit() or self.current_char() == '.'):
            value += self.current_char()
            self.advance()
        return float(value) if '.' in value else int(value)
    
    def read_identifier(self) -> str:
        value = ''
        while self.current_char() and (self.current_char().isalnum() or self.current_char() == '_'):
            value += self.current_char()
            self.advance()
        return value
    
    def next_token(self) -> Optional[Token]:
        while self.current_char():
            # Skip whitespace
            if self.current_char() in ' \t\r':
                self.skip_whitespace()
                continue
            
            # Handle newlines
            if self.current_char() == '\n':
                token = Token(TokenType.NEWLINE, '\n', self.line, self.column)
                self.advance()
                return token
            
            # Skip comments
            if self.current_char() == '#':
                self.skip_comment()
                continue
            
            # String literals
            if self.current_char() == '"':
                line, col = self.line, self.column
                value = self.read_string()
                return Token(TokenType.STRING_LITERAL, value, line, col)
            
            # Numeric literals
            if self.current_char().isdigit():
                line, col = self.line, self.column
                value = self.read_number()
                return Token(TokenType.NUMERIC_LITERAL, value, line, col)
            
            # Identifiers and keywords
            if self.current_char().isalpha() or self.current_char() == '_':
                line, col = self.line, self.column
                value = self.read_identifier()
                token_type = self.keywords.get(value, TokenType.IDENTIFIER)
                return Token(token_type, value, line, col)
            
            # Operators
            line, col = self.line, self.column
            
            # Two-character operators
            if self.current_char() == '=' and self.peek_char() == '=':
                self.advance()
                self.advance()
                return Token(TokenType.EQ, '==', line, col)
            
            if self.current_char() == '!' and self.peek_char() == '=':
                self.advance()
                self.advance()
                return Token(TokenType.NEQ, '!=', line, col)
            
            if self.current_char() == '<' and self.peek_char() == '=':
                self.advance()
                self.advance()
                return Token(TokenType.LTE, '<=', line, col)
            
            if self.current_char() == '>' and self.peek_char() == '=':
                self.advance()
                self.advance()
                return Token(TokenType.GTE, '>=', line, col)
            
            # Single-character operators
            if self.current_char() == '<':
                self.advance()
                return Token(TokenType.LT, '<', line, col)

            if self.current_char() == '>':
                self.advance()
                return Token(TokenType.GT, '>', line, col)

            # Single = (assignment in mutate expressions)
            if self.current_char() == '=':
                self.advance()
                return Token(TokenType.ASSIGN, '=', line, col)

            # Arithmetic operators
            if self.current_char() == '+':
                self.advance()
                return Token(TokenType.PLUS, '+', line, col)

            if self.current_char() == '-':
                self.advance()
                return Token(TokenType.MINUS, '-', line, col)

            if self.current_char() == '*':
                self.advance()
                return Token(TokenType.STAR, '*', line, col)

            if self.current_char() == '/':
                self.advance()
                return Token(TokenType.SLASH, '/', line, col)

            if self.current_char() == '%':
                self.advance()
                return Token(TokenType.PERCENT, '%', line, col)

            # Punctuation
            if self.current_char() == '{':
                self.advance()
                return Token(TokenType.LBRACE, '{', line, col)
            
            if self.current_char() == '}':
                self.advance()
                return Token(TokenType.RBRACE, '}', line, col)
            
            if self.current_char() == '[':
                self.advance()
                return Token(TokenType.LBRACKET, '[', line, col)
            
            if self.current_char() == ']':
                self.advance()
                return Token(TokenType.RBRACKET, ']', line, col)
            
            if self.current_char() == ':':
                self.advance()
                return Token(TokenType.COLON, ':', line, col)
            
            if self.current_char() == ',':
                self.advance()
                return Token(TokenType.COMMA, ',', line, col)
            
            # Unknown character
            raise SyntaxError(f"Unexpected character '{self.current_char()}' at line {self.line}, column {self.column}")
        
        return Token(TokenType.EOF, None, self.line, self.column)
    
    def tokenize(self) -> List[Token]:
        tokens = []
        while True:
            token = self.next_token()
            if token.type != TokenType.NEWLINE:  # Filter out newlines for simpler parsing
                tokens.append(token)
            if token.type == TokenType.EOF:
                break
        return tokens
