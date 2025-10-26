"""
Noeta Lexer - Tokenizes Noeta DSL source code
"""
import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional

class TokenType(Enum):
    # Keywords
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
    
    # Other keywords
    AS = auto()
    BY = auto()
    WITH = auto()
    ON = auto()
    AGG = auto()
    COLUMN = auto()
    COLUMNS = auto()
    VALUE = auto()
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
    VALUES = auto()
    LABELS = auto()
    TO = auto()
    FORMAT = auto()
    FILENAME = auto()
    WIDTH = auto()
    HEIGHT = auto()
    DESC = auto()
    
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
        
        # Keywords mapping
        self.keywords = {
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
            'as': TokenType.AS,
            'by': TokenType.BY,
            'with': TokenType.WITH,
            'on': TokenType.ON,
            'agg': TokenType.AGG,
            'column': TokenType.COLUMN,
            'columns': TokenType.COLUMNS,
            'value': TokenType.VALUE,
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
            'values': TokenType.VALUES,
            'labels': TokenType.LABELS,
            'to': TokenType.TO,
            'format': TokenType.FORMAT,
            'filename': TokenType.FILENAME,
            'width': TokenType.WIDTH,
            'height': TokenType.HEIGHT,
            'desc': TokenType.DESC,
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
