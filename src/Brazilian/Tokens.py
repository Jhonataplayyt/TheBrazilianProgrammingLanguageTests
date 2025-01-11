from enum import Enum, auto
from typing import *

class TokenType(Enum):
  INT				 = auto()
  FLOAT    	 = auto()
  STRING		 = auto()
  BIN        = auto()
  BYTES  		 = auto()
  IDENTIFIER = auto()
  KEYWORD		 = auto()
  PLUS     	 = auto()
  MINUS    	 = auto()
  MUL      	 = auto()
  DIV      	 = auto()
  CBAR     	 = auto()
  PERC       = auto()
  POW				 = auto()
  EQ				 = auto()
  LPAREN   	 = auto()
  RPAREN   	 = auto()
  LSQUARE    = auto()
  RSQUARE    = auto()
  EE				 = auto()
  NE				 = auto()
  LT				 = auto()
  GT				 = auto()
  LTE				 = auto()
  GTE				 = auto()
  COMMA			 = auto()
  ARROW			 = auto()
  LCURLY     = auto()
  RCURLY     = auto()
  COLON      = auto()
  DOT        = auto()
  NEWLINE		 = auto()
  EOF				 = auto()

KEYWORDS = [
  'and',
  'or',
  'not',
  'if',
  'elif',
  'else',
  'for',
  'to',
  'step',
  'while',
  'function',
  'then',
  'end',
  'return',
  'continue',
  'break',
  "pass",
  'import',
  'do',
  'try',
  'catch',
  'as',
  'from',
  'in',
  'switch',
  'case',
  'const',
  'global',
  'class',
  'namespace',
  'struct'
]

class Token:
  def __init__(self, type_, value=None, pos_start=None, pos_end=None):
    self.type = type_
    self.value = value

    if pos_start:
      self.pos_start = pos_start.copy()
      self.pos_end = pos_start.copy()
      self.pos_end.advance()

    if pos_end:
      self.pos_end = pos_end.copy()

  def copy(self):
    return Token(self.type, self.value, self.pos_start.copy(), self.pos_end.copy())

  def matches(self, type_, value):
    return self.type == type_ and self.value == value
  
  def __repr__(self):
    if self.value: return f'{self.type.name}:{self.value}'
    return f'{self.type.name}'

SINGLE_CHAR_TOKS: Dict[str, TokenType] = {
  ";": TokenType.NEWLINE,
  "\n": TokenType.NEWLINE,
  "\\": TokenType.CBAR,
  "+": TokenType.PLUS,
  "*": TokenType.MUL,
  "/": TokenType.DIV,
  "%": TokenType.PERC,
  "^": TokenType.POW,
  "(": TokenType.LPAREN,
  ")": TokenType.RPAREN,
  "[": TokenType.LSQUARE,
  "]": TokenType.RSQUARE,
  "{": TokenType.LCURLY,
  "}": TokenType.RCURLY,
  ",": TokenType.COMMA,
  ":": TokenType.COLON,
  ".": TokenType.DOT,
}