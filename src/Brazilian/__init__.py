#######################################
# IMPORTS
#######################################

import string
import emoji
import os
import re
import numpy as np
import uuid
import math
import time
import sys
import string
import struct
from enum import Enum, auto
from dataclasses import dataclass
import pickle
import psutil
from typing import *
import Brazilian.Libs.basBR as basBR
import ctypes
from Brazilian.Libs.faster import fast_memorize_for_loop, fast_memorize_while_loop, fast_memorize_for_in_loop
from numba import cuda
import yaml

#######################################
# StringsWithArrowsAndMore
#######################################

def string_with_arrows(text, pos_start, pos_end):
    result = ''
    
    idx_start = max(text.rfind('\n', 0, pos_start.idx), 0)
    idx_end = text.find('\n', idx_start + 1)
    if idx_end < 0: idx_end = len(text)
        
    line_count = pos_end.ln - pos_start.ln + 1
    for i in range(line_count):
        line = text[idx_start:idx_end]
        col_start = pos_start.col if i == 0 else 0
        col_end = pos_end.col if i == line_count - 1 else len(line) - 1
        
        result += line + '\n'
        result += ' ' * col_start + '^' * (col_end - col_start)
        idx_start = idx_end
        idx_end = text.find('\n', idx_start + 1)
        if idx_end < 0: idx_end = len(text)
            
    return result.replace('\t', '')

def to_bytes(value):
    if isinstance(value, bytes):
        return value
    elif isinstance(value, str):
        return value.encode('utf-8')
    elif isinstance(value, int):
        return value.to_bytes((value.bit_length() + 7) // 8 or 1, byteorder='big', signed=True)
    elif isinstance(value, float):
        return struct.pack('>d', value)
    elif isinstance(value, bool):
        return b'\x01' if value else b'\x00'
    elif isinstance(value, (list, tuple, dict, set)):
        return pickle.dumps(value)
    elif hasattr(value, '__bytes__'):
        return bytes(value)
    else:
        return pickle.dumps(value)

def to_bytes_forL(value):
  val = str(value)

  if "." in val:
    return to_bytes(float(val))
  if val.isdigit():
    return to_bytes(int(val))
  else:
    return to_bytes(val)

def convert_forL(value):
  val = str(value)

  if "." in val:
    return float(val)
  if val.isdigit():
    return int(val)
  else:
    return val

def convert_types_to_values(value):
  if isinstance(value, (int, float)):
    return Number(value)
  elif isinstance(value, (str)):
    return String(value)
  elif isinstance(value, (list)):
    return List(value)
  else:
    return

def lineTry(Try: Any, Except: any, exception:Any = None):
  try:
    return Try
  except (exception if Exception == None else Exception) as e:
    return Except

def origin_module(obj):
    mod = obj.__class__.__module__
    
    if mod in ("builtins", "__main__"):
        return "."

    return mod.split(".")[0]

def br_args_to_python(args):
    n_args = []
    
    for arg in args:
        if isinstance(arg, List):
            n_args.append(br_args_to_python(arg.value))
        else:
            n_args.append(arg.value)
    
    return n_args

def br_args_to_python_for_extern(args):
    n_args = []
    
    for arg in args:
        n_args.append(repr(arg.value))
    
    return n_args

_MEMORY_REGISTRY = []

def _keep_alive(obj):
  _MEMORY_REGISTRY.append(obj)

def is_normal_double(val):
  bits = struct.unpack('>Q', struct.pack('>d', val))[0]

  exponent = (bits >> 52) & 0x7FF
  mantissa = bits & ((1 << 52) - 1)

  return exponent != 0 and exponent != 0x7FF

def is_address_mapped(address: int) -> bool:
  if not isinstance(address, int) or address <= 0:
    return False
  try:
    with open("/proc/self/maps", "r") as f:
      for line in f:
        m = re.match(r'^([0-9a-fA-F]+)-([0-9a-fA-F]+)\s', line)
        if m:
          start = int(m.group(1), 16)
          end = int(m.group(2), 16)
          if start <= address < end:
            return True
  except Exception:
    return False
  return False

#######################################
# OPEN FILES (so they don't get automatically closed by GC)
#######################################

files = {}

#######################################
# CONSTANTS
#######################################

IMPORT_PATH_NAME = ".path"

if not os.path.isfile(IMPORT_PATH_NAME):
  IMPORT_PATH = """

dependencies:
  math: "./TheBrazilianProgrammingLanguageTests/modules/math.br"
  os: "./TheBrazilianProgrammingLanguageTests/modules/os.br"
  random: "./TheBrazilianProgrammingLanguageTests/modules/random.br"
  std: "./TheBrazilianProgrammingLanguageTests/modules/std.br"
  string: "./TheBrazilianProgrammingLanguageTests/modules/string.br"

"""

  with open(IMPORT_PATH_NAME, "w") as f:
    f.write(IMPORT_PATH)
else:
  with open(IMPORT_PATH_NAME, "r") as f:
    IMPORT_PATH = f.read()

def VALID_IDENTIFIERS(s: str) -> bool:
    forbidden = r'[=\[\]\^%}{|~`\'\"+\-&()/?!*:;#@<>]'
    
    allowed_chars = r'[•√π§∆£¢€¥^©®™✓$_]'
    
    has_allowed_chars = None
    
    if re.search(forbidden, s):
      return False
    
    if re.search(allowed_chars, s):
      has_allowed_chars = True

    allowed_blocks = (
        r'[\u0041-\u007A'
        r'\u00C0-\u00FF'
        r'\u0100-\u017F'
        r'\u1E00-\u1EFF'
        r'\u0400-\u04FF'
        r'\u0600-\u06FF'
        r'\u4E00-\u9FFF'
        r'\u0900-\u097F'
        r'\uAC00-\uD7AF'
        r'0-9_'
        r']'
    )

    has_allowed_char = bool(re.search(allowed_blocks, s))
    has_emoji = bool(emoji.emoji_count(s))

    return has_allowed_char or has_emoji or has_allowed_chars

global_variables = {}

temp_func_name = []

classes = {}

current_op = None

current_class = None

_registry = {}

CALLBACK = ctypes.CFUNCTYPE(ctypes.c_void_p)

#######################################
# ERRORS
#######################################

class Value:
  def __init__(self):
    self.set_pos()
    self.set_context()

  def set_pos(self, pos_start=None, pos_end=None):
    self.pos_start = pos_start
    self.pos_end = pos_end
    return self

  def set_context(self, context=None):
    self.context = context
    return self

  def added_to(self, other):
    return None, self.illegal_operation(other)

  def subbed_by(self, other):
    return None, self.illegal_operation(other)

  def multed_by(self, other):
    return None, self.illegal_operation(other)

  def dived_by(self, other):
    return None, self.illegal_operation(other)

  def powed_by(self, other):
    return None, self.illegal_operation(other)
  
  def percent_by(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_eq(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_ne(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_lt(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_gt(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_lte(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_gte(self, other):
    return None, self.illegal_operation(other)

  def anded_by(self, other):
    return None, self.illegal_operation(other)

  def ored_by(self, other):
    return None, self.illegal_operation(other)
  
  def xored(self, other):
    return None, self.illegal_operation(other)
  
  def left_shiffed(self, other):
    return None, self.illegal_operation(other)
  
  def right_shiffed(self, other):
    return None, self.illegal_operation(other)
  
  def bitwise_and(self, other):
    return None, self.illegal_operation(other)
  
  def bitwise_or(self, other):
    return None, self.illegal_operation(other)
  
  def bitwise_not(self, other):
    return None, self.illegal_operation(other)

  def notted(self, other):
    return None, self.illegal_operation(other)
  
  def iter(self):
    return Iterator(self.gen)
  
  def gen(self):
    yield RTResult().failure(self.illegal_operation())
  
  def get_index(self, index):
    return None, self.illegal_operation(index)

  def set_index(self, index, value):
    return None, self.illegal_operation(index, value)

  def execute(self, args):
    return RTResult().failure(self.illegal_operation())
  
  def get_dot(self, verb):
    if hasattr(self, verb):
        return getattr(self, verb), None
    return None, RTError(
        self.pos_start, self.pos_end,
        f"'{type(self).__name__}' object has no attribute '{verb}'",
        self.context
    )

  def set_dot(self, verb, value):
    return None, self.illegal_operation(verb, value)

  def copy(self):
    raise Exception('No copy method defined')

  def is_true(self):
    return False

  def illegal_operation(self, *others):
    if len(others) == 0:
      others = self,
    
    return RTError(
      self.pos_start, others[-1].pos_end,
      'Illegal operation',
      self.context
    )

class Error(Value):
  def __init__(self, pos_start, pos_end, error_name, details):
    self.pos_start = pos_start
    self.pos_end = pos_end
    self.error_name = error_name
    self.details = details
  
  def set_pos(self, pos_start=None, pos_end=None):
    return self

  def __repr__(self) -> str:
    return f'{self.error_name}: {self.details}'

  def as_string(self):
    result  = f'{self.error_name}: {self.details}\n'
    result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
    result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
    return result
  
  def copy(self):
    return __class__(self.pos_start, self.pos_end, self.error_name, self.details)

class IllegalCharError(Error):
  def __init__(self, pos_start, pos_end, details):
    super().__init__(pos_start, pos_end, 'Illegal Character', details)

class ExpectedCharError(Error):
  def __init__(self, pos_start, pos_end, details):
    super().__init__(pos_start, pos_end, 'Expected Character', details)

class InvalidSyntaxError(Error):
  def __init__(self, pos_start, pos_end, details=''):
    super().__init__(pos_start, pos_end, 'Invalid Syntax', details)

class RTError(Error):
  def __init__(self, pos_start, pos_end, details, context):
    super().__init__(pos_start, pos_end, 'Runtime Error', details)
    self.context = context

  def set_context(self, context=None):
    return self

  def as_string(self):
    result  = self.generate_traceback()
    result += f'{self.error_name}: {self.details}'
    result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
    return result

  def generate_traceback(self):
    result = ''
    pos = self.pos_start
    ctx = self.context

    while ctx:
      result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
      pos = ctx.parent_entry_pos
      ctx = ctx.parent

    return 'Traceback (most recent call last):\n' + result
  
  def copy(self):
    return __class__(self.pos_start, self.pos_end, self.details, self.context)

class TryError(RTError):
  def __init__(self, pos_start, pos_end, details, context, prev_error):
    super().__init__(pos_start, pos_end, details, context)
    self.prev_error = prev_error 
  
  def generate_traceback(self):
    result = ""
    if self.prev_error:
      result += self.prev_error.as_string()
    result += "\nDuring the handling of the above error, another error occurred:\n\n"
    return result + super().generate_traceback()
    
#######################################
# POSITION
#######################################

class Position:
  def __init__(self, idx, ln, col, fn, ftxt):
    self.idx = idx
    self.ln = ln
    self.col = col
    self.fn = fn
    self.ftxt = ftxt

  def advance(self, current_char=None):
    self.idx += 1
    self.col += 1

    if current_char == '\n':
      self.ln += 1
      self.col = 0

    return self

  def copy(self):
    return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

#######################################
# TOKENS
#######################################

class TokenType(Enum):
  INT				 = auto()
  FLOAT    	 = auto()
  STRING		 = auto()
  BIN        = auto()
  BYTES  		 = auto()
  IDENTIFIER = auto()
  TYPES      = auto()
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
  BITWISEXOR = auto()
  BITWISEOR  = auto()
  BITWISEAND = auto()
  BITWISENOT = auto()
  LEFTSH     = auto()
  RIGHTSH    = auto()
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
  'yield',
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
  'struct',
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
  
struct_name = []
class_name = []

#######################################
# LEXER
#######################################

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

TYPES = [
  "int",
  "float", 
  "double",
  "char",
  "string", 
  "bin", 
  "bytes", 
  "list", 
  "dict",
  "func",
  "bool",
]

class Lexer:
  def __init__(self, fn, text):
    self.fn = fn
    self.text = text
    self.pos = Position(-1, 0, -1, fn, text)
    self.current_char = None
    self.advance()
    self.in_num = False
    self.real_num = ''

  def advance(self):
    self.pos.advance(self.current_char)
    self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

  def make_tokens(self):
    tokens = []

    while self.current_char != None:
      if self.current_char in SINGLE_CHAR_TOKS:
        self.in_num = False

        tt = SINGLE_CHAR_TOKS[self.current_char]
        pos = self.pos.copy()
        self.advance()

        tokens.append(Token(tt, pos_start=pos))
      elif self.current_char.isspace():
        self.in_num = False

        self.advance()
      elif self.current_char == '#':
        self.in_num = False

        self.skip_comment()
      elif self.current_char in "0123456789":
        self.in_num = True

        tokens.append(self.make_number())
      elif self.in_num and self.current_char == '_':
        self.advance()

        self.in_num = False
      elif VALID_IDENTIFIERS(self.current_char):
        self.in_num = False

        tokens.append(self.make_identifier())
      elif self.current_char == '$':
        self.in_num = False
          
        tokens.append(self.make_identifier())
      elif self.current_char == '"':
        self.in_num = False

        tokens.append(self.make_string())
      elif self.current_char == '~':
        self.in_num = False

        tokens.append(self.make_bytes())
      elif self.current_char == '-':
        self.in_num = False

        tokens.append(self.make_minus_or_arrow())
      elif self.current_char == '!':
        self.in_num = False

        token, error = self.make_not_equals()
        if error: return [], error
        tokens.append(token)
      elif self.current_char == '\\':
        self.in_num = False

        self.advance()
      elif self.current_char == '=':
        self.in_num = False

        tokens.append(self.make_equals())
      elif self.current_char == '<':
        self.in_num = False

        tokens.append(self.make_less_than())
      elif self.current_char == '>':
        self.in_num = False

        tokens.append(self.make_greater_than())
      elif self.current_char == '|':
        self.in_num = False

        tokens.append(self.make_bitwise_or())
      elif self.current_char == '&':
        self.in_num = False

        tokens.append(self.make_bitwise_and())
      elif self.current_char == '°':
        self.in_num = False

        tokens.append(self.make_bitwise_xor())
      else:
        pos_start = self.pos.copy()
        char = self.current_char
        self.advance()
        return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

    self.in_num = False

    tokens.append(Token(TokenType.EOF, pos_start=self.pos))
    return tokens, None

  def make_number(self):
    num_str = ''
    dot_count = 0
    pos_start = self.pos.copy()
    val = None

    while self.current_char != None and self.current_char in "0123456789" + '.':
      if self.current_char == '.':
        if dot_count == 1: break
        dot_count += 1
      elif self.current_char == '_':
        self.advance()
        pass

      num_str += self.current_char
      self.advance()

    self.real_num += num_str

    if dot_count == 0 and self.current_char != '_':
      val = Token(TokenType.INT, int(self.real_num), pos_start, self.pos)
      
      self.real_num = ''
    else:
      if self.current_char != '_':
        val = Token(TokenType.FLOAT, float(self.real_num), pos_start, self.pos)

        self.real_num = ''
      else:
        self.advance()

    return val

  def make_string(self):
    string = ''
    pos_start = self.pos.copy()
    escape_character = False
    self.advance()

    while self.current_char != None and (self.current_char != '"' or escape_character):
      if escape_character:
        escape_character = False
      elif self.current_char == '\\':
        escape_character = True
      string += self.current_char
      self.advance()
    
    self.advance()
    return Token(TokenType.STRING, string.encode('raw_unicode_escape').decode('unicode_escape'), pos_start, self.pos)

  def make_bytes(self):
    bytes = ''
    pos_start = self.pos.copy()
    self.advance()

    while self.current_char != None and (self.current_char != '~'):
      bytes += self.current_char
      self.advance()
    
    self.advance()

    return Token(TokenType.BYTES, to_bytes(bytes.encode('raw_unicode_escape').decode('unicode_escape')), pos_start, self.pos)

  def make_identifier(self):
    id_str = ''
    pos_start = self.pos.copy()

    while self.current_char != None and VALID_IDENTIFIERS(self.current_char)or self.current_char != None and self.current_char == '$':
      id_str += self.current_char
      self.advance()

    tok_type = TokenType.KEYWORD if id_str in KEYWORDS else (TokenType.TYPES if id_str in TYPES else TokenType.IDENTIFIER)
    return Token(tok_type, id_str, pos_start, self.pos)

  def make_minus_or_arrow(self):
    tok_type = TokenType.MINUS
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '>':
      self.advance()
      tok_type = TokenType.ARROW

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def make_not_equals(self):
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      return Token(TokenType.NE, pos_start=pos_start, pos_end=self.pos), None

    self.advance()
    return None, ExpectedCharError(pos_start, self.pos, "'=' (after '!')")

  def make_equals(self):
    tok_type = TokenType.EQ
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      tok_type = TokenType.EE

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def make_less_than(self):
    tok_type = TokenType.LT
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      tok_type = TokenType.LTE
    elif self.current_char == '<':
      self.advance()
      tok_type = TokenType.LEFTSH

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def make_greater_than(self):
    tok_type = TokenType.GT
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      tok_type = TokenType.GTE
    if self.current_char == '>':
      self.advance()
      tok_type = TokenType.RIGHTSH

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def make_bitwise_xor(self):
    tok_type = TokenType.BITWISEXOR
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '°':
      tok_type = TokenType.BITWISENOT

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)
  
  def make_bitwise_or(self):
    tok_type = TokenType.BITWISEOR
    pos_start = self.pos.copy()
    self.advance()

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def make_bitwise_and(self):
    tok_type = TokenType.BITWISEAND
    pos_start = self.pos.copy()
    self.advance()

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def skip_comment(self):
    multi_line_comment = False
    self.advance()
    if self.current_char == "*":
      multi_line_comment = True

    while self.current_char is not None:
      if self.current_char == "*" and multi_line_comment:
        self.advance()
        if self.current_char != "#": continue
        else: break
      elif self.current_char == "\n" and not multi_line_comment: break
      self.advance()

    self.advance()

#######################################
# NODES
#######################################

class NumberNode:
  def __init__(self, tok):
    self.tok = tok

    self.pos_start = self.tok.pos_start
    self.pos_end = self.tok.pos_end

  def __repr__(self):
    return f'{self.tok}'
  
class BinNode:
  def __init__(self, tok):
    self.tok = tok

    self.pos_start = self.tok.pos_start
    self.pos_end = self.tok.pos_end

  def __repr__(self):
    return f'{self.tok}'
  
class ByteNode:
  def __init__(self, tok):
    self.tok = tok

    self.pos_start = self.tok.pos_start
    self.pos_end = self.tok.pos_end

  def __repr__(self):
    return f'{self.tok}'

class StringNode:
  def __init__(self, tok):
    self.tok = tok

    self.pos_start = self.tok.pos_start
    self.pos_end = self.tok.pos_end

  def __repr__(self):
    return f'{self.tok}'

class ListNode:
  def __init__(self, element_nodes, pos_start, pos_end):
    self.element_nodes = element_nodes

    self.pos_start = pos_start
    self.pos_end = pos_end

class VarAccessNode:
  def __init__(self, var_name_tok):
    self.var_name_tok = var_name_tok

    self.pos_start = self.var_name_tok.pos_start
    self.pos_end = self.var_name_tok.pos_end

class VarAssignNode:
  def __init__(self, var_name_tok, value_node, is_const=False, is_global=False, current_op=[TokenType.EQ]):
    self.var_name_tok = var_name_tok
    self.value_node = value_node
    self.is_const = is_const
    self.is_global = is_global
    self.current_op = [item for item in current_op if item != "" and item is not None]

    self.pos_start = self.var_name_tok.pos_start
    self.pos_end = self.value_node.pos_end
  
  def __repr__(self):
    const_global = "CONST " if self.is_const  else "GLOBAL" if self.is_global else ""
    return f"({const_global}{self.var_name_tok} = {self.value_node!r})"

class BinOpNode:
  def __init__(self, left_node, op_tok, right_node):
    self.left_node = left_node
    self.op_tok = op_tok
    self.right_node = right_node

    self.pos_start = self.left_node.pos_start
    self.pos_end = self.right_node.pos_end

  def __repr__(self):
    return f'({self.left_node}, {self.op_tok}, {self.right_node})'

class UnaryOpNode:
  def __init__(self, op_tok, node):
    self.op_tok = op_tok
    self.node = node

    self.pos_start = self.op_tok.pos_start
    self.pos_end = node.pos_end

  def __repr__(self):
    return f'({self.op_tok}, {self.node})'

class IfNode:
  def __init__(self, cases, else_case):
    self.cases = cases
    self.else_case = else_case

    self.pos_start = self.cases[0][0].pos_start
    self.pos_end = (self.else_case or self.cases[len(self.cases) - 1])[0].pos_end

class ForNode:
  def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node, should_return_null):
    self.var_name_tok = var_name_tok
    self.start_value_node = start_value_node
    self.end_value_node = end_value_node
    self.step_value_node = step_value_node
    self.body_node = body_node
    self.should_return_null = should_return_null

    self.pos_start = self.var_name_tok.pos_start
    self.pos_end = self.body_node.pos_end

class WhileNode:
  def __init__(self, condition_node, body_node, should_return_null):
    self.condition_node = condition_node
    self.body_node = body_node
    self.should_return_null = should_return_null

    self.pos_start = self.condition_node.pos_start
    self.pos_end = self.body_node.pos_end

class FuncDefNode:
  def __init__(self, var_name_tok, arg_name_toks, defaults, dynamics, body_node, should_auto_return):
    self.var_name_tok = var_name_tok
    self.arg_name_toks = arg_name_toks
    self.defaults = defaults
    self.dynamics = dynamics
    self.body_node = body_node
    self.should_auto_return = should_auto_return

    if self.var_name_tok:
      self.pos_start = self.var_name_tok.pos_start
    elif len(self.arg_name_toks) > 0:
      self.pos_start = self.arg_name_toks[0].pos_start
    else:
      self.pos_start = self.body_node.pos_start

    self.pos_end = self.body_node.pos_end

class CallNode:
  def __init__(self, node_to_call, arg_nodes):
    self.node_to_call = node_to_call
    self.arg_nodes = arg_nodes

    self.pos_start = self.node_to_call.pos_start

    if len(self.arg_nodes) > 0:
      self.pos_end = self.arg_nodes[len(self.arg_nodes) - 1].pos_end
    else:
      self.pos_end = self.node_to_call.pos_end

class ReturnNode:
  def __init__(self, node_to_return, pos_start, pos_end):
    self.node_to_return = node_to_return

    self.pos_start = pos_start
    self.pos_end = pos_end

class YieldNode:
  def __init__(self, node_to_yield, pos_start, pos_end):
    self.node_to_yield = node_to_yield

    self.pos_start = pos_start
    self.pos_end = pos_end

class ContinueNode:
  def __init__(self, pos_start, pos_end):
    self.pos_start = pos_start
    self.pos_end = pos_end

class BreakNode:
  def __init__(self, pos_start, pos_end):
    self.pos_start = pos_start
    self.pos_end = pos_end

class PassNode:
  def __init__(self, pos_start, pos_end):
    self.pos_start = pos_start
    self.pos_end = pos_end

@dataclass
class ImportNode:
  string_node: StringNode
  pos_start: Position
  pos_end: Position

  def __repr__(self) -> str:
    return f"IMPORT {self.string_node!r}"

@dataclass
class DoNode:
  statements: ListNode
  pos_start: Position
  pos_end: Position

  def __repr__(self) -> str:
    return f'(DO {self.statements!r} END)'

@dataclass
class TryNode:
  try_block: ListNode
  exc_iden: Token
  catch_block: Any
  pos_start: Position
  pos_end: Position

  def __repr__(self) -> str:
    return f'(TRY {self.try_block!r} CATCH AS {self.exc_iden!r} THEN {self.catch_block!r})'

@dataclass
class ForInNode:
  var_name_tok: Token
  iterable_node: Any
  body_node: Any
  pos_start: Position
  pos_end: Position
  should_return_null: bool
  should_yield_null: bool

  def __repr__(self) -> str:
    return f"(FOR {self.var_name_tok} IN {self.iterable_node!r} THEN {self.body_node!r})"

@dataclass
class IndexGetNode:
  indexee: Any
  index: Any
  pos_start: Position
  pos_end: Position

  def __repr__(self):
    return f"({self.indexee!r}[{self.index!r}])"

@dataclass
class IndexSetNode:
  indexee: Any
  index: Any
  value: Any
  pos_start: Position
  pos_end: Position

  def __repr__(self):
    return f"({self.indexee!r}[{self.index!r}]={self.value!r})"

@dataclass
class DictNode:
  pairs: Tuple[Any, Any]
  pos_start: Position
  pos_end: Position

  def __repr__(self):
    result = "({"
    for key, value in self.pairs:
      result += f"{key!r}: {value!r}"
    return result + "})"

INDENTATION = 4

@dataclass
class SwitchNode:
  condition: Any
  cases: list[Tuple[Any, ListNode]]
  else_case: ListNode
  pos_start: Position
  pos_end: Position

  def __repr__(self):
    return f"(SWITCH {self.condition!r}\n " + (" " * INDENTATION) + ("\n "+ " " * INDENTATION).join(
      f"CASE {case_cond!r}\n " + (" " * INDENTATION * 2) + f"{case_body!r}" for case_cond, case_body in list(self.cases)
    ) + "\n " + (" " * INDENTATION) + "ELSE\n" + (" " * INDENTATION * 2) + f"{self.else_case!r})"

@dataclass
class DotGetNode:
  noun: Any
  verb: Token
  pos_start: Position
  pos_end: Position

  def __repr__(self):
    return f"({self.noun!r}.{self.verb.value})"

@dataclass
class DotSetNode:
  noun: Any
  verb: Token
  value: Any
  pos_start: Position
  pos_end: Position

  def __repr__(self):
    return f"({self.noun!r}.{self.verb.value}={self.value!r})"

@dataclass
class NamespaceNode:
  name: Optional[Token]
  body: Any
  pos_start: Position
  pos_end: Position

  def __repr__(self):
    return f"""(NAMESPACE {
      self.name.value if self.name is not None
      else ""
    }\n{self.body!r}\nEND)"""
  
@dataclass
class StructNode:
    name: str
    fields: list[str]
    pos_start: Position
    pos_end: Position

    def __repr__(self):
        return f"STRUCT {self.name}: {', '.join(self.fields)}"

@dataclass
class ClassNode:
    name: str
    fields: list[str]
    pos_start: Position
    pos_end: Position

    def __repr__(self):
        return f"Class {self.name}: {', '.join([a for a in self.fields])}"


@dataclass
class StructCreationNode:
    name: str
    pos_start: Optional[Position] = None
    pos_end: Optional[Position] = None

    def __repr__(self):
        return f"{self.name}{{}}"
    
@dataclass
class ClassCreationNode:
    name: str
    pos_start: Optional[Position] = None
    pos_end: Optional[Position] = None

    def __repr__(self):
        return f"{self.name}{{}}"

#######################################
# PARSE RESULT
#######################################

class ParseResult:
  def __init__(self):
    self.error = None
    self.node = None
    self.last_registered_advance_count = 0
    self.advance_count = 0
    self.to_reverse_count = 0

  def register_advancement(self):
    self.last_registered_advance_count = 1
    self.advance_count += 1

  
  def register(self, res):
    self.last_registered_advance_count = res.advance_count
    self.advance_count += res.advance_count
    if res.error: self.error = res.error
    return res.node

  
  def try_register(self, res):
    if res.error:
      self.to_reverse_count = res.advance_count
      return None
    return self.register(res)

  
  def success(self, node):
    self.node = node
    return self

  
  def failure(self, error):
    if not self.error or self.last_registered_advance_count == 0:
      self.error = error
    return self

#######################################
# PARSER
#######################################

class Parser:
  def __init__(self, tokens):
    self.tokens = tokens
    self.tok_idx = -1
    dummy = ParseResult()
    self.advance(dummy)

  def advance(self, res: ParseResult):
    self.tok_idx += 1
    self.update_current_tok()
    res.register_advancement()
    return self.current_tok

  
  def reverse(self, amount=1):
    self.tok_idx -= amount
    self.update_current_tok()
    return self.current_tok

  def update_current_tok(self):
    if self.tok_idx >= 0 and self.tok_idx < len(self.tokens):
      self.current_tok = self.tokens[self.tok_idx]

  def parse(self):
    res = self.statements()

    if not res.error and self.current_tok.type != TokenType.EOF:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Token cannot appear after previous tokens"
      ))
  
    return res

  ###################################

  def statements(self):
    res = ParseResult()
    statements = []
    pos_start = self.current_tok.pos_start.copy()

    while self.current_tok.type == TokenType.NEWLINE:
      self.advance(res)

    statement = res.register(self.statement())
    if res.error: return res
    statements.append(statement)

    more_statements = True

    while True:
      newline_count = 0
      while self.current_tok.type == TokenType.NEWLINE:
        self.advance(res)
        newline_count += 1
      if newline_count == 0:
        more_statements = False
      
      if not more_statements: break
      statement = res.try_register(self.statement())
      if not statement:
        self.reverse(res.to_reverse_count)
        more_statements = False
        continue
      statements.append(statement)

    return res.success(ListNode(
      statements,
      pos_start,
      self.current_tok.pos_end.copy()
    ))

  def statement(self):
    res = ParseResult()
    pos_start = self.current_tok.pos_start.copy()

    if self.current_tok.matches(TokenType.KEYWORD, 'return'):
      self.advance(res)

      expr = res.try_register(self.expr())
      if not expr:
        self.reverse(res.to_reverse_count)
      return res.success(ReturnNode(expr, pos_start, self.current_tok.pos_start.copy()))
    
    if self.current_tok.matches(TokenType.KEYWORD, 'yield'):
      self.advance(res)

      expr = res.try_register(self.expr())
      if not expr:
        self.reverse(res.to_reverse_count)
      
      return res.success(YieldNode(expr, pos_start, self.current_tok.pos_start.copy()))
    
    if self.current_tok.matches(TokenType.KEYWORD, 'continue'):
      self.advance(res)
      return res.success(ContinueNode(pos_start, self.current_tok.pos_start.copy()))
      
    if self.current_tok.matches(TokenType.KEYWORD, 'break'):
      self.advance(res)
      return res.success(BreakNode(pos_start, self.current_tok.pos_start.copy()))
    
    if self.current_tok.matches(TokenType.KEYWORD, 'pass'):
      self.advance(res)
      return res.success(PassNode(pos_start, self.current_tok.pos_start.copy()))
    
    if self.current_tok.matches(TokenType.KEYWORD, 'import'):
      self.advance(res)

      if not self.current_tok.type == TokenType.STRING:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected string"
        ))
      
      string = res.register(self.atom())
      return res.success(ImportNode(string, pos_start, self.current_tok.pos_start.copy()))

    if self.current_tok.matches(TokenType.KEYWORD, 'try'):
      self.advance(res)
      try_node = res.register(self.try_statement())
      return res.success(try_node)
    
    if self.current_tok.matches(TokenType.KEYWORD, 'switch'):
      self.advance(res)
      switch_node = res.register(self.switch_statement())
      return res.success(switch_node)
    
    if self.current_tok.matches(TokenType.KEYWORD, 'struct'):
      self.advance(res)
      struct_node = res.register(self.struct_def())
      return res.success(struct_node)
    
    if self.current_tok.matches(TokenType.KEYWORD, 'class'):
      self.advance(res)
      class_node = res.register(self.class_def())
      return res.success(class_node)

    expr = res.register(self.expr())
    if res.error:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected 'switch', 'return', 'continue', 'break', 'if', 'for', 'while', 'function', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
      ))
    return res.success(expr)

  def expr(self):
    res = ParseResult()
    global current_op

    var_assign_node = res.try_register(self.assign_expr())
    if var_assign_node: return res.success(var_assign_node)
    else: self.reverse(res.to_reverse_count)

    if self.current_tok.matches(TokenType.KEYWORD, "const"):
      self.advance(res)

      if self.current_tok.type != TokenType.IDENTIFIER:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected identifier"
        ))

      identifier = self.current_tok

      self.advance(res)

      if self.current_tok.type in [TokenType.PLUS, TokenType.MINUS, TokenType.MUL, TokenType.DIV, TokenType.PERC, TokenType.POW]:
        current_op = self.current_tok.type

        self.advance(res)

        if self.current_tok.type != TokenType.EQ:
          current_op = None

      if self.current_tok.type != TokenType.EQ:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected '='"
        ))
      
      self.advance(res)

      assign_expr = res.register(self.expr())
      if res.error: return res

      return res.success(VarAssignNode(identifier, assign_expr, is_const=True, is_global=False, current_op=[current_op if current_op else None, TokenType.EQ]))
    
    if self.current_tok.matches(TokenType.KEYWORD, "global"):
      self.advance(res)

      if self.current_tok.type != TokenType.IDENTIFIER:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected identifier"
        ))

      identifier = self.current_tok

      self.advance(res)

      if self.current_tok.type in [TokenType.PLUS, TokenType.MINUS, TokenType.MUL, TokenType.DIV, TokenType.PERC, TokenType.POW]:
        current_op = self.current_tok.type

        self.advance(res)

        if self.current_tok.type != TokenType.EQ:
          current_op = None

      if self.current_tok.type != TokenType.EQ:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected '='"
        ))
      
      self.advance(res)

      assign_expr = res.register(self.expr())
      if res.error: return res

      return res.success(VarAssignNode(identifier, assign_expr, is_const=False, is_global=True, current_op=[current_op if current_op else None, TokenType.EQ]))

    node = res.register(self.bin_op(self.comp_expr, ((TokenType.KEYWORD, 'and'), (TokenType.KEYWORD, 'or'))))

    if res.error:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected 'if', 'for', 'while', 'function', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
      ))
    
    if self.current_tok.type == TokenType.EQ:
      return res.failure(InvalidSyntaxError(
        node.pos_start, node.pos_end,
        "Invalid assignment"
      ))

    return res.success(node)

  def assign_expr(self):
      res = ParseResult()
      pos_start = self.current_tok.pos_start
      global current_op

      if self.current_tok.type != TokenType.IDENTIFIER: 
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected 'if', 'for', 'while', 'function', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
        ))
    
      var_name_tok = self.current_tok

      self.advance(res)
      if self.current_tok.value in TYPES:
        Type = self.current_tok.value
        self.advance(res)

        if self.current_tok.type in [TokenType.PLUS, TokenType.MINUS, TokenType.MUL, TokenType.DIV, TokenType.PERC, TokenType.POW]:
          current_op = self.current_tok.type

          self.advance(res)

          if self.current_tok.type != TokenType.EQ:
            current_op = None

        if self.current_tok.type != TokenType.EQ:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected '='"
          ))
    
        self.advance(res)
        is_type = self.current_tok.value

        assign_expr = res.register(self.expr())
        if res.error: return res

        if Type == "int":
          if ((isinstance(assign_expr, NumberNode) or isinstance(assign_expr, CallNode) or isinstance(assign_expr, VarAccessNode) or isinstance(assign_expr, BaseFunction)) and isinstance(is_type, int)) or (True if is_type in ['null'] else False):
            return res.success(VarAssignNode(var_name_tok, assign_expr, current_op=[current_op if current_op else None, TokenType.EQ]))
          else:
            return res.failure(ValueError(
              self.current_tok.pos_start, self.current_tok.pos_end,
              f"The variable cant's be int"
            ))
        elif Type == "float":
          if ((isinstance(assign_expr, NumberNode) or isinstance(assign_expr, CallNode) or isinstance(assign_expr, VarAccessNode)) and isinstance(is_type, float)) or (True if is_type in ['null'] else False):
            return res.success(VarAssignNode(var_name_tok, assign_expr, current_op=[current_op if current_op else None, TokenType.EQ]))
          else:
            return res.failure(ValueError(
              self.current_tok.pos_start, self.current_tok.pos_end,
              f"The variable cant's be float"
            ))
        elif Type == "char":
          if ((isinstance(assign_expr, StringNode) or isinstance(assign_expr, CallNode) or isinstance(assign_expr, VarAccessNode)) and (isinstance(is_type, string)) and len(is_type) == 1) or (True if is_type in ['null'] else False):
            return res.success(VarAssignNode(var_name_tok, assign_expr, current_op=[current_op if current_op else None, TokenType.EQ]))
          else:
            return res.failure(ValueError(
              self.current_tok.pos_start, self.current_tok.pos_end,
              f"The variable cant's be char"
            ))
        elif Type == "string":
          if (((isinstance(assign_expr, StringNode) or isinstance(assign_expr, CallNode) or isinstance(assign_expr, VarAccessNode)) and isinstance(is_type, str))) or (True if is_type in ['null'] else False):
            return res.success(VarAssignNode(var_name_tok, assign_expr, current_op=[current_op if current_op else None, TokenType.EQ]))
          else:
            return res.failure(ValueError(
              self.current_tok.pos_start, self.current_tok.pos_end,
              f"The variable cant's be string"
            ))
        elif Type == "list":
          if ((isinstance(assign_expr, ListNode) or isinstance(assign_expr, CallNode) or isinstance(assign_expr, VarAccessNode)) and isinstance(is_type, list)) or (True if is_type in ['null'] else False):
            return res.success(VarAssignNode(var_name_tok, assign_expr, current_op=[current_op if current_op else None, TokenType.EQ]))
          else:
            return res.failure(ValueError(
              self.current_tok.pos_start, self.current_tok.pos_end,
              f"The variable cant's be list"
            ))
        elif Type == "dict":
          if ((isinstance(assign_expr, DictNode) or isinstance(assign_expr, CallNode) or isinstance(assign_expr, VarAccessNode)) and isinstance(is_type, dict)) or (True if is_type in ['null'] else False):
            return res.success(VarAssignNode(var_name_tok, assign_expr, current_op=[current_op if current_op else None, TokenType.EQ]))
          else:
            return res.failure(ValueError(
              self.current_tok.pos_start, self.current_tok.pos_end,
              f"The variable cant's be dictionary"
            ))
        elif Type == "bool":
          if (isinstance(assign_expr, CallNode) or isinstance(assign_expr, VarAccessNode)) or (True if is_type in ['null', 'true', 'false'] else False):
            return res.success(VarAssignNode(var_name_tok, assign_expr, current_op=[current_op if current_op else None, TokenType.EQ]))
          else:
            return res.failure(ValueError(
              self.current_tok.pos_start, self.current_tok.pos_end,
              f"The variable cant's be boolean"
            ))
        elif Type == "bin":
          if (isinstance(assign_expr, BinNode) or isinstance(assign_expr, CallNode) or isinstance(assign_expr, VarAccessNode)) or (True if is_type in ['null'] else False):
            return res.success(VarAssignNode(var_name_tok, assign_expr, current_op=[current_op if current_op else None, TokenType.EQ]))
          else:
            return res.failure(ValueError(
              self.current_tok.pos_start, self.current_tok.pos_end,
              f"The variable cant's be binary"
            ))
        elif Type == "bytes":
          if (isinstance(assign_expr, ByteNode) or isinstance(assign_expr, CallNode) or isinstance(assign_expr, VarAccessNode)) or (True if is_type in ['null'] else False):
            return res.success(VarAssignNode(var_name_tok, assign_expr, current_op=[current_op if current_op else None, TokenType.EQ]))
          else:
            return res.failure(ValueError(
              self.current_tok.pos_start, self.current_tok.pos_end,
              f"The variable cant's be bytes"
            ))
        elif Type == "func":
          if (isinstance(assign_expr, FuncDefNode) or isinstance(assign_expr, CallNode) or isinstance(assign_expr, VarAccessNode)) or (True if is_type in ['null'] else False):
            return res.success(VarAssignNode(var_name_tok, assign_expr, current_op=[current_op if current_op else None, TokenType.EQ]))
          else:
            return res.failure(ValueError(
              self.current_tok.pos_start, self.current_tok.pos_end,
              f"The variable cant's be func"
            ))
        else:
          return res.failure(ValueError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"This type: '{str(self.current_tok)}' not in types."
          ))
      else:
        if self.current_tok.type in [TokenType.PLUS, TokenType.MINUS, TokenType.MUL, TokenType.DIV, TokenType.PERC, TokenType.POW]:
          current_op = self.current_tok.type

          self.advance(res)

          if self.current_tok.type != TokenType.EQ:
            current_op = None

        if self.current_tok.type != TokenType.EQ:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected '='"
          ))
    
        self.advance(res)

        assign_expr = res.register(self.expr())
        if res.error: return res

        n_current_op = current_op

        current_op = None

        return res.success(VarAssignNode(var_name_tok, assign_expr, current_op=[n_current_op if n_current_op else None, TokenType.EQ]))  

  def comp_expr(self):
    res = ParseResult()

    if self.current_tok.matches(TokenType.KEYWORD, 'not'):
      op_tok = self.current_tok
      self.advance(res)

      node = res.register(self.comp_expr())
      if res.error: return res
      return res.success(UnaryOpNode(op_tok, node))
    elif self.current_tok.type == TokenType.BITWISENOT:
      op_tok = self.current_tok
      self.advance(res)

      node = res.register(self.comp_expr())
      if res.error: return res
      return res.success(UnaryOpNode(op_tok, node))
    
    node = res.register(self.bin_op(self.bit_expr, (TokenType.EE, TokenType.NE, TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE)))
    
    if res.error:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected int, float, identifier, '+', '-', '(', '[', 'if', 'for', 'while', 'function' or 'not'"
      ))

    return res.success(node)

  def bit_expr(self):
    return self.bin_op(self.arith_expr, (TokenType.BITWISEXOR, TokenType.LEFTSH, TokenType.RIGHTSH, TokenType.BITWISEAND, TokenType.BITWISEOR))

  def arith_expr(self):
    return self.bin_op(self.term, (TokenType.PLUS, TokenType.MINUS))

  def term(self):
    return self.bin_op(self.factor, (TokenType.MUL, TokenType.DIV))

  def factor(self):
    res = ParseResult()
    tok = self.current_tok

    if tok.type in (TokenType.PLUS, TokenType.MINUS):
      self.advance(res)
      factor = res.register(self.factor())
      if res.error: return res
      return res.success(UnaryOpNode(tok, factor))

    return self.bin_op(self.call, (TokenType.POW, TokenType.PERC), self.factor)

  def call(self):
    res = ParseResult()
    func = res.register(self.index())
    if res.error: return res

    if self.current_tok.type == TokenType.LPAREN:
      self.advance(res)
      arg_nodes = []

      if self.current_tok.type == TokenType.RPAREN:
        self.advance(res)
      else:
        arg_nodes.append(res.register(self.expr()))
        if res.error:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected ')', 'if', 'for', 'while', 'function', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
          ))

        while self.current_tok.type == TokenType.COMMA:
          self.advance(res)

          arg_nodes.append(res.register(self.expr()))
          if res.error: return res

        if self.current_tok.type != TokenType.RPAREN:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected ',' or ')'"
          ))

        self.advance(res)

      return res.success(CallNode(func, arg_nodes))
    return res.success(func)

  def index(self):
    res = ParseResult()
    noun = res.register(self.dot())
    if res.error: return res

    node = noun
    while self.current_tok.type == TokenType.LSQUARE:
      self.advance(res)
      index = res.register(self.expr())
      if res.error: return res

      if self.current_tok.type != TokenType.RSQUARE:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_start,
          "Expected ']'"
        ))
      
      node = IndexGetNode(node, index, node.pos_start, self.current_tok.pos_end)
      self.advance(res)
    
    if self.current_tok.type == TokenType.EQ and isinstance(node, IndexGetNode):
      self.advance(res)

      value = res.register(self.expr())
      if res.error: return res

      node = IndexSetNode(node.indexee, node.index, value, node.pos_start, self.current_tok.pos_end)
    
    return res.success(node)

  def dot(self):
    res = ParseResult()
    noun = res.register(self.atom())
    if res.error: return res

    node = noun
    while self.current_tok.type == TokenType.DOT:
      self.advance(res)

      if self.current_tok.type != TokenType.IDENTIFIER:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_start,
          "Expected identifier"
        ))

      node = DotGetNode(node, self.current_tok, node.pos_start, self.current_tok.pos_end)
      self.advance(res)
    
    if self.current_tok.type == TokenType.EQ and isinstance(node, DotGetNode):
      self.advance(res)

      value = res.register(self.expr())
      if res.error: return res

      node = DotSetNode(node.noun, node.verb, value, node.pos_start, self.current_tok.pos_end)

    return res.success(node)

  def atom(self):
    res = ParseResult()
    tok = self.current_tok
    node = None
    global class_name
    global struct_name

    if tok.type in (TokenType.INT, TokenType.FLOAT):
      self.advance(res)
      node = NumberNode(tok)

    elif tok.type == TokenType.STRING:
      self.advance(res)
      node = StringNode(tok)
    
    elif tok.type == TokenType.BIN:
      self.advance(res)
      node = BinNode(tok)
    
    elif tok.type == TokenType.BYTES:
      self.advance(res)
      node = ByteNode(tok)

    elif tok.type == TokenType.IDENTIFIER:
      self.advance(res)
      if tok.value in struct_name:
        if self.current_tok.type == TokenType.LCURLY:
          structt_name = tok.value
          self.advance(res)
          if self.current_tok.type != TokenType.RCURLY:
              return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '}'"
            ))
          self.advance(res)
          node = StructCreationNode(structt_name)
      elif tok.value in class_name:
        if self.current_tok.type == TokenType.LPAREN:
          clas_name = tok.value
          self.advance(res)
          if self.current_tok.type != TokenType.RPAREN:
              return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '}'"
            ))
          self.advance(res)
          node = ClassCreationNode(clas_name)
      else:
        node = VarAccessNode(tok)

    elif tok.type == TokenType.LPAREN:
      self.advance(res)
      expr = res.register(self.expr())
      if res.error: return res
      if self.current_tok.type == TokenType.RPAREN:
        self.advance(res)
        node = expr
      else:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected ')'"
        ))

    elif tok.type == TokenType.LSQUARE:
      list_expr = res.register(self.list_expr())
      if res.error: return res
      node = list_expr
    
    elif tok.matches(TokenType.KEYWORD, 'if'):
      if_expr = res.register(self.if_expr())
      if res.error: return res
      node = if_expr

    elif tok.matches(TokenType.KEYWORD, 'for'):
      for_expr = res.register(self.for_expr())
      if res.error: return res
      node = for_expr

    elif tok.matches(TokenType.KEYWORD, 'while'):
      while_expr = res.register(self.while_expr())
      if res.error: return res
      node = while_expr

    elif tok.matches(TokenType.KEYWORD, 'function'):
      func_def = res.register(self.func_def())
      if res.error: return res
      node = func_def
    
    elif tok.matches(TokenType.KEYWORD, 'namespace'):
      ns_expr = res.register(self.namespace_expr())
      if res.error: return res
      node = ns_expr

    elif tok.matches(TokenType.KEYWORD, 'do'):
      do_expr = res.register(self.do_expr())
      if res.error: return res
      node = do_expr
    
    elif tok.type == TokenType.LCURLY:
      dict_expr = res.register(self.dict_expr())
      if res.error: return res
      node = dict_expr

    if node is None:
      return res.failure(InvalidSyntaxError(
        tok.pos_start, tok.pos_end,
        "Expected int, float, identifier, '+', '-', '(', '[', if', 'for', 'while', 'function'"
      ))
    
    return res.success(node)

  def list_expr(self):
    res = ParseResult()
    element_nodes = []
    pos_start = self.current_tok.pos_start.copy()

    if self.current_tok.type != TokenType.LSQUARE:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '['"
      ))

    self.advance(res)

    if self.current_tok.type == TokenType.RSQUARE:
      self.advance(res)
    else:
      element_nodes.append(res.register(self.expr()))
      if res.error:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected ']', 'if', 'for', 'while', 'function', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
        ))

      while self.current_tok.type == TokenType.COMMA:
        self.advance(res)

        element_nodes.append(res.register(self.expr()))
        if res.error: return res

      if self.current_tok.type != TokenType.RSQUARE:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected ',' or ']'"
        ))

      self.advance(res)

    return res.success(ListNode(
      element_nodes,
      pos_start,
      self.current_tok.pos_end.copy()
    ))

  def dict_expr(self):
    res = ParseResult()
    pairs = []
    pos_start = self.current_tok.pos_start.copy()

    if self.current_tok.type != TokenType.LCURLY:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected '{'"
      ))

    self.advance(res)

    if self.current_tok.type == TokenType.RCURLY:
      self.advance(res)
    else:
      key = res.register(self.expr())
      if res.error: return res

      if self.current_tok.type != TokenType.COLON:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected ':'"
        ))
      
      self.advance(res)
        
      value = res.register(self.expr())
      if res.error: return res

      pairs.append((key, value))

      while self.current_tok.type == TokenType.COMMA:
        self.advance(res)

        key = res.register(self.expr())
        if res.error: return res

        if self.current_tok.type != TokenType.COLON:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected ':'"
          ))
        
        self.advance(res)

        value = res.register(self.expr())
        if res.error: return res

        pairs.append((key, value))

      if self.current_tok.type != TokenType.RCURLY:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected ',' or '}'"
        ))

      self.advance(res)

    return res.success(DictNode(
      pairs,
      pos_start,
      self.current_tok.pos_end.copy()
    ))

  def if_expr(self):
    res = ParseResult()
    all_cases = res.register(self.if_expr_cases('if'))
    if res.error: return res
    cases, else_case = all_cases
    return res.success(IfNode(cases, else_case))

  def if_expr_b(self):
    return self.if_expr_cases('elif')

  def if_expr_c(self):
    res = ParseResult()
    else_case = None

    if self.current_tok.matches(TokenType.KEYWORD, 'else'):
      self.advance(res)

      if self.current_tok.type == TokenType.NEWLINE:
        self.advance(res)

        statements = res.register(self.statements())
        if res.error: return res
        else_case = (statements, True)

        if self.current_tok.matches(TokenType.KEYWORD, 'end'):
          self.advance(res)
        else:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected 'end'"
          ))
      else:
        expr = res.register(self.statement())
        if res.error: return res
        else_case = (expr, False)

    return res.success(else_case)

  def if_expr_b_or_c(self):
    res = ParseResult()
    cases, else_case = [], None

    if self.current_tok.matches(TokenType.KEYWORD, 'elif'):
      all_cases = res.register(self.if_expr_b())
      if res.error: return res
      cases, else_case = all_cases
    else:
      else_case = res.register(self.if_expr_c())
      if res.error: return res
    
    return res.success((cases, else_case))

  def if_expr_cases(self, case_keyword):
    res = ParseResult()
    cases = []
    else_case = None

    if not self.current_tok.matches(TokenType.KEYWORD, case_keyword):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '{case_keyword}'"
      ))

    self.advance(res)

    condition = res.register(self.expr())
    if res.error: return res

    if not self.current_tok.matches(TokenType.KEYWORD, 'then'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'then'"
      ))

    self.advance(res)

    if self.current_tok.type == TokenType.NEWLINE:
      self.advance(res)

      statements = res.register(self.statements())
      if res.error: return res
      cases.append((condition, statements, True))

      if self.current_tok.matches(TokenType.KEYWORD, 'end'):
        self.advance(res)
      else:
        all_cases = res.register(self.if_expr_b_or_c())
        if res.error: return res
        new_cases, else_case = all_cases
        cases.extend(new_cases)
    else:
      expr = res.register(self.statement())
      if res.error: return res
      cases.append((condition, expr, False))

      all_cases = res.register(self.if_expr_b_or_c())
      if res.error: return res
      new_cases, else_case = all_cases
      cases.extend(new_cases)

    return res.success((cases, else_case))

  def for_expr(self):
    res = ParseResult()
    pos_start = self.current_tok.pos_start.copy()

    if not self.current_tok.matches(TokenType.KEYWORD, 'for'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'for'"
      ))

    self.advance(res)

    if self.current_tok.type != TokenType.IDENTIFIER:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected identifier"
      ))

    var_name = self.current_tok
    self.advance(res)

    is_for_in = False

    if self.current_tok.type != TokenType.EQ and not self.current_tok.matches(TokenType.KEYWORD, "in"):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '=' or 'in'"
      ))
    elif self.current_tok.matches(TokenType.KEYWORD, "in"):
      self.advance(res)
      is_for_in = True

      iterable_node = res.register(self.expr())
      if res.error: return res
      
    else:
      self.advance(res)

      start_value = res.register(self.expr())
      if res.error: return res

      if not self.current_tok.matches(TokenType.KEYWORD, 'to'):
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected 'to'"
        ))
    
      self.advance(res)

      end_value = res.register(self.expr())
      if res.error: return res

      if self.current_tok.matches(TokenType.KEYWORD, 'step'):
        self.advance(res)

        step_value = res.register(self.expr())
        if res.error: return res
      else:
        step_value = None

    if not self.current_tok.matches(TokenType.KEYWORD, 'then'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'then'"
      ))

    self.advance(res)

    if self.current_tok.type == TokenType.NEWLINE:
      self.advance(res)

      body = res.register(self.statements())
      if res.error: return res

      if not self.current_tok.matches(TokenType.KEYWORD, 'end'):
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected 'end'"
        ))

      pos_end = self.current_tok.pos_end.copy()
      self.advance(res)

      if is_for_in:
        return res.success(ForInNode(var_name, iterable_node, body, pos_start, pos_end, True, True))
      return res.success(ForNode(var_name, start_value, end_value, step_value, body, True))
    
    body = res.register(self.statement())
    if res.error: return res

    pos_end = self.current_tok.pos_end.copy()

    if is_for_in:
      print(iterable_node)
    
      return res.success(ForInNode(var_name, iterable_node, body, pos_start, pos_end, False))
    return res.success(ForNode(var_name, start_value, end_value, step_value, body, False))

  def while_expr(self):
    res = ParseResult()

    if not self.current_tok.matches(TokenType.KEYWORD, 'while'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'while'"
      ))

    self.advance(res)

    condition = res.register(self.expr())
    if res.error: return res

    if not self.current_tok.matches(TokenType.KEYWORD, 'then'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'then'"
      ))

    self.advance(res)

    if self.current_tok.type == TokenType.NEWLINE:
      self.advance(res)

      body = res.register(self.statements())
      if res.error: return res

      if not self.current_tok.matches(TokenType.KEYWORD, 'end'):
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected 'end'"
        ))

      self.advance(res)

      return res.success(WhileNode(condition, body, True))
    
    body = res.register(self.statement())
    if res.error: return res

    return res.success(WhileNode(condition, body, False))

  def func_def(self):
    res = ParseResult()

    if not self.current_tok.matches(TokenType.KEYWORD, 'function'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'function'"
      ))

    self.advance(res)

    if self.current_tok.type == TokenType.IDENTIFIER:
      var_name_tok = self.current_tok
      self.advance(res)
      if self.current_tok.type != TokenType.LPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected '('"
        ))
    else:
      var_name_tok = None
      if self.current_tok.type != TokenType.LPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected identifier or '('"
        ))
    
    self.advance(res)
    arg_name_toks = []
    defaults = []
    dynamics = []
    hasOptionals = False

    if self.current_tok.type == TokenType.IDENTIFIER:
      pos_start = self.current_tok.pos_start.copy()
      pos_end = self.current_tok.pos_end.copy()
      arg_name_toks.append(self.current_tok)
      self.advance(res)

      if self.current_tok.type == TokenType.EQ:
        self.advance(res)
        default = res.register(self.expr())
        if res.error: return res
        defaults.append(default)
        hasOptionals = True
      elif hasOptionals:
        return res.failure(InvalidSyntaxError(
          pos_start, pos_end,
          "Expected optional parameter."
        ))
      else:
        defaults.append(None)
      
      if self.current_tok.matches(TokenType.KEYWORD, 'from'):
        self.advance(res)
        dynamics.append(res.register(self.expr()))
        if res.error: return res
      else:
        dynamics.append(None)

      
      while self.current_tok.type == TokenType.COMMA:
        self.advance(res)

        if self.current_tok.type != TokenType.IDENTIFIER:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected identifier"
          ))

        pos_start = self.current_tok.pos_start.copy()
        pos_end = self.current_tok.pos_end.copy()
        arg_name_toks.append(self.current_tok)
        self.advance(res)

        if self.current_tok.type == TokenType.EQ:
          self.advance(res)
          default = res.register(self.expr())
          if res.error: return res
          defaults.append(default)
          hasOptionals = True
        elif hasOptionals:
          return res.failure(InvalidSyntaxError(
            pos_start, pos_end,
            "Expected optional parameter."
          ))
        else:
          defaults.append(None)
        
        if self.current_tok.matches(TokenType.KEYWORD, 'from'):
          self.advance(res)
          dynamics.append(res.register(self.expr()))
          if res.error: return res
        else:
          dynamics.append(None)
      
      if self.current_tok.type != TokenType.RPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected ',', ')' or '='"
        ))
    else:
      if self.current_tok.type != TokenType.RPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected identifier or ')'"
        ))

    self.advance(res)

    if self.current_tok.type == TokenType.ARROW:
      self.advance(res)

      body = res.register(self.expr())
      if res.error: return res

      return res.success(FuncDefNode(
        var_name_tok,
        arg_name_toks,
        defaults,
        dynamics,
        body,
        True
      ))
    
    if self.current_tok.type != TokenType.NEWLINE:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '->' or newline"
      ))

    self.advance(res)

    body = res.register(self.statements())
    if res.error: return res

    if not self.current_tok.matches(TokenType.KEYWORD, 'end'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'end'"
      ))

    self.advance(res)
    
    return res.success(FuncDefNode(
      var_name_tok,
      arg_name_toks,
      defaults,
      dynamics,
      body,
      False
    ))

  def method_def(self):
    res = ParseResult()
    global temp_func_name

    if not self.current_tok.matches(TokenType.KEYWORD, 'function'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'function'"
      ))

    self.advance(res)

    temp_func_name.append(self.current_tok.value)

    if self.current_tok.type == TokenType.IDENTIFIER:
      var_name_tok = self.current_tok
      self.advance(res)
      if self.current_tok.type != TokenType.LPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected '('"
        ))
    else:
      var_name_tok = None
      if self.current_tok.type != TokenType.LPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected identifier or '('"
        ))
    
    self.advance(res)
    arg_name_toks = []
    defaults = []
    dynamics = []
    hasOptionals = False

    if self.current_tok.type == TokenType.IDENTIFIER:
      pos_start = self.current_tok.pos_start.copy()
      pos_end = self.current_tok.pos_end.copy()
      arg_name_toks.append(self.current_tok)
      self.advance(res)

      if self.current_tok.type == TokenType.EQ:
        self.advance(res)
        default = res.register(self.expr())
        if res.error: return res
        defaults.append(default)
        hasOptionals = True
      elif hasOptionals:
        return res.failure(InvalidSyntaxError(
          pos_start, pos_end,
          "Expected optional parameter."
        ))
      else:
        defaults.append(None)
      
      if self.current_tok.matches(TokenType.KEYWORD, 'from'):
        self.advance(res)
        dynamics.append(res.register(self.expr()))
        if res.error: return res
      else:
        dynamics.append(None)

      
      while self.current_tok.type == TokenType.COMMA:
        self.advance(res)

        if self.current_tok.type != TokenType.IDENTIFIER:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected identifier"
          ))

        pos_start = self.current_tok.pos_start.copy()
        pos_end = self.current_tok.pos_end.copy()
        arg_name_toks.append(self.current_tok)
        self.advance(res)

        if self.current_tok.type == TokenType.EQ:
          self.advance(res)
          default = res.register(self.expr())
          if res.error: return res
          defaults.append(default)
          hasOptionals = True
        elif hasOptionals:
          return res.failure(InvalidSyntaxError(
            pos_start, pos_end,
            "Expected optional parameter."
          ))
        else:
          defaults.append(None)
        
        if self.current_tok.matches(TokenType.KEYWORD, 'from'):
          self.advance(res)
          dynamics.append(res.register(self.expr()))
          if res.error: return res
        else:
          dynamics.append(None)
      
      if self.current_tok.type != TokenType.RPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected ',', ')' or '='"
        ))
    else:
      if self.current_tok.type != TokenType.RPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected identifier or ')'"
        ))

    self.advance(res)

    if self.current_tok.type == TokenType.ARROW:
      self.advance(res)

      body = res.register(self.expr())
      if res.error: return res

      return res.success(FuncDefNode(
        var_name_tok,
        arg_name_toks,
        defaults,
        dynamics,
        body,
        True
      ))
    
    if self.current_tok.type != TokenType.NEWLINE:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '->' or newline"
      ))

    self.advance(res)

    body = res.register(self.statements())
    if res.error: return res

    if not self.current_tok.matches(TokenType.KEYWORD, 'end'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'end'"
      ))

    self.advance(res)

    arg_name_toks.append(Token(TokenType.IDENTIFIER, "this"))
    
    return res.success(FuncDefNode(
      var_name_tok,
      arg_name_toks,
      defaults,
      dynamics,
      body,
      False
    ))

  def class_statement(self):
    res = ParseResult()
    res = ParseResult()
    pos_start = self.current_tok.pos_start.copy()

    if self.current_tok.matches(TokenType.KEYWORD, 'return'):
      self.advance(res)

      expr = res.try_register(self.expr())
      if not expr:
        self.reverse(res.to_reverse_count)
      return res.success(ReturnNode(expr, pos_start, self.current_tok.pos_start.copy()))
    
    if self.current_tok.matches(TokenType.KEYWORD, 'continue'):
      self.advance(res)
      return res.success(ContinueNode(pos_start, self.current_tok.pos_start.copy()))
      
    if self.current_tok.matches(TokenType.KEYWORD, 'break'):
      self.advance(res)
      return res.success(BreakNode(pos_start, self.current_tok.pos_start.copy()))
    
    if self.current_tok.matches(TokenType.KEYWORD, 'pass'):
      self.advance(res)
      return res.success(PassNode(pos_start, self.current_tok.pos_start.copy()))
    
    if self.current_tok.matches(TokenType.KEYWORD, 'function'):
      func_def = res.register(self.method_def())
      if res.error: return res
      return res.success(func_def)
    
    if self.current_tok.matches(TokenType.KEYWORD, 'import'):
      self.advance(res)

      if not self.current_tok.type == TokenType.STRING:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected string"
        ))
      
      string = res.register(self.atom())
      return res.success(ImportNode(string, pos_start, self.current_tok.pos_start.copy()))
  
    expr = res.register(self.expr())
    if res.error:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected 'switch', 'return', 'continue', 'break', 'if', 'for', 'while', 'function', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
      ))
    return res.success(expr)

  def class_def(self):
        res = ParseResult()
        global class_name
        global temp_func_name
        global classes

        if self.current_tok.type != TokenType.IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected identifier"
            ))

        pos_start = self.current_tok.pos_start
        clas_name = self.current_tok.value
        class_name.append(clas_name)
        self.advance(res)

        statements = []
        pos_start = self.current_tok.pos_start.copy()

        while self.current_tok.type == TokenType.NEWLINE:
          self.advance(res)

        statement = res.register(self.class_statement())
        if res.error: return res
        statements.append(statement)

        more_statements = True

        while True:
          newline_count = 0
          while self.current_tok.type == TokenType.NEWLINE:
            self.advance(res)
            newline_count += 1
          if newline_count == 0:
            more_statements = False
      
          if not more_statements: break
          statement = res.try_register(self.class_statement())
          if not statement:
            self.reverse(res.to_reverse_count)
            more_statements = False
            continue
          statements.append(statement)

        if not self.current_tok.matches(TokenType.KEYWORD, 'end'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'end' or identifier"
            ))

        pos_end = self.current_tok.pos_end
        self.advance(res)

        fields = {}

        i = 0

        for x in range(len(statements)):
          try:
            if isinstance(statements[x], FuncDefNode):
              fields[temp_func_name[i]] = statements[x]
              i += 1
            else:
              fields[statements[x].var_name_tok.value] = statements[x].value_node
          except Exception as e:
            print(e)

        classes[clas_name] = {}

        for field in fields:
          classes[class_name[class_name.index(clas_name)]][field] = fields[field]

        names = []

        for name in classes[clas_name]:
          names.append(name)

        temp_func_name = []

        return res.success(ClassNode(name=clas_name, fields=names, pos_start=pos_start, pos_end=pos_end))

  def do_expr(self):
    res = ParseResult()
    pos_start = self.current_tok.pos_start.copy()

    self.advance(res)
    
    statements = res.register(self.statements())

    if not self.current_tok.matches(TokenType.KEYWORD, 'end'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected 'end', 'return', 'continue', 'break', 'pass', 'if', 'for', 'while', 'function', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
      ))
    
    pos_end = self.current_tok.pos_end.copy()
    self.advance(res)
    return res.success(DoNode(statements, pos_start, pos_end))

  def try_statement(self):
    res = ParseResult()
    pos_start = self.current_tok.pos_start.copy()

    try_block = res.register(self.statements())
    if res.error: return res

    if not self.current_tok.matches(TokenType.KEYWORD, 'catch'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected 'catch', 'return', 'continue', 'break', 'pass', 'if', 'for', 'while', 'function', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
      ))
    
    self.advance(res)

    if not self.current_tok.matches(TokenType.KEYWORD, 'as'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected 'as'"
      ))

    self.advance(res)

    if self.current_tok.type != TokenType.IDENTIFIER:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected identifier"
      ))
    
    exc_iden = self.current_tok.copy()
    self.advance(res)

    if self.current_tok.type != TokenType.NEWLINE:
      if not self.current_tok.matches(TokenType.KEYWORD, 'then'):
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected 'then' or newline"
        ))
      
      self.advance(res)
      catch_block = res.register(self.statement())
    else:
      self.advance(res)
      catch_block = res.register(self.statements())
      if res.error: return res

      if not self.current_tok.matches(TokenType.KEYWORD, 'end'):
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected 'end', 'return', 'continue', 'break', 'pass', 'if', 'for', 'while', 'function', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
        ))
      
      self.advance(res)
    
    return res.success(TryNode(try_block, exc_iden, catch_block, pos_start, self.current_tok.pos_end.copy()))

  def switch_statement(self):
    res = ParseResult()
    pos_start = self.current_tok.pos_start

    condition = res.register(self.expr())
    if res.error: 
      (res.error)
      return res

    if self.current_tok.type != TokenType.NEWLINE:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected newline"
      ))
    self.advance(res)

    cases = []
    while self.current_tok.matches(TokenType.KEYWORD, "case"):
      self.advance(res)
      case = res.register(self.expr())
      if res.error: return res

      if self.current_tok.type != TokenType.NEWLINE:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected newline"
        ))
      self.advance(res)


      body = res.register(self.statements())

      if res.error: return res

      cases.append((case, body))
    
    else_case = None
    if self.current_tok.matches(TokenType.KEYWORD, "else"):
      self.advance(res)
      else_case = res.register(self.statements())
      if res.error: return res
    
    if not self.current_tok.matches(TokenType.KEYWORD, "end"):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected 'END'"
      ))
    
    pos_end = self.current_tok.pos_end
    self.advance(res)

    node = SwitchNode(condition, cases, else_case, pos_start, pos_end)
    return res.success(node)

  def struct_def(self):
        res = ParseResult()
        global struct_name

        if self.current_tok.type != TokenType.IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected identifier"
            ))

        pos_start = self.current_tok.pos_start
        name = self.current_tok.value
        struct_name.append(name)
        self.advance(res)

        while self.current_tok.type == TokenType.NEWLINE:
            self.advance(res)

        fields = []
        while self.current_tok.type == TokenType.IDENTIFIER:
            fields.append(self.current_tok.value)
            self.advance(res)
            while self.current_tok.type == TokenType.NEWLINE:
                self.advance(res)

        if not self.current_tok.matches(TokenType.KEYWORD, 'end'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'end' or identifier"
            ))

        pos_end = self.current_tok.pos_end
        self.advance(res)
        return res.success(StructNode(name=name, fields=fields, pos_start=pos_start, pos_end=pos_end))

  def namespace_expr(self):
    res = ParseResult()
    pos_start = self.current_tok.pos_start.copy()

    self.advance(res)
    
    statements = res.register(self.statements())

    if not self.current_tok.matches(TokenType.KEYWORD, 'end'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected 'end', 'return', 'continue', 'break', 'pass', 'if', 'for', 'while', 'function', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
      ))
    
    pos_end = self.current_tok.pos_end.copy()
    self.advance(res)
    return res.success(DoNode(statements, pos_start, pos_end))

  ###################################

  
  def bin_op(self, func_a, ops, func_b=None):
    if func_b == None:
      func_b = func_a
    
    res = ParseResult()
    left = res.register(func_a())
    if res.error: return res

    while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
      op_tok = self.current_tok
      self.advance(res)
      right = res.register(func_b())
      if res.error: return res
      left = BinOpNode(left, op_tok, right)

    return res.success(left)

#######################################
# RUNTIME RESULT
#######################################

class RTResult:
  def __init__(self):
    self.reset()

  def reset(self):
    self.value = None
    self.error = None
    self.func_return_value = None
    self.loop_should_continue = False
    self.loop_should_break = False
    self.loop_should_pass = False
    self.func_yield_value = None

  
  def register(self, res):
    self.error = res.error
    self.func_return_value = res.func_return_value
    self.func_yield_value = res.func_yield_value
    self.loop_should_continue = res.loop_should_continue
    self.loop_should_break = res.loop_should_break
    self.loop_should_pass = res.loop_should_pass
    
    return res.value

  
  def success(self, value):
    self.reset()
    
    self.value = value
    return self

  
  def success_return(self, value):
    self.reset()
    self.func_return_value = value
    return self

  def success_yield(self, value):
    self.reset()
    self.func_yield_value = value
    return self

  def success_continue(self):
    self.reset()
    self.loop_should_continue = True
    return self

  def success_break(self):
    self.reset()
    self.loop_should_break = True
    return self

  def success_pass(self):
    self.reset()
    self.loop_should_pass = True
    return self

  
  def failure(self, error):
    self.reset()
    self.error = error
    return self

  def should_return(self):
    # Note: this will allow you to continue and break outside the current function
    return (
      self.error or
      self.func_return_value or
      self.loop_should_continue or
      self.loop_should_break
    )
  
  def should_yield(self):
    # Note: this will allow you to continue and break outside the current function
    return (
      self.error or
      self.func_yield_value or
      self.loop_should_continue or
      self.loop_should_break
    )

#######################################
# VALUES
#######################################

class Number(Value):
  def __init__(self, value, _id=None):
    super().__init__()
    self.value = value
    
    if _id is None:
      self.id = uuid.uuid4()
    
    self.id = _id
  
  def added_to(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, BaseFunction):
      return Number(self.value + other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def subbed_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, BaseFunction):
      return Number(self.value - other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def multed_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, BaseFunction):
      return Number(self.value * other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def dived_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, BaseFunction):
      if other.value == 0:
        return None, RTError(
          other.pos_start, other.pos_end,
          'Division by zero',
          self.context
        )

      return Number(self.value / other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def powed_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, BaseFunction):
      return Number(self.value ** other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def percent_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, BaseFunction):
      return Number(self.value % other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def get_comparison_eq(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return Number(self.value == other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def get_comparison_ne(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return Number(self.value != other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def get_comparison_lt(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return Number(self.value < other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def get_comparison_gt(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return Number(self.value > other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def get_comparison_lte(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return Number(self.value <= other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def get_comparison_gte(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return Number(self.value >= other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def xored(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes)or isinstance(other, BaseFunction):
      return Number(self.value ^ other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def left_shifted(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, BaseFunction):
      return Number(self.value << other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def right_shifted(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, BaseFunction):
      return Number(self.value >> other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def bitwise_and(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, BaseFunction):
      return Number(self.value & other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def bitwise_or(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, BaseFunction):
      return Number(self.value | other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def bitwise_not(self):
    return Number(~self.value).set_context(self.context), None

  
  def anded_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return Number(self.value and other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def ored_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return Number(self.value or other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def notted(self):
    return Number(1 if self.value == 0 else 0).set_context(self.context), None

  def copy(self):
    copy = Number(self.value, self.id)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def is_true(self):
    return self.value != 0

  def __typeof__(self):
    if isinstance(self.value, int):
      return f"<Number 'int' {self.id}>"
    else:
      return f"<Number 'float' {self.id}>"

  def __str__(self):
    return str(self.value)

  def __repr__(self):
    return str(self.value)

class Object(Value):
  def __init__(self, value, _id=None):
    super().__init__()
    self.value = value
    
    if _id is None:
      self.id = uuid.uuid4()
    
    self.id = _id

  def added_to(self, other):
    return Object(self.value + other.value).set_context(self.context), None

  
  def subbed_by(self, other):
    return Object(self.value - other.value).set_context(self.context), None

  
  def multed_by(self, other):
    return Object(self.value * other.value).set_context(self.context), None

  
  def dived_by(self, other):
    return Object(self.value / other.value).set_context(self.context), None

  
  def powed_by(self, other):
    return Object(self.value ** other.value).set_context(self.context), None

  
  def percent_by(self, other):
    return Object(self.value % other.value).set_context(self.context), None

  
  def get_comparison_eq(self, other):
    return Object(self.value == other.value).set_context(self.context), None

  
  def get_comparison_ne(self, other):
    return Object(self.value != other.value).set_context(self.context), None

  
  def get_comparison_lt(self, other):
    return Object(self.value < other.value).set_context(self.context), None

  
  def get_comparison_gt(self, other):
    return Object(self.value > other.value).set_context(self.context), None

  
  def get_comparison_lte(self, other):
    return Object(self.value <= other.value).set_context(self.context), None
  
  def get_comparison_gte(self, other):
    return Object(self.value >= other.value).set_context(self.context), None

  
  def xored(self, other):
    return Object(self.value ^ other.value).set_context(self.context), None
  
  def left_shifted(self, other):
    return Object(self.value << other.value).set_context(self.context), None

  
  def right_shifted(self, other):
    return Object(self.value >> other.value).set_context(self.context), None

  
  def bitwise_and(self, other):
    return Object(self.value & other.value).set_context(self.context), None

  
  def bitwise_or(self, other):
    return Object(self.value | other.value).set_context(self.context), None
    
  def bitwise_not(self):
    return Object(~self.value).set_context(self.context), None
  
  def anded_by(self, other):
    return Object(self.value and other.value).set_context(self.context), None

  
  def ored_by(self, other):
    return Object(self.value or other.value).set_context(self.context), None
  
  def get_dot(self, verb):
        return Object(eval(f'self.value.{verb}')).set_context(self.context), None

    
  def set_dot(self, verb, obj):
      self.value = eval(f'self.value.{verb} = {repr(obj)}')
      
      return None, None

  def execute(self, args):
    res = RTResult()
    
    n_args = [repr(arg.value) if isinstance(arg, Object) else repr(arg) for arg in args]
    
    imps = ""
    
    for arg in args:
      if isinstance(arg, Object):
        imps += f'from {origin_module(arg.value)} import *\n'
    
    exec(imps)
    
    val = f'''self.value({", ".join(n_args)})'''
    
    return res.success(Object(eval(val)))
  
  def copy(self):
    copy = self
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)

    return copy

  def __typeof__(self):
    return f"<Object '{type(self.value)}' {self.id}>"

  def __str__(self):
    return str(self.value)

  def __repr__(self):
    return f'<Object: {repr(self.value)}>'

class Null(Value):
  def __init__(self, value):
    super().__init__()
    self.value = value
  
  def __repr__(self):
    return 'null'

  def __str__(self):
    return 'null'
    
  def is_true(self):
    return 0
  
  def copy(self):
    copy = self.value
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

class false(Value):
  def __init__(self, value):
    super().__init__()
    self.value = value
  
  def __repr__(self):
    return 'false'

  def __str__(self):
    return 'false'

  def is_true(self):
    return False
  
  def copy(self):
    copy = self.value
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy
  
class true(Value):
  def __init__(self, value):
    super().__init__()
    self.value = value
  
  def __repr__(self):
    return 'true'

  def __str__(self):
    return 'true'
    
  def is_true(self):
    return True
  
  def copy(self):
    copy = self.value
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

Number.null = Null(Number(0))
Number.false = false(Number(0))
Number.true = true(Number(1))
Number.math_PI = Number(math.pi)

class Bin(Value):
  def __init__(self, value, _id=None):
    super().__init__()
    self.value = value
    
    if _id is None:
      self.id = uuid.uuid4()
    
    self.id = _id
  
  def added_to(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes):
      return Bin(self.value + other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def subbed_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes):
      return Bin(self.value - other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def multed_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes):
      return Bin(self.value * other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def dived_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes):
      if other.value == 0:
        return None, RTError(
          other.pos_start, other.pos_end,
          'Division by zero',
          self.context
        )

      return Bin(self.value / other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def powed_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes):
      return Bin(self.value ** other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def percent_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes):
      return Bin(self.value % other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def get_comparison_eq(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bin(self.value == other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def get_comparison_ne(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bin(self.value != other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def get_comparison_lt(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bin(self.value < other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def get_comparison_gt(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bin(self.value > other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def get_comparison_lte(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bin(self.value <= other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def get_comparison_gte(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bin(self.value >= other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def xored(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return Bin(self.value ^ other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def left_shiffed(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return Bin(self.value << other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def right_shiffed(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return Bin(self.value >> other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def bitwise_and(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return Bin(self.value & other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def bitwise_or(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return Bin(self.value | other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def bitwise_not(self):
    return Bin(~self.value).set_context(self.context), None

  
  def anded_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bin(self.value and other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def ored_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bin(self.value or other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def notted(self):
    return Bin(bin(1) if self.value == bin(0) else bin(0)).set_context(self.context), None

  def copy(self):
    copy = Bin(self.value, self.id)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def is_true(self):
    return self.value != 0

  def __typeof__(self):
    return f"<Bin '{self.value}'> {self.id}"

  def __str__(self):
    return str(self.value)
  
  def __repr__(self):
    return str(self.value)
  
class Bytes(Value):
  def __init__(self, value, _id=None):
    super().__init__()
    self.value = value
    
    if _id is None:
      self.id = uuid.uuid4()
    
    self.id = _id
  
  def added_to(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes):
      return Bytes(self.value + other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def subbed_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes):
      return Bytes(self.value - other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def multed_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes):
      return Bytes(self.value * other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def dived_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes):
      if other.value == 0:
        return None, RTError(
          other.pos_start, other.pos_end,
          'Division by zero',
          self.context
        )

      return Bytes(self.value / other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def powed_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes):
      return Bytes(self.value ** other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def percent_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes):
      return Bytes(self.value % other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def get_comparison_eq(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bytes(to_bytes(self.value == other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def get_comparison_ne(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bytes(to_bytes(self.value != other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def get_comparison_lt(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bytes(to_bytes(self.value < other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def get_comparison_gt(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bytes(to_bytes(self.value > other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def get_comparison_lte(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bytes(to_bytes(self.value <= other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def get_comparison_gte(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bytes(to_bytes(self.value >= other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def xored(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return Bytes(to_bytes(self.value ^ other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def left_shiffed(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return Bytes(to_bytes(self.value << other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def right_shiffed(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return Bytes(to_bytes(self.value >> other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def bitwise_and(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return Bytes(self.value & other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def bitwise_or(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return Bytes(self.value | other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def bitwise_not(self):
    return Bytes(~self.value).set_context(self.context), None

  
  def anded_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bytes(to_bytes(self.value and other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def ored_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bytes(to_bytes(self.value or other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def notted(self):
    return Bytes(b'1' if self.value == b'0' else b'0').set_context(self.context), None

  def copy(self):
    copy = Bytes(self.value, self.id)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def is_true(self):
    return self.value != 0

  def __typeof__(self):
    return f"<Bytes '{self.value}' {self.id}>"

  def __str__(self):
    return str(to_bytes(self.value))
  
  def __repr__(self):
    return str(to_bytes(self.value))

class String(Value):
  def __init__(self, value, _id=None):
    super().__init__()
    self.value = value
    
    if _id is None:
      self.id = uuid.uuid4()
    
    self.id = _id
  
  def added_to(self, other):
    if isinstance(other, String):
      return String(self.value + other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def multed_by(self, other):
    if isinstance(other, Number):
      return String(self.value * other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def dived_by(self, other):
    if isinstance(other, Number):
      return String(self.value[other.value - 1]).set_context(self.context), None
    elif isinstance(other, List):
      strb = "String(self.value["

      lval = other.value

      if len(lval) > 3 or len(lval) < 3:
        return None, Value.illegal_operation(self, other)

      for x in lval:
        if not isinstance(x, Null) and isinstance(x, Number):
          strb += f'{x.value - 1}:'
        else:
          strb += ':'
      
      strb = strb[:-1] + ']).set_context(self.context), None'
      
      return eval(strb)

    else:
      return None, Value.illegal_operation(self, other)
  
  def percent_by(self, other):
    if isinstance(other, Number):
      return String(self.value % other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def gen(self):
    for char in self.value:
      yield RTResult().success(String(char))

  
  def get_index(self, index):
    if isinstance(index, Number):
      return String(self.value[index.value - 1]).set_context(self.context), None
    elif isinstance(index, List):
      strb = "String(self.value["

      lval = index.value

      if len(lval) > 3 or len(lval) < 3:
        return None, Value.illegal_operation(self, index)

      for x in lval:
        #print(type(x))
        if not isinstance(x, Null) and isinstance(x, Number):
          strb += f'{x.value - 1}:'
        else:
          strb += ':'
      
      strb = strb[:-1] + ']).set_context(self.context), None'
      
      return eval(strb)

    else:
      return None, Value.illegal_operation(self, index)

  
  def get_comparison_eq(self, other):
    if not (isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction)):
      return None, self.illegal_operation(other)
    return Number(int(self.value == other.value)), None

  
  def get_comparison_ne(self, other):
    if not (isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction)):
      return None, self.illegal_operation(other)
    return Number(int(self.value != other.value)), None

  
  def xored(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return String(self.value ^ other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def left_shiffed(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return String(self.value << other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def right_shiffed(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return String(self.value >> other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def bitwise_and(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return String(self.value & other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  
  def bitwise_or(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String) or isinstance(other, BaseFunction):
      return String(self.value | other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def bitwise_not(self):
    return String(~self.value).set_context(self.context), None

  def is_true(self):
    return len(self.value) > 0

  def copy(self):
    copy = String(self.value, self.id)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def __typeof__(self):
    return f"<String {repr(self)} {self.id}>"

  def __str__(self):
    return self.value

  def __repr__(self):
    return f'"{self.value}"'

class List(Value):
  def __init__(self, elements, _id=None):
    super().__init__()
    self.elements = elements
    self.value = elements
    
    if _id is None:
      self.id = uuid.uuid4()
    
    self.id = _id
  
  def added_to(self, other):
    new_list = self.copy()
    new_list.elements.append(other)
    return new_list, None

  
  def subbed_by(self, other):
    if isinstance(other, Number):
      new_list = self.copy()
      try:
        new_list.elements.pop(other.value)
        return new_list, None
      except:
        return None, RTError(
          other.pos_start, other.pos_end,
          'Element at this index could not be removed from list because index is out of bounds',
          self.context
        )
    else:
      return None, Value.illegal_operation(self, other)

  
  def multed_by(self, other):
    if isinstance(other, List):
      new_list = self.copy()
      new_list.elements.extend(other.elements)
      return new_list, None
    else:
      return None, Value.illegal_operation(self, other)

  def dived_by(self, other):
    if isinstance(other, Number):
      return self.elements[other.value - 1], None
    elif isinstance(other, List):
      strb = "List(self.elements["

      lval = other.value

      if len(lval) > 3 or len(lval) < 3:
        return None, Value.illegal_operation(self, other)

      for x in lval:
        if not isinstance(x, Null) and isinstance(x, Number):
          strb += f'{x.value - 1}:'
        else:
          strb += ':'
      
      strb = strb[:-1] + ']), None'
      
      return eval(strb)

    else:
      return None, Value.illegal_operation(self, other)

  def gen(self):
    for elt in self.elements:
      yield RTResult().success(elt)

  
  def get_index(self, index):
    try:
      if isinstance(index, Number):
        return self.elements[index.value - 1], None
      elif isinstance(index, List):
        strb = "List(self.elements["

        lval = index.value

        if len(lval) > 3 or len(lval) < 3:
          return None, Value.illegal_operation(self, index)

        for x in lval:
          if not isinstance(x, Null) and isinstance(x, Number):
            strb += f'{x.value - 1}:'
          else:
            strb += ':'
      
        strb = strb[:-1] + ']), None'
      
        return eval(strb)
    except IndexError:
      return None, RTError(
        index.pos_start, index.pos_end,
        f"Cannot retrieve element {index} from list {self!r} because it is out of bounds.",
        self.context
      )

  
  def set_index(self, index, value):
    if not isinstance(index, Number):
      return None, self.illegal_operation(index)
    try:
      self.elements[index.value - 1] = value
    except IndexError:
      return None, RTError(
        index.pos_start, index.pos_end,
        f"Cannot set element {index} from list {self!r} to {value!r} because it is out of bounds.",
        self.context
      )
    
    return self, None

  def copy(self):
    copy = List(self.elements, self.id)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def __typeof__(self):
    return f"<List {repr(self)} {self.id}>"

  def __str__(self):
    return "[" + ", ".join([repr(x) for x in self.elements]) + "]"

  def __repr__(self):
    return "[" + ", ".join([repr(x) for x in self.elements]) + "]"

class BaseFunction(Value):
  def __init__(self, name):
    super().__init__()
    self.name = name or "<anonymous>"

  def set_context(self, context=None):
    if hasattr(self, "context") and self.context: return self
    return super().set_context(context)

  def generate_new_context(self):
    new_context = Context(self.name, self.context, self.pos_start)

    if new_context.parent:
      new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)
    else:
      new_context.symbol_table = SymbolTable()
    return new_context

  def check_args(self, arg_names, args, defaults):
    res = RTResult()

    if len(args) > len(arg_names):
      return res.failure(RTError(
        self.pos_start, self.pos_end,
        f"{len(args) - len(arg_names)} too many args passed into {self}",
        self.context
      ))
    
    if len(args) < len(arg_names) - len(list(filter(lambda default: default is not None, defaults))):
      return res.failure(RTError(
        self.pos_start, self.pos_end,
        f"{(len(arg_names) - len(list(filter(lambda default: default is not None, defaults)))) - len(args)} too few args passed into {self}",
        self.context
      ))

    return res.success(None)

  def populate_args(self, arg_names, args, defaults, dynamics, exec_ctx):
    res = RTResult()
    for i in range(len(arg_names)):
      arg_name = arg_names[i]

      try:
        dynamic = dynamics[i]
      except:
        dynamic = None

      arg_value = defaults[i] if i >= len(args) else args[i]
      if dynamic is not None:
        dynamic_context = Context(f"{self.name} (dynamic argument '{arg_name}')", exec_ctx, dynamic.pos_start.copy())
        dynamic_context.symbol_table = SymbolTable(exec_ctx.symbol_table)
        dynamic_context.symbol_table.set("$", arg_value)
        arg_value = res.register(Interpreter().visit(dynamic, dynamic_context))
        if res.should_return(): return res
      arg_value.set_context(exec_ctx)
      exec_ctx.symbol_table.set(arg_name, arg_value)
    return res.success(None)

  def check_and_populate_args(self, arg_names, args, defaults, dynamics, exec_ctx):
    res = RTResult()
    res.register(self.check_args(arg_names, args, defaults))
    if res.should_return(): return res
    res.register(self.populate_args(arg_names, args, defaults, dynamics, exec_ctx))
    if res.should_return(): return res
    return res.success(None)

class Function(BaseFunction):
  def __init__(self, name, body_node, arg_names, defaults, dynamics, should_auto_return, _id=None):
    super().__init__(name)
    self.body_node = body_node
    self.arg_names = arg_names
    self.defaults = defaults
    self.dynamics = dynamics
    self.should_auto_return = should_auto_return
    
    if _id is None:
      self.id = uuid.uuid4()
    
    self.id = _id

  
  def execute(self, args):
    res = RTResult()
    interpreter = Interpreter()
    exec_ctx = self.generate_new_context()

    res.register(self.check_and_populate_args(self.arg_names, args, self.defaults, self.dynamics, exec_ctx))
    if res.should_return(): return res

    value = res.register(interpreter.visit(self.body_node, exec_ctx))
    if res.should_return() and res.func_return_value == None: return res

    ret_value = (value if self.should_auto_return else None) or res.func_return_value or Number.null
    return res.success(ret_value)

  def copy(self):
    copy = Function(self.name, self.body_node, self.arg_names, self.defaults, self.dynamics, self.should_auto_return, self.id)
    copy.set_context(self.context)
    copy.set_pos(self.pos_start, self.pos_end)
    return copy

  def __typeof__(self):
    return f"<Function {self.name} {self.id}>"

  def __repr__(self):
    return f"<function {self.name}>"

class BuiltInFunction(BaseFunction):
  def __init__(self, name, _id=None):
    super().__init__(name)
    
    if _id is None:
      self.id = uuid.uuid4()
    
    self.id = _id

  
  def execute(self, args):
    res = RTResult()
    exec_ctx = self.generate_new_context()

    method_name = f'execute_{self.name}'
    method = getattr(self, method_name, self.no_execute_method)

    res.register(self.check_and_populate_args(method.arg_names, args, method.defaults, method.dynamics, exec_ctx))
    if res.should_return(): return res

    return_value = res.register(method(exec_ctx))
    if res.should_return(): return res
    return res.success(return_value)

  def no_execute_method(self, node, context):
    raise Exception(f'No execute_{self.name} method defined')

  def copy(self):
    copy = BuiltInFunction(self.name, self.id)
    copy.set_context(self.context)
    copy.set_pos(self.pos_start, self.pos_end)
    return copy

  def __typeof__(self):
    return f"<Built-in Function {self.name} {self.id}>"

  def __repr__(self):
    return f"<built-in function {self.name}>"

  #####################################

  # Decorator for built-in functions
  @staticmethod
  def args(arg_names, defaults=None, dynamics=None):
    if defaults is None:
      defaults = [None] * len(arg_names)
    if dynamics is None:
      dynamics = [None] * len(arg_names)
    def _args(f):
      f.arg_names = arg_names
      f.defaults = defaults
      f.dynamics = dynamics
      return f
    return _args

  #####################################

  
  @args(['value'])
  def execute_print(self, exec_ctx):
    print(str(exec_ctx.symbol_table.get('value')), end="")
    return RTResult().success(Number.null)

  @args(['start', 'stop', 'step'])
  def execute_range(self, exec_ctx):
    start = exec_ctx.symbol_table.get('start')
    stop = exec_ctx.symbol_table.get('stop')
    step = exec_ctx.symbol_table.get('step')

    if not any(isinstance(clss, Number) for clss in (start, stop, step)) and not any(isinstance(clss, int) for clss in (start.value, stop.value, step.value)):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "The all arguments needs be Number Int",
        exec_ctx
      ))

    return RTResult().success(List(range(start.value, stop.value, step.value)))

  
  @args(['R', 'G', "B"])
  def execute_foregroundColor(self, exec_ctx):
    R = int(str(exec_ctx.symbol_table.get('R')))
    G = int(str(exec_ctx.symbol_table.get('G')))
    B = int(str(exec_ctx.symbol_table.get('B')))

    print(f'\033[38;2;{R};{G};{B}m', end="")

    return RTResult().success(Number.null)

  
  @args(['R', 'G', "B"])
  def execute_backgroundColor(self, exec_ctx):
    R = int(str(exec_ctx.symbol_table.get('R')))
    G = int(str(exec_ctx.symbol_table.get('G')))
    B = int(str(exec_ctx.symbol_table.get('B')))

    print(f'\033[48;2;{R};{G};{B}m', end="")

    return RTResult().success(Number.null)

  
  @args([])
  def execute_resetColor(self, exec_ctx):
    print(f'\033[0m', end="")

    return RTResult().success(Number.null)

  @args(["value"])
  def execute_reverse(self, exec_ctx):
    value = exec_ctx.symbol_table.get('value')

    if not isinstance(value, (String, List, Bin, Bytes)):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "The value needs be String, List, Bin or Bytes not " + {str(type(value))},
        exec_ctx
      ))
    
    if isinstance(value, String):
      return RTResult().success(String(value.value[::-1]))
    elif isinstance(value, List):
      return RTResult().success(List(value.value[::-1]))
    elif isinstance(value, Bin):
      return RTResult().success(Bin(value.value[::-1]))
    elif isinstance(value, Bytes):
      return RTResult().success(Bytes(value.value[::-1]))
  
  @args(['value'])
  def execute_println(self, exec_ctx):
    print(str(exec_ctx.symbol_table.get('value')))
    return RTResult().success(Number.null)

  
  @args(['value'])
  def execute_error(self, exec_ctx):
    print(str(exec_ctx.symbol_table.get('value')))

    sys.exit(1)

    return RTResult().success(Number.null)

  
  @args(['value'])
  def execute_id(self, exec_ctx):
    val = exec_ctx.symbol_table.get('value')

    if isinstance(val, String):
      return RTResult().success(Number(id(str(val))))
    elif isinstance(val, Number):
      return RTResult().success(Number(id(int(str(val)))))
    elif isinstance(val, Bin):
      return RTResult().success(Number(convert_forL(val)))
    elif isinstance(val, Bytes):
      return RTResult().success(Number(id(to_bytes_forL(val))))
    elif isinstance(val, BaseFunction):
      return RTResult().success(Number(id(val)))
    else:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "The index needs be Number, String, Bytes or Bin type, not '" + str(type(val)) + "'",
        exec_ctx
      ))

  
  @args(['value', 'bit'], [Number.null, Number.null])
  def execute_Int(self, exec_ctx):
    val = exec_ctx.symbol_table.get('value')
    bit = exec_ctx.symbol_table.get('bit')

    try:
      if not isinstance(bit, (Null, true, false)):
        return RTResult().success(Number(int(val.value, int(bit.value))))
      else:
        return RTResult().success(Number(int(val.value)))
    except Exception as e:
      print(e)
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "The value or bit needs be Integer, not '" + str(type(val)) + "'",
        exec_ctx
      ))

  @args(['value'])
  def execute_Float(self, exec_ctx):
    val = exec_ctx.symbol_table.get('value')
    
    try:
      return RTResult().success(Number(float(val.value)))
    except Exception as e:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "The value or bit needs be Float, not '" + str(type(val)) + "'",
        exec_ctx
      ))

  @args(['value'])
  def execute_hex(self, exec_ctx):
    val = exec_ctx.symbol_table.get('value')

    if isinstance(val, (Bin, Bytes)):
      val = Number(int.from_bytes(val.value, byteorder="little", signed=True))
    else:
      val = Number(val.value)

    try:
      return RTResult().success(Bin(hex(val.value)))
    except:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "The value or bit needs be Number or Bytes, not '" + str(type(val)) + "'",
        exec_ctx
      ))
  
  @args(['value', 'bit'], [None, Number.null])
  def execute_Char(self, exec_ctx):
    val = exec_ctx.symbol_table.get('value')
    bit = exec_ctx.symbol_table.get('bit')

    try:
      return RTResult().success(String(str(val.value if len(val.value) == 1 else None, int(bit.value) if not isinstance(bit, (Null, true, false)) else 0)))
    except:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "The value or bit needs be Number, Char or Bytes, not '" + str(type(val)) + "'",
        exec_ctx
      ))
  
  @args(['value'])
  def execute_List(self, exec_ctx):
    val = exec_ctx.symbol_table.get('value')

    try:
      val = list(val.value)
      nval = []

      for el in val:
        nval.append(convert_types_to_values(el))

      return RTResult().success(List(nval))
    except:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "The value needs be List, String, Char, not '" + str(type(val)) + "'",
        exec_ctx
      ))

  @args(['value', 'bit'], [None, Number.null])
  def execute_Str(self, exec_ctx):
    val = exec_ctx.symbol_table.get('value')
    bit = exec_ctx.symbol_table.get('bit')

    try:
      if isinstance(bit, Null):
        return RTResult().success(String(str(val)))
      else:
        return RTResult().success(String(str(val, bit.value)))
    except:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "The value or bit needs be Number, String Bin or Bytes, not '" + str(type(val)) + "'",
        exec_ctx
      ))

  @args(['value', 'bit'], [None, Number.null])
  def execute_Bin(self, exec_ctx):
    val = exec_ctx.symbol_table.get('value')
    bit = exec_ctx.symbol_table.get('bit')

    if isinstance(val, (Bin, Bytes)):
      val = Number(int.from_bytes(val.value, byteorder="little", signed=True))
    else:
      val = Number(val.value)

    try:
      return RTResult().success(Bin(bin(val.value, int(bit.value) if not isinstance(bit, (Null, true, false)) else 0)))
    except:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "The value or bit needs be Number or Bytes, not '" + str(type(val)) + "'",
        exec_ctx
      ))

  
  @args(['value', 'index'])
  def execute_split(self, exec_ctx):
    value = exec_ctx.symbol_table.get('value')
    index = str(exec_ctx.symbol_table.get('index'))
    if isinstance(value, String):
      value = str(value)
    else:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "The index needs be String type, not " + str(type(value)),
        exec_ctx
      ))

    temp = value.split(index)
    c = []
    for x in temp:
      c.append(String(x))
      
    return RTResult().success(List(c))

  
  @args(['value'])
  def execute_system(self, exec_ctx):
    os.system(str(exec_ctx.symbol_table.get('value')))
    return RTResult().success(Number.null)

  
  @args(['value'])
  def execute_print_ret(self, exec_ctx):
    return RTResult().success(String(str(exec_ctx.symbol_table.get('value'))))

  
  @args(['value'], [Number.null])
  def execute_input(self, exec_ctx):
    text = input("" if str(exec_ctx.symbol_table.get('value')) == '0' else str(exec_ctx.symbol_table.get('value')))
    return RTResult().success(String(text))

  
  @args(['value'], [Number.null])
  def execute_input_key(self, exec_ctx):
    text = basBR.input_char("" if str(exec_ctx.symbol_table.get('value')) == '0' else str(exec_ctx.symbol_table.get('value')))

    if not isinstance(text, bytes):
      return RTResult().success(String(text))
    else:
      return RTResult().success(Bytes(text))

  @args(['obj'])
  def execute_typeof(self, exec_ctx):
    obj = exec_ctx.symbol_table.get("obj")
    
    return RTResult().success(String(obj.__typeof__()))
  
  @args(['obj'])
  def execute_repr(self, exec_ctx):
    obj = exec_ctx.symbol_table.get("obj")
    
    return RTResult().success(String(repr(obj)))
    
  @args(['value'], [Number.null])
  def execute_input_int(self, exec_ctx):
    while True:
      text = input("" if str(exec_ctx.symbol_table.get('value')) == '0' else str(exec_ctx.symbol_table.get('value')))
      try:
        number = int(text)
        break
      except ValueError:
        print(f"'{text}' must be an integer. Try again!")

    return RTResult().success(Number(number))
  
  @args(['lang', 'code', 'funcn', 'args', "pkg", 'include', "lib"], [Number.null, Number.null, Number.null, Number.null, Number.null, Number.null, Number.null])
  def execute_extern(self, exec_ctx):
    lang = exec_ctx.symbol_table.get('lang')
    code = exec_ctx.symbol_table.get('code')
    funcn = exec_ctx.symbol_table.get('funcn')
    args = exec_ctx.symbol_table.get('args')
    pkg = exec_ctx.symbol_table.get('pkg')
    include = exec_ctx.symbol_table.get('include')
    lib = exec_ctx.symbol_table.get('lib')

    if not isinstance(lang, (String, Null)):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        f"The lang needs be String not {str(type(lib))}",
        exec_ctx
      ))
    
    if not isinstance(code, (String, Null)):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        f"The code needs be String not {str(type(lib))}",
        exec_ctx
      ))
    
    if not isinstance(funcn, (String, Null)):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        f"The funcn needs be String not {str(type(lib))}",
        exec_ctx
      ))
    
    if not isinstance(args, (List, Null)):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        f"The args needs be List not {str(type(lib))}",
        exec_ctx
      ))
    
    if not isinstance(pkg, (List, Null)):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        f"The pkg needs be List not {str(type(lib))}",
        exec_ctx
      ))

    if not isinstance(include, (List, Null)):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        f"The include needs be List not {str(type(lib))}",
        exec_ctx
      ))

    obj = None

    if lang.value == "Python":
      namespace = {}
      
      exec(code.value, namespace)

      if isinstance(funcn, Null):
        return RTResult().success(Number.null)
      else:
        if isinstance(args, Null):
          obj = eval(f'{funcn.value}()')
        else:
          n_args = br_args_to_python_for_extern(args.value)

          obj = eval(f'namespace["{funcn.value}"]({", ".join(n_args)})')
          
    elif lang.value == "Cpp":
      import cppyy

      cppyy.cppdef(code.value)

      if isinstance(include, List):
        for inc in include.value:
          cppyy.include(inc.value)
      elif isinstance(include, Null):
        pass
      else:
        return RTResult().failure(RTError(
          self.pos_start, self.pos_end,
          f"The include needs be List not {str(type(lib))}",
          exec_ctx
        ))
      
      if isinstance(lib, List):
        for lb in lib.value:
          cppyy.load_library(lb.value)
      elif isinstance(lib, Null):
        pass
      else:
        return RTResult().failure(RTError(
          self.pos_start, self.pos_end,
          f"The lib needs be List not {str(type(lib))}",
          exec_ctx
        ))

      if isinstance(funcn, Null):
        obj = cppyy.gbl.main()
      else:
        if isinstance(args, Null):
          obj = eval(f'cppyy.gbl.{funcn.value}()')
        else:
          n_args = br_args_to_python_for_extern(args.value)

          obj = eval(f'cppyy.gbl.{funcn.value}({", ".join(n_args)})')
    elif lang.value == "Javascript":
      import quickjs

      context = quickjs.Context()

      context.eval(code.value)

      if not isinstance(funcn, Null):
        if isinstance(args, Null):
          obj = eval(f'context.call("{funcn.value}")')
        else:
          n_args = []

          n_args = br_args_to_python_for_extern(args.value)

          obj = eval(f'context.call("{funcn.value}", {", ".join(n_args)})')
    elif lang.value == "Asm":
      from pyinlineasm import inline_asm as __asm__
      
      obj = __asm__(code.value)
    elif lang.value == "Lua":
      from lupa import LuaRuntime
        
      lua = LuaRuntime()

      lua.execute(code.value)

      if isinstance(funcn, Null):
        return RTResult().success(Number.null)
      else:
        if isinstance(args, Null):
          obj = eval(f'lua.eval("{funcn.value}")()')
        else:
          n_args = br_args_to_python_for_extern(args.value)

          obj = eval(f'lua.eval("{funcn.value}")({", ".join(n_args)})')
    elif lang.value == "Julia":
      pass
    else:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        f"Extern language not identified '{str(lang)}'",
        exec_ctx
      ))

    return RTResult().success(Object(obj))

  
  @args(['value'], [Number.null])
  def execute_exit(self, exec_ctx):
    sys.exit(0 if str(exec_ctx.symbol_table.get('value')) == '0' else int(str(exec_ctx.symbol_table.get('value'))))

    return RTResult().success(Number.null)

  @args(["value"])
  def execute_pointer(self, exec_ctx):
    global CALLBACK, _registry

    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, String):
      buf = ctypes.create_string_buffer(value.value.encode("utf-8"))
      _keep_alive(buf)

      return RTResult().success(Object(buf))
    elif isinstance(value, Number):
      if float(value.value).is_integer():
        cnum = ctypes.c_int(int(value.value))
      else:
        cnum = ctypes.c_double(float(value.value))

      _keep_alive(cnum)

      return RTResult().success(Object(cnum))
    elif isinstance(value, Bytes):
      buf = ctypes.create_string_buffer(value.value, len(value.value))
      _keep_alive(buf)

      return RTResult().success(Object(buf))
    elif isinstance(value, Bin):
      bits = value.value

      if len(bits) % 8 != 0:
        return RTResult().failure(RTError(self.pos_start, self.pos_end, "Bin length must be a multiple of 8", exec_ctx))
      
      raw = int(bits, 2).to_bytes(len(bits) // 8, "big")
      buf = ctypes.create_string_buffer(raw, len(raw))
      _keep_alive(buf)

      return RTResult().success(Object(buf))
    elif isinstance(value, (BaseFunction, StructInstance, ClassInstance)):
      def pfunc():
        obj = value
        addr = id(obj)
        _registry[addr] = obj
        return addr
      
      cf = CALLBACK(pfunc)
      _keep_alive(cf)

      _registry[id(cf)] = value
      return RTResult().success(Object(cf))
    else:
      return RTResult().failure(RTError(self.pos_start, self.pos_end, "Invalid value for pointer(): expected Number, String, Bytes, Bin or Function", exec_ctx))

  @args(["ptr"])
  def execute_address(self, exec_ctx):
    global CALLBACK, _registry

    ptr = exec_ctx.symbol_table.get("ptr")

    try:
      if isinstance(ptr.value, ctypes._CFuncPtr):
        addr = ctypes.cast(ptr.value, ctypes.c_void_p).value
        _registry[addr] = ptr.value

        return RTResult().success(Number(addr))
      elif hasattr(ptr.value, "value") and hasattr(ptr.value, "_type_"):
        return RTResult().success(Number(ctypes.addressof(ptr.value)))
      else:
        raise Exception()
    except:
      if isinstance(ptr, String):
        buf = ctypes.create_string_buffer(ptr.value.encode("utf-8"))
        _keep_alive(buf)

        return RTResult().success(Number(ctypes.addressof(buf)))
      elif isinstance(ptr, Number):
        if float(ptr.value).is_integer():
          cnum = ctypes.c_int(int(ptr.value))
        else:
          cnum = ctypes.c_double(float(ptr.value))

        _keep_alive(cnum)

        return RTResult().success(Number(ctypes.addressof(cnum)))
      elif isinstance(ptr, Bytes):
        buf = ctypes.create_string_buffer(ptr.value, len(ptr.value))
        _keep_alive(buf)

        return RTResult().success(Number(ctypes.addressof(buf)))
      elif isinstance(ptr, Bin):
        cnum = ctypes.c_int(int(ptr.value, 2))
        _keep_alive(cnum)

        return RTResult().success(Number(ctypes.addressof(cnum)))
      elif isinstance(ptr, (BaseFunction, StructInstance, ClassInstance)):
        def pfunc():
          obj = ptr
          addr = id(obj)
          _registry[addr] = obj
          return addr
        
        cf = CALLBACK(pfunc)
        _keep_alive(cf)
        _registry[id(cf)] = ptr

        return RTResult().success(Number(ctypes.cast(cf, ctypes.c_void_p).value))
      else:
        return RTResult().failure(RTError(self.pos_start, self.pos_end, "Unsupported type for address()", exec_ctx))

  @args(["addr", "auto", "typ", "len"], [Number.null, Number.true, Number.null, Number(32)])
  def execute_deref(self, exec_ctx):
    global _registry, CALLBACK
    addr = exec_ctx.symbol_table.get("addr")
    auto = exec_ctx.symbol_table.get("auto")
    typ  = exec_ctx.symbol_table.get("typ")
    _len = exec_ctx.symbol_table.get("len")
    
    if not (isinstance(addr, Number) and (isinstance(typ, String) or typ == Number.null)):
      return RTResult().failure(RTError(self.pos_start, self.pos_end, f"The arguments for deref must be (Number, true/false, String or null), not '{type(addr)}, {type(typ)}'", exec_ctx))
    
    address = addr.value
    length = _len.value if isinstance(_len, Number) and _len.value is not None else 32

    try:
      type_str = typ.value.lower()
    except:
      type_str = None
    if auto.value and type_str is None:
      if address in _registry:
        obj = _registry[address]

        try:
          obj = obj()

          return RTResult().success(_registry[obj])
        except:
          return RTResult().success(obj)
      else:
        min_int, max_int = -2147483648, 2147483647

        str_ptr = None

        try:
          str_ptr = ctypes.string_at(address).decode('utf-8')
        except:
          pass

        float_ptr = ctypes.cast(address, ctypes.POINTER(ctypes.c_double)).contents.value
        int_ptr = ctypes.cast(address, ctypes.POINTER(ctypes.c_int)).contents.value

        float_c = is_normal_double(float_ptr)
        int_c = min_int <= int_ptr <= max_int
        str_c = (re.fullmatch(r"[ -~]+", str_ptr) if str_ptr is not None else False)

        if str_c:
          raw = ctypes.string_at(address)
        
          return RTResult().success(String(raw.decode("utf-8")))
        elif int_c and (not float_c):
          return RTResult().success(Number(int_ptr))
        elif float_c:
          return RTResult().success(Number(float_ptr))
        else:
          pass
        
    try:
      if type_str == 'string':
        raw = ctypes.string_at(address, length)
        s = raw.split(b'\x00', 1)[0]

        return RTResult().success(String(s.decode("utf-8")))
      elif type_str == 'bytes':
        raw = ctypes.string_at(address, length)

        return RTResult().success(Bytes(raw))
      elif type_str == 'bin':
        raw = ctypes.string_at(address, length)
        bits = ''.join(f'{b:08b}' for b in raw)

        return RTResult().success(Bin(bits))
      elif type_str == 'int':
        val = ctypes.cast(address, ctypes.POINTER(ctypes.c_int)).contents.value

        return RTResult().success(Number(val))
      elif type_str == 'float':
        val = ctypes.cast(address, ctypes.POINTER(ctypes.c_double)).contents.value

        return RTResult().success(Number(val))
      elif type_str == 'function':
        obj = _registry[address]

        try:
          obj = obj()

          return RTResult().success(_registry[obj])
        except:
          return RTResult().success(obj)
        
        return RTResult().failure(RTError(self.pos_start, self.pos_end, "Function not found in registry", exec_ctx))
      else:
        return RTResult().failure(RTError(self.pos_start, self.pos_end, f"Unknown type '{type_str}'", exec_ctx))
    except Exception as e:
      return RTResult().failure(RTError(self.pos_start, self.pos_end, f"Error while dereferencing as '{type_str}': {e}", exec_ctx))

  @args(["space", "typ"], [Number.null, Number.null])
  def execute_malloc(self, exec_ctx):
    space = exec_ctx.symbol_table.get("space")
    typ = exec_ctx.symbol_table.get("typ")

    if isinstance(typ, String) and typ.value == "function":
      size = 8
    elif isinstance(typ, String) and typ.value == "int":
      size = ctypes.sizeof(ctypes.c_int)
    elif isinstance(typ, String) and typ.value == "float":
      size = ctypes.sizeof(ctypes.c_double)
    elif isinstance(space, Number):
      size = space.value
    else:
      size = 8

    ptr = basBR.malloc(size)

    return RTResult().success(Number(ptr))

  @args(["ptr", "content", "length"])
  def execute_memmove(self, exec_ctx):
    ptr = exec_ctx.symbol_table.get("ptr")
    content = exec_ctx.symbol_table.get("content")
    length = exec_ctx.symbol_table.get("length")

    if not (hasattr(ptr, "value") and isinstance(ptr.value, int)):
      return RTResult().failure(RTError(self.pos_start, self.pos_end, "Invalid ptr", exec_ctx))
    
    dest = ptr.value

    if isinstance(content, String):
      data = content.value.encode('utf-8')
      _keep_alive(data)

      _exec = basBR.memmove(ctypes.c_void_p(dest), ctypes.c_char_p(data), length.value if isinstance(length, Number) else len(data))

      _registry[dest] = content

      return RTResult().success(Number(_exec))
    elif isinstance(content, Number):
      if float(content.value).is_integer():
        cnum = ctypes.c_int(int(content.value))
      else:
        cnum = ctypes.c_double(float(content.value))

      _keep_alive(cnum)

      _exec = basBR.memmove(ctypes.c_void_p(dest), ctypes.byref(cnum), ctypes.sizeof(cnum))

      _registry[dest] = content

      return RTResult().success(Number(_exec))
    elif isinstance(content, Bytes):
      data = content.value

      _keep_alive(data)
      
      _exec = basBR.memmove(ctypes.c_void_p(dest), ctypes.c_char_p(data), len(data) if isinstance(length, Number) and length.value else len(data))
      
      _registry[dest] = content

      return RTResult().success(Number(_exec))
    elif isinstance(content, Bin):
      raw = int(content.value, 2).to_bytes((len(content.value) + 7) // 8, 'big')

      _keep_alive(raw)

      _exec = basBR.memmove(ctypes.c_void_p(dest), ctypes.c_char_p(raw), len(raw))

      _registry[dest] = content

      return RTResult().success(Number(_exec))
    elif isinstance(content, (BaseFunction, StructInstance, ClassInstance)):
      def pfunc():
        obj = content
        addr = id(obj)
        _registry[addr] = obj
        return addr
      
      cf = CALLBACK(pfunc)

      _keep_alive(cf)

      cf_addr = ctypes.cast(cf, ctypes.c_void_p).value

      _registry[cf_addr] = content
      _registry[dest] = content

      _exec = basBR.memmove(ctypes.c_void_p(dest), ctypes.byref(ctypes.c_void_p(cf_addr)), ctypes.sizeof(ctypes.c_void_p))

      return RTResult().success(Number(_exec))
    else:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Invalid value for memmove(): " + str(type(content)),
        exec_ctx
      ))

  @args(["ptr"])
  def execute_free(self, exec_ctx):
    ptr = exec_ctx.symbol_table.get("ptr")
    basBR.free(ptr.value)

    return RTResult().success(Number.null)

  @args(["lpAddress", "dwSize", "flAllocationType", "flProtect"])
  def execute_virtualAlloc(self, exec_ctx):
    lpAddress = exec_ctx.symbol_table.get("lpAddress")
    dwSize = exec_ctx.symbol_table.get("dwSize")
    flAllocationType = exec_ctx.symbol_table.get("flAllocationType")
    flProtect = exec_ctx.symbol_table.get("flProtect")

    _exec = basBR.virtualAlloc(lpAddress, dwSize, flAllocationType, flProtect)

    addr = ctypes.cast(_exec, ctypes.c_void_p).value

    return RTResult().success(Number(addr))
  
  @args(["lpAddress", "dwSize", "dwFreeType"])
  def execute_virtualFree(self, exec_ctx):
    lpAddress = exec_ctx.symbol_table.get("lpAddress")
    dwSize = exec_ctx.symbol_table.get("dwSize")
    dwFreeType = exec_ctx.symbol_table.get("dwFreeType")

    basBR.virtualFree(lpAddress, dwSize, dwFreeType)

    return RTResult().success(Number.Null)

  @args([])
  def execute_clear(self, exec_ctx):
    os.system('cls' if os.name == 'nt' else 'clear') 
      
    return RTResult().success(Number.null)

  
  @args(["value"])
  def execute_is_number(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), Number)
    return RTResult().success(Number.true if is_number else Number.false)

  
  @args(["value"])
  def execute_is_string(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), String)
    return RTResult().success(Number.true if is_number else Number.false)

  
  @args(["value"])
  def execute_is_list(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), List)
    return RTResult().success(Number.true if is_number else Number.false)

  
  @args(["value"])
  def execute_is_function(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), BaseFunction)
    return RTResult().success(Number.true if is_number else Number.false)

  @args(["value", "encoding"], [Number.null, String('utf-8')])
  def execute_encode(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Null):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
          f"The value needs be String not {str(type(value))}",
          exec_ctx
      ))

    encoding = exec_ctx.symbol_table.get("encoding")

    return RTResult().success(Bytes(value.value.encode(encoding.value)))

  @args(["value", "decoding"], [Number.null, String('utf-8')])
  def execute_decode(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Null) or not isinstance(value, Bytes):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
          f"The value needs be Bytes not {str(type(value))}",
          exec_ctx
      ))

    decoding = exec_ctx.symbol_table.get("decoding")

    return RTResult().success(String(value.value.decode(decoding.value)))
  
  @args(["value"])
  def execute_chr(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if not isinstance(value, Number) and not isinstance(value.value, int):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
          f"The value needs be String not {str(type(value))}",
          exec_ctx
      ))

    return RTResult().success(String(str(chr(value.value))))
  
  @args(["list", "value"])
  def execute_append(self, exec_ctx):
    list_ = exec_ctx.symbol_table.get("list")
    value = exec_ctx.symbol_table.get("value")

    if not isinstance(list_, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "First argument must be list",
        exec_ctx
      ))

    list_.elements.append(value)
    return RTResult().success(Number.null)

  
  @args(["list", "index"])
  def execute_pop(self, exec_ctx):
    list_ = exec_ctx.symbol_table.get("list")
    index = exec_ctx.symbol_table.get("index")

    if not isinstance(list_, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "First argument must be list",
        exec_ctx
      ))

    if not isinstance(index, Number):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Second argument must be number",
        exec_ctx
      ))

    try:
      element = list_.elements.pop(index.value)
    except:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        'Element at this index could not be removed from list because index is out of bounds',
        exec_ctx
      ))
    return RTResult().success(element)

  
  @args(["listA", "listB"])
  def execute_extend(self, exec_ctx):
    listA = exec_ctx.symbol_table.get("listA")
    listB = exec_ctx.symbol_table.get("listB")

    if not isinstance(listA, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "First argument must be list",
        exec_ctx
      ))

    if not isinstance(listB, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Second argument must be list",
        exec_ctx
      ))

    listA.elements.extend(listB.elements)
    return RTResult().success(Number.null)

  
  @args(["value"])
  def execute_len(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    try:
      return RTResult().success(Number(len(value.elements)))
    except:
      return RTResult().success(Number(len(value.value)))

  
  @args(["fn"])
  def execute_run(self, exec_ctx):
    fn = exec_ctx.symbol_table.get("fn")

    if not isinstance(fn, String):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Second argument must be string",
        exec_ctx
      ))

    fn = fn.value

    try:
      with open(fn, "r") as f:
        script = f.read()
    except Exception as e:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        f"Failed to load script \"{fn}\"\n" + str(e),
        exec_ctx
      ))

    _, error = run(fn, script)
    
    if error:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        f"Failed to finish executing script \"{fn}\"\n" +
        error.as_string(),
        exec_ctx
      ))

    return RTResult().success(Number.null)

  
  @args(["secs"])
  def execute_wait(self, exec_ctx):
    sym = exec_ctx.symbol_table
    fake_pos = create_fake_pos("<built-in function wait>")
    res = RTResult()

    secs = sym.get("secs")
    if not isinstance(secs, Number):
      return res.failure(RTError(
        fake_pos, fake_pos,
        f"1st argument of function 'wait' ('secs') must be Number",
        exec_ctx
      ))
    secs = secs.value

    time.sleep(secs)

    return RTResult().success(Number.null)

BuiltInFunction.pointer         = BuiltInFunction("pointer")
BuiltInFunction.hex             = BuiltInFunction("hex")
BuiltInFunction.get_mem_val     = BuiltInFunction("get_mem_val")
BuiltInFunction.print           = BuiltInFunction("print")
BuiltInFunction.error           = BuiltInFunction("error")
BuiltInFunction.println         = BuiltInFunction("println")
BuiltInFunction.backgroundColor = BuiltInFunction("backgroundColor")
BuiltInFunction.foregroundColor = BuiltInFunction("foregroundColor")
BuiltInFunction.resetColor      = BuiltInFunction("resetColor")
BuiltInFunction.print_ret       = BuiltInFunction("print_ret")
BuiltInFunction.system          = BuiltInFunction("system")
BuiltInFunction.id              = BuiltInFunction("id")
BuiltInFunction.Int             = BuiltInFunction("Int")
BuiltInFunction.Char            = BuiltInFunction("Char")
BuiltInFunction.List            = BuiltInFunction("List")
BuiltInFunction.bit_to_int      = BuiltInFunction("bit_to_int")
BuiltInFunction.Bin             = BuiltInFunction("Bin")
BuiltInFunction.Float           = BuiltInFunction("Float")
BuiltInFunction.Str             = BuiltInFunction("Str")
BuiltInFunction.split           = BuiltInFunction("split")
BuiltInFunction.from_bytes_int  = BuiltInFunction("from_bytes_int")
BuiltInFunction.input           = BuiltInFunction("input")
BuiltInFunction.input_key       = BuiltInFunction("input_key")
BuiltInFunction.clear           = BuiltInFunction("clear")
BuiltInFunction.exit            = BuiltInFunction("exit")
BuiltInFunction.is_number       = BuiltInFunction("is_number")
BuiltInFunction.is_string       = BuiltInFunction("is_string")
BuiltInFunction.is_list         = BuiltInFunction("is_list")
BuiltInFunction.is_function     = BuiltInFunction("is_function")
BuiltInFunction.append          = BuiltInFunction("append")
BuiltInFunction.pop             = BuiltInFunction("pop")
BuiltInFunction.extend          = BuiltInFunction("extend")
BuiltInFunction.len		          = BuiltInFunction("len")
BuiltInFunction.run			        = BuiltInFunction("run")
BuiltInFunction.wait            = BuiltInFunction("wait")
BuiltInFunction.deref           = BuiltInFunction("deref")
BuiltInFunction.address         = BuiltInFunction("address")
BuiltInFunction.malloc          = BuiltInFunction("malloc")
BuiltInFunction.memmove         = BuiltInFunction("memmove")
BuiltInFunction.free            = BuiltInFunction("free")
BuiltInFunction.virtualAlloc    = BuiltInFunction("virtualAlloc")
BuiltInFunction.virtualFree     = BuiltInFunction("virtualFree")
BuiltInFunction.chr             = BuiltInFunction("chr")
BuiltInFunction.encode          = BuiltInFunction("encode")
BuiltInFunction.decode          = BuiltInFunction("decode")
BuiltInFunction.range           = BuiltInFunction("range")
BuiltInFunction.reverse         = BuiltInFunction("reverse")
BuiltInFunction.extern          = BuiltInFunction("extern")
BuiltInFunction.typeof          = BuiltInFunction("typeof")
BuiltInFunction.repr            = BuiltInFunction("repr")

class GPUFunction(BaseFunction):
    def __init__(self, name):
      super().__init__(name)

  
    def execute(self, args):
      res = RTResult()

      try:
        exec_ctx = self.generate_new_context()

        method_name = f'execute_{self.name}'
        method = getattr(self, method_name, self.no_execute_method)

        res.register(self.check_and_populate_args(method.arg_names, args, method.defaults, method.dynamics, exec_ctx))
        if res.should_return(): return res

        return_value = res.register(method(exec_ctx))
        if res.should_return(): return res
        return res.success(return_value)
      except:
        fake_pos = create_fake_pos("<GPU function CUDA Error>")

        return res.failure(RTError(
          fake_pos, fake_pos,
          f"You don't have CUDA SDK.",
          exec_ctx
        ))

    def no_execute_method(self, node, context):
      raise Exception(f'No execute_{self.name} method defined')

    def copy(self):
      copy = GPUFunction(self.name)
      copy.set_context(self.context)
      copy.set_pos(self.pos_start, self.pos_end)
      return copy

    def __repr__(self):
      return f"<GPU function {self.name}>"

    #####################################

    # Decorator for built-in functions
    @staticmethod
    def args(arg_names, defaults=None, dynamics=None):
      if defaults is None:
        defaults = [None] * len(arg_names)
      if dynamics is None:
        dynamics = [None] * len(arg_names)
      def _args(f):
        f.arg_names = arg_names
        f.defaults = defaults
        f.dynamics = dynamics
        return f
      return _args

    #####################################

    @cuda.jit
    def add(x, y, result):
      idx = cuda.grid(1)
      if idx < x.size:
        result[idx] = x[idx] + y[idx]

    @cuda.jit
    def sub(x, y, result):
      idx = cuda.grid(1)
      if idx < x.size:
        result[idx] = x[idx] - y[idx]
    
    @cuda.jit
    def mul(x, y, result):
      idx = cuda.grid(1)
      if idx < x.size:
        result[idx] = x[idx] * y[idx]

    @cuda.jit
    def div(x, y, result):
      idx = cuda.grid(1)
      if idx < x.size:
        result[idx] = x[idx] / y[idx]

    @cuda.jit
    def sqrt(x, result):
      idx = cuda.grid(1)
      if idx < x.size:
        result[idx] = math.sqrt(x[idx])
    
    @cuda.jit
    def floor(x, result):
      idx = cuda.grid(1)
      if idx < x.size:
        result[idx] = math.floor(x[idx])
    
    @cuda.jit
    def round(x, y, result):
      idx = cuda.grid(1)
      if idx < x.size:
        result[idx] = round(x[idx], y[idx])
  
    @cuda.jit
    def sin(x, result):
      idx = cuda.grid(1)
      if idx < x.size:
        result[idx] = math.sin(x[idx])
    
    @cuda.jit
    def cos(x, result):
      idx = cuda.grid(1)
      if idx < x.size:
        result[idx] = math.cos(x[idx])
    
    @cuda.jit
    def tan(x, result):
      idx = cuda.grid(1)
      if idx < x.size:
        result[idx] = math.tan(x[idx])
    
    @cuda.jit
    def factorial(x, result):
      idx = cuda.grid(1)
      if idx < x.size:
        result[idx] = math.factorial(x[idx])
    
    @cuda.jit
    def radians(x, result):
      idx = cuda.grid(1)
      if idx < x.size:
        result[idx] = math.radians(x[idx])
    
    @cuda.jit
    def gamma(x, result):
      idx = cuda.grid(1)
      if idx < x.size:
        result[idx] = math.gamma(x[idx])
    
    @cuda.jit
    def log2(x, result):
      idx = cuda.grid(1)
      if idx < x.size:
        result[idx] = math.log2(x[idx])
  
    @cuda.jit
    def exp(x, result):
      idx = cuda.grid(1)
      if idx < x.size:
        result[idx] = math.exp(x[idx])
    
    def run_cuda_kernel_2Args(self, kernel, x, y):
      a = np.array([x], dtype=np.float32)
      b = np.array([y], dtype=np.float32)
      c = np.zeros(x, dtype=np.float32)

      a_device = cuda.to_device(a)
      b_device = cuda.to_device(b)
      c_device = cuda.device_array_like(c)

      threads_per_block = 256
      blocks_per_grid = (a.size + (threads_per_block - 1)) // threads_per_block

      kernel[blocks_per_grid, threads_per_block](a_device, b_device, c_device)

      return c_device.copy_to_host()

    def run_cuda_kernel_1Args(self, kernel, x):
      a = np.array([x], dtype=np.float32)
      c = np.zeros(x, dtype=np.float32)

      a_device = cuda.to_device(a)
      c_device = cuda.device_array_like(c)

      threads_per_block = 256
      blocks_per_grid = (a.size + (threads_per_block - 1)) // threads_per_block

      kernel[blocks_per_grid, threads_per_block](a_device, c_device)

      return c_device.copy_to_host()

    @args(['x', 'y'])
    def execute_add(self, exec_ctx):
        x = exec_ctx.symbol_table.get("x")
        y = exec_ctx.symbol_table.get("y")

        result = self.run_cuda_kernel_2Args(self.add, x.value, y.value)[0]

        if isinstance(x, String) and isinstance(y, String):
            return RTResult().success(String(result))
        elif isinstance(x, Number) and isinstance(y, Number):
            return RTResult().success(Number(result))
        elif isinstance(x, Dict) and isinstance(y, Dict):
            return RTResult().success(Dict(result))
    
    @args(['x', 'y'])
    def execute_sub(self, exec_ctx):
        x = exec_ctx.symbol_table.get("x")
        y = exec_ctx.symbol_table.get("y")

        result = self.run_cuda_kernel_2Args(self.sub, x.value, y.value)[0]

        if isinstance(x, String) and isinstance(y, String):
            return RTResult().success(String(result))
        elif isinstance(x, Number) and isinstance(y, Number):
            return RTResult().success(Number(result))
        elif isinstance(x, Dict) and isinstance(y, Dict):
            return RTResult().success(Dict(result))
    
    @args(['x', 'y'])
    def execute_mul(self, exec_ctx):
        x = exec_ctx.symbol_table.get("x")
        y = exec_ctx.symbol_table.get("y")

        result = self.run_cuda_kernel_2Args(self.mul, x.value, y.value)[0]

        if isinstance(x, String) and isinstance(y, String):
            return RTResult().success(String(result))
        elif isinstance(x, Number) and isinstance(y, Number):
            return RTResult().success(Number(result))
        elif isinstance(x, Dict) and isinstance(y, Dict):
            return RTResult().success(Dict(result))
    
    @args(['x', 'y'])
    def execute_div(self, exec_ctx):
        x = exec_ctx.symbol_table.get("x")
        y = exec_ctx.symbol_table.get("y")

        result = self.run_cuda_kernel_2Args(self.add, x.value, y.value)[0]

        if isinstance(x, String) and isinstance(y, String):
            return RTResult().success(String(result))
        elif isinstance(x, Number) and isinstance(y, Number):
            return RTResult().success(Number(result))
        elif isinstance(x, Dict) and isinstance(y, Dict):
            return RTResult().success(Dict(result))

    @args(['x'])
    def execute_sqrt(self, exec_ctx):
        x = exec_ctx.symbol_table.get("x")

        result = self.run_cuda_kernel_1Args(self.sqrt, x.value)

        return RTResult().success(Number(result))
    
    @args(['x', 'y'])
    def execute_round(self, exec_ctx):
        x = exec_ctx.symbol_table.get("x")
        y = exec_ctx.symbol_table.get("y")

        result = self.run_cuda_kernel_2Args(self.round, x.value, y.value)[0]

        if isinstance(x, String) and isinstance(y, String):
            return RTResult().success(String(result))
        elif isinstance(x, Number) and isinstance(y, Number):
            return RTResult().success(Number(result))
        elif isinstance(x, List) and isinstance(y, List):
            return RTResult().success(List(result))
        elif isinstance(x, Dict) and isinstance(y, Dict):
            return RTResult().success(Dict(result))
    
    @args(['x'])
    def execute_floor(self, exec_ctx):
        x = exec_ctx.symbol_table.get("x")

        result = self.run_cuda_kernel_1Args(self.floor, x.value)

        return RTResult().success(Number(result))
    
    @args(['x'])
    def execute_sin(self, exec_ctx):
        x = exec_ctx.symbol_table.get("x")

        result = self.run_cuda_kernel_1Args(self.sin, x.value)

        return RTResult().success(Number(result))
    
    @args(['x'])
    def execute_cos(self, exec_ctx):
        x = exec_ctx.symbol_table.get("x")

        result = self.run_cuda_kernel_1Args(self.cos, x.value)

        return RTResult().success(Number(result))
    
    @args(['x'])
    def execute_tan(self, exec_ctx):
        x = exec_ctx.symbol_table.get("x")

        result = self.run_cuda_kernel_1Args(self.tan, x.value)

        return RTResult().success(Number(result))
    
    @args(['x'])
    def execute_factorial(self, exec_ctx):
        x = exec_ctx.symbol_table.get("x")

        result = self.run_cuda_kernel_1Args(self.factorial, x.value)

        return RTResult().success(Number(result))
    
    @args(['x'])
    def execute_radians(self, exec_ctx):
        x = exec_ctx.symbol_table.get("x")

        result = self.run_cuda_kernel_1Args(self.radians, x.value)

        return RTResult().success(Number(result))
    
    @args(['x'])
    def execute_gamma(self, exec_ctx):
        x = exec_ctx.symbol_table.get("x")

        result = self.run_cuda_kernel_1Args(self.gamma, x.value)

        return RTResult().success(Number(result))
    
    @args(['x'])
    def execute_log2(self, exec_ctx):
        x = exec_ctx.symbol_table.get("x")

        result = self.run_cuda_kernel_1Args(self.log2, x.value)

        return RTResult().success(Number(result))
    
    @args(['x'])
    def execute_exp(self, exec_ctx):
        x = exec_ctx.symbol_table.get("x")

        result = self.run_cuda_kernel_1Args(self.exp, x.value)

        return RTResult().success(Number(result))

GPUFunction.add         = GPUFunction("add")
GPUFunction.sub         = GPUFunction("sub")
GPUFunction.mul         = GPUFunction("mul")
GPUFunction.div         = GPUFunction("div")
GPUFunction.sqrt        = GPUFunction("sqrt")
GPUFunction.round       = GPUFunction("round")
GPUFunction.floor       = GPUFunction("round")
GPUFunction.sin         = GPUFunction("sin")
GPUFunction.cos         = GPUFunction("cos")
GPUFunction.tan         = GPUFunction("tan")
GPUFunction.factorial   = GPUFunction("factorial")
GPUFunction.radians     = GPUFunction("radians")
GPUFunction.gamma       = GPUFunction("gamma")
GPUFunction.log2        = GPUFunction("tan")
GPUFunction.exp         = GPUFunction("exp")

class Iterator(Value):
  def __init__(self, generator):
    super().__init__()
    self.it = generator()
  
  def iter(self):
    return self
  
  def __iter__(self):
    return self

  def __next__(self):
    return next(self.it)
  
  def __str__(self):
    return '<iterator>'

  def __repr__(self):
    return str(self)
  
  def __getattr__(self, attr):
    if attr.startswith("get_comparison_"):
      return lambda self, other: Number(self is other), None

  def copy(self):
    return Iterator(self.it)

class Dict(Value):
  def __init__(self, values):
    super().__init__()
    self.values = values
    self.value = values

  
  def added_to(self, other):
    if not isinstance(other, Dict):
      return None, self.illegal_operation(other)
    
    new_dict = self.copy()
    for key, value in other.values.items():
      new_dict.values[key] = value
    
    return new_dict, None

  
  def gen(self):
    fake_pos = create_fake_pos("<dict key>")
    for key in self.values.keys():
      key_as_value = String(key).set_pos(fake_pos, fake_pos).set_context(self.context)
      yield RTResult().success(key_as_value)

  
  def get_index(self, index):
    if not isinstance(index, String):
      return None, self.illegal_operation(index)
    
    try:
      return self.values[index.value], None
    except KeyError:
      return None, RTError(
        self.pos_start, self.pos_end,
        f"Could not find key {index!r} in dict {self!r}",
        self.context
      )

  
  def set_index(self, index, value):
    if not isinstance(index, String):
      return None, self.illegal_operation(index)
    
    self.values[index.value] = value

    return self, None

  def __str__(self):
    result = ""
    for key, value in self.values.items():
      result += f"{key}: {value}\n"
    
    return result[:-1]
  
  def __repr__(self):
    result = "{"
    for key, value in self.values.items():
      result += f"{key!r}: {value!r}, "
    
    return result[:-2] + "}"

  def copy(self):
    return Dict(self.values).set_pos(self.pos_start, self.pos_end).set_context(self.context)
  
class StructInstance(Value):
    def __init__(self, struct_name, fields):
        super().__init__()
        self.struct_name = struct_name
        self.fields = fields

    def __repr__(self):
        result = f"{self.struct_name} {{"
        for key, value in self.fields.items():
            result += f"{key}: {value!r}, "

        return result[:-2] + "}"

    
    def get_dot(self, verb):
        if verb in self.fields:
            #print(verb)
            #print(self.fields)
            return self.fields[verb].copy(), None
        else:
            return None, RTError(
                self.pos_start, self.pos_end,
                f"Could not find property {verb!r} in struct {self.struct_name!r}",
                self.context)

    
    def set_dot(self, verb, obj):
        if verb in self.fields:
            self.fields[verb] = obj
            return Number.null, None
        else:
            return None, RTError(
                self.pos_start, self.pos_end,
                f"Could not find property {verb!r} in struct {self.struct_name!r}",
                self.context)

    def copy(self):
        return StructInstance(self.struct_name, self.fields).set_pos(self.pos_start, self.pos_end).set_context(self.context)

class ClassInstance(Value):
    def __init__(self, class_name, fields):
        super().__init__()
        self.class_name = class_name
        self.fields = fields

    
    def __repr__(self):
        result = f"{self.class_name} {{"
        for key, value in self.fields.items():
            result += f"{key}: {value!r}, "
        return result[:-2] + "}"

    
    def get_dot(self, verb):
        if verb in self.fields:
            return self.fields[verb], None
        else:
            return None, RTError(
                self.pos_start, self.pos_end,
                f"Could not find property {verb!r} in class {self.class_name!r}",
                self.context)

    
    def set_dot(self, verb, obj):
        if verb in self.fields:
            self.fields[verb] = obj
            return Number.null, None
        else:
            return None, RTError(
                self.pos_start, self.pos_end,
                f"Could not find property {verb!r} in struct {self.struct_name!r}",
                self.context)

    def copy(self):
        return ClassInstance(self.class_name, self.fields).set_pos(self.pos_start, self.pos_end).set_context(self.context)

#######################################
# CONTEXT
#######################################

class Context:
  def __init__(self, display_name, parent=None, parent_entry_pos=None):
    self.display_name = display_name
    self.parent = parent
    self.parent_entry_pos = parent_entry_pos
    self.symbol_table = None

#######################################
# SYMBOL TABLE
#######################################

class SymbolTable:
  def __init__(self, parent=None):
    self.symbols = {}
    self.structs: dict[str: dict[str: Any]] = {}
    self.classes = {}
    self.parent = parent
    self.const = set()
    self.globall = set()

  
  def get(self, name):
    value = self.symbols.get(name, None)
    if value == None and self.parent:
      return self.parent.get(name)
    return value

  
  def get_global(self, name):
    global global_variables
    
    value = global_variables[name]
    if value == None and self.parent:
      return self.parent.get_global(name)
    return value

  def set(self, name, value):
    self.symbols[name] = value

  def set_const(self, name, value):
    self.symbols[name] = value
    self.const.add(name)

  def set_global(self, name, value):
    global global_variables

    global_variables[name] = value
    self.globall.add(name)

  def remove(self, name):
    del self.symbols[name]

#######################################
# INTERPRETER
#######################################

global_symbol_table = SymbolTable()

class Interpreter:
  def __init__(self):
    self.context = None
    self.node = None

  
  def visit(self, node, context):
    self.context = context
    self.node = node

    method_name = f'visit_{type(node).__name__}'
    method = getattr(self, method_name, self.no_visit_method)
    return method(node, context)

  def no_visit_method(self, node, context):
    raise Exception(f'No visit_{type(node).__name__} method defined')

  ###################################

  
  def visit_NumberNode(self, node, context):
    return RTResult().success(
      Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  
  def visit_BinNode(self, node, context):
    return RTResult().success(
      Bin(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  
  def visit_ByteNode(self, node, context):
    return RTResult().success(
      Bytes(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  
  def visit_StringNode(self, node, context):
    return RTResult().success(
      String(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  
  def visit_ListNode(self, node, context):
    res = RTResult()
    elements = []

    for element_node in node.element_nodes:
      elements.append(res.register(self.visit(element_node, context)))
      if res.should_return(): return res

    return res.success(
      List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
    )
  
  def visit_VarAccessNode(self, node, context):
    global global_variables

    res = RTResult()
    var_name = node.var_name_tok.value
    value = None

    if var_name in global_variables:
      value = context.symbol_table.get_global(var_name)
    else:
      value = context.symbol_table.get(var_name)

    if not value:
      return res.failure(RTError(
        node.pos_start, node.pos_end,
        f"'{var_name}' is not defined",
        context
      ))
    
    if not isinstance(value, (true, false, Null)):
      value = value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
    return res.success(value)

  
  def visit_VarAssignNode(self, node, context):
    res = RTResult()
    var_name = node.var_name_tok.value
    op = node.current_op

    if op == [TokenType.PLUS, TokenType.EQ]:
      left = res.register(self.visit(node.value_node, context))
      right = context.symbol_table.get(var_name)

      if isinstance(left, Number):
        value = Number(left.value + right.value)
      if isinstance(left, Bin):
        value = Bin(left.value + right.value)
      if isinstance(left, Bytes):
        value = Bytes(left.value + right.value)
      if isinstance(left, String):
        value = String(left.value + right.value)
      if isinstance(left, List):
        value = List(left.value + right.value)
      if isinstance(left, Dict):
        value = Dict(left.value + right.value)
    elif op == [TokenType.MINUS, TokenType.EQ]:
      left = res.register(self.visit(node.value_node, context))
      right = context.symbol_table.get(var_name)

      if isinstance(left, Number):
        value = Number(left.value - right.value)
      if isinstance(left, Bin):
        value = Bin(left.value - right.value)
      if isinstance(left, Bytes):
        value = Bytes(left.value - right.value)
      if isinstance(left, String):
        value = String(left.value - right.value)
      if isinstance(left, List):
        value = List(left.value - right.value)
      if isinstance(left, Dict):
        value = Dict(left.value - right.value)
    elif op == [TokenType.MUL, TokenType.EQ]:
      left = res.register(self.visit(node.value_node, context))
      right = context.symbol_table.get(var_name)

      if isinstance(left, Number):
        value = Number(left.value * right.value)
      if isinstance(left, Bin):
        value = Bin(left.value * right.value)
      if isinstance(left, Bytes):
        value = Bytes(left.value * right.value)
      if isinstance(left, String):
        value = String(left.value * right.value)
      if isinstance(left, List):
        value = List(left.value * right.value)
      if isinstance(left, Dict):
        value = Dict(left.value * right.value)
    elif op == [TokenType.DIV, TokenType.EQ]:
      left = res.register(self.visit(node.value_node, context))
      right = context.symbol_table.get(var_name)

      if isinstance(left, Number):
        value = Number(left.value / right.value)
      if isinstance(left, Bin):
        value = Bin(left.value / right.value)
      if isinstance(left, Bytes):
        value = Bytes(left.value / right.value)
      if isinstance(left, String):
        value = String(left.value / right.value)
      if isinstance(left, List):
        value = List(left.value / right.value)
      if isinstance(left, Dict):
        value = Dict(left.value / right.value)
    elif op == [TokenType.PERC, TokenType.EQ]:
      left = res.register(self.visit(node.value_node, context))
      right = context.symbol_table.get(var_name)

      if isinstance(left, Number):
        value = Number(left.value % right.value)
      if isinstance(left, Bin):
        value = Bin(left.value % right.value)
      if isinstance(left, Bytes):
        value = Bytes(left.value % right.value)
      if isinstance(left, String):
        value = String(left.value % right.value)
      if isinstance(left, List):
        value = List(left.value % right.value)
      if isinstance(left, Dict):
        value = Dict(left.value % right.value)
    elif op == [TokenType.POW, TokenType.EQ]:
      left = res.register(self.visit(node.value_node, context))
      right = context.symbol_table.get(var_name)

      if isinstance(left, Number):
        value = Number(left.value ** right.value)
      if isinstance(left, Bin):
        value = Bin(left.value ** right.value)
      if isinstance(left, Bytes):
        value = Bytes(left.value ** right.value)
      if isinstance(left, String):
        value = String(left.value ** right.value)
      if isinstance(left, List):
        value = List(left.value ** right.value)
      if isinstance(left, Dict):
        value = Dict(left.value ** right.value)
    elif op == [TokenType.EQ]:
      value = res.register(self.visit(node.value_node, context))

    if res.should_return(): return res

    if node.is_const:
      method = context.symbol_table.set_const
    elif node.is_global:
      method = context.symbol_table.set_global
    else:
      method = context.symbol_table.set
    
    if var_name not in context.symbol_table.const:
      method(var_name, value)
      return res.success(value)
    else:
      return res.failure(RTError(
        node.pos_start, node.pos_end,
        f"Assignment to constant variable '{var_name}'",
        context
      ))

  
  def visit_BinOpNode(self, node, context):
    res = RTResult()
    left = res.register(self.visit(node.left_node, context))
    if res.should_return(): return res
    right = res.register(self.visit(node.right_node, context))
    if res.should_return(): return res

    if node.op_tok.type == TokenType.PLUS:
      result, error = left.added_to(right)
    elif node.op_tok.type == TokenType.MINUS:
      result, error = left.subbed_by(right)
    elif node.op_tok.type == TokenType.MUL:
      result, error = left.multed_by(right)
    elif node.op_tok.type == TokenType.DIV:
      result, error = left.dived_by(right)
    elif node.op_tok.type == TokenType.POW:
      result, error = left.powed_by(right)
    elif node.op_tok.type == TokenType.PERC:
      result, error = left.percent_by(right)
    elif node.op_tok.type == TokenType.EE:
      result, error = left.get_comparison_eq(right)
    elif node.op_tok.type == TokenType.NE:
      result, error = left.get_comparison_ne(right)
    elif node.op_tok.type == TokenType.LT:
      result, error = left.get_comparison_lt(right)
    elif node.op_tok.type == TokenType.GT:
      result, error = left.get_comparison_gt(right)
    elif node.op_tok.type == TokenType.LTE:
      result, error = left.get_comparison_lte(right)
    elif node.op_tok.type == TokenType.GTE:
      result, error = left.get_comparison_gte(right)
    elif node.op_tok.type == TokenType.BITWISEXOR:
      result, error = left.xored(right)
    elif node.op_tok.type == TokenType.LEFTSH:
      result, error = left.left_shifted(right)
    elif node.op_tok.type == TokenType.RIGHTSH:
      result, error = left.right_shifted(right)
    elif node.op_tok.type == TokenType.BITWISEAND:
      result, error = left.bitwise_and(right)
    elif node.op_tok.type == TokenType.BITWISEOR:
      result, error = left.bitwise_or(right)
    elif node.op_tok.matches(TokenType.KEYWORD, 'and'):
      result, error = left.anded_by(right)
    elif node.op_tok.matches(TokenType.KEYWORD, 'or'):
      result, error = left.ored_by(right)

    if error:
      return res.failure(error)
    else:
      return res.success(result.set_pos(node.pos_start, node.pos_end))

  
  def visit_UnaryOpNode(self, node, context):
    res = RTResult()
    number = res.register(self.visit(node.node, context))
    if res.should_return(): return res

    error = None

    if node.op_tok.type == TokenType.MINUS:
      number, error = number.multed_by(Number(-1))
    elif node.op_tok.matches(TokenType.KEYWORD, 'not'):
      number, error = number.notted()
    elif node.op_tok.type == TokenType.BITWISENOT:
      number, error = number.bitwise_not()

    if error:
      return res.failure(error)
    else:
      return res.success(number.set_pos(node.pos_start, node.pos_end))

  
  def visit_IfNode(self, node, context):
    res = RTResult()

    for condition, expr, should_return_null in node.cases:
      condition_value = res.register(self.visit(condition, context))
      if res.should_return(): return res

      if condition_value.is_true():
        expr_value = res.register(self.visit(expr, context))
        if res.should_return(): return res
        return res.success(Number.null if should_return_null else expr_value)

    if node.else_case:
      expr, should_return_null = node.else_case
      expr_value = res.register(self.visit(expr, context))
      if res.should_return(): return res
      return res.success(Number.null if should_return_null else expr_value)

    return res.success(Number.null)

  
  def visit_ForNode(self, node, context):
        res = RTResult()
        elements = []

        start_value = res.register(self.visit(node.start_value_node, context))
        if res.should_return():
            return res

        end_value = res.register(self.visit(node.end_value_node, context))
        if res.should_return():
            return res

        if node.step_value_node:
            step_value = res.register(
                self.visit(node.step_value_node, context))
            if res.should_return():
                return res
        else:
            step_value = Number(1)

        i = start_value.value

        if step_value.value >= 0:
            def condition(): return i < end_value.value + 1
        else:
            def condition(): return i > end_value.value + 1

        while condition():
            context.symbol_table.set(node.var_name_tok.value, Number(i))
            i += step_value.value

            value = res.register(self.visit(node.body_node, context))
            if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False:
                return res

            if res.loop_should_continue:
                continue

            if res.loop_should_break:
                break

            elements.append(value)

        return res.success(
            Number.null if node.should_return_null else
            List(elements).set_context(context).set_pos(
                node.pos_start, node.pos_end)
        )

  def visit_WhileNode(self, node, context):
        res = RTResult()
        elements = []

        while True:
            condition = res.register(self.visit(node.condition_node, context))
            if res.should_return():
                return res

            if not condition.is_true():
                break

            value = res.register(self.visit(node.body_node, context))
            if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False:
                return res

            if res.loop_should_continue:
                continue

            if res.loop_should_break:
                break

            elements.append(value)

        return res.success(
            Number.null if node.should_return_null else
            List(elements).set_context(context).set_pos(
                node.pos_start, node.pos_end)
        )
  
  def visit_FuncDefNode(self, node, context):
    res = RTResult()

    func_name = node.var_name_tok.value if node.var_name_tok else None
    body_node = node.body_node
    arg_names = [arg_name.value for arg_name in node.arg_name_toks]
    defaults = []
    for default in node.defaults:
      if default is None:
        defaults.append(None)
        continue
      default_value = res.register(self.visit(default, context))
      if res.should_return(): return res
      defaults.append(default_value)
    
    func_value = Function(func_name, body_node, arg_names, defaults, node.dynamics, node.should_auto_return).set_context(context).set_pos(node.pos_start, node.pos_end)
    
    if node.var_name_tok:
      context.symbol_table.set(func_name, func_value)

    return res.success(func_value)

  
  def visit_CallNode(self, node, context):
    res = RTResult()
    args = []

    global current_class

    value_to_call = res.register(self.visit(node.node_to_call, context))
    if res.should_return(): return res
    value_to_call = value_to_call.copy().set_pos(node.pos_start, node.pos_end)

    for arg_node in node.arg_nodes:
      args.append(res.register(self.visit(arg_node, context)))

      if res.should_return(): return res

    try:
      if value_to_call.name in res.register(self.visit(current_class, context)).fields:
        args.append(res.register(self.visit(current_class, context)))
      else:
        pass
    except:
      pass

    current_class = None

    return_value = res.register(value_to_call.execute(args))
    
    if res.should_return(): return res
    
    if not isinstance(return_value, (true, false, Null)):
      return_value = return_value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)

    return res.success(return_value)

  def visit_ReturnNode(self, node, context):
    res = RTResult()

    if node.node_to_return:
      value = res.register(self.visit(node.node_to_return, context))
      if res.should_return(): return res
    else:
      value = Number.null
    
    return res.success_return(value)

  def visit_YieldNode(self, node, context):
    res = RTResult()
    
    if node.node_to_yield:
      value = res.register(self.visit(node.node_to_yield, context))
      
      if res.should_yield(): return res
    else:
      value = Number.null
    
    return res.success_yield(value)

  
  def visit_ContinueNode(self, node, context):
    return RTResult().success_continue()

  
  def visit_BreakNode(self, node, context):
    return RTResult().success_break()

  
  def visit_PassNode(self, node, context):
    return RTResult().success_pass()

  
  def visit_ImportNode(self, node, context):
    res = RTResult()
    filename = res.register(self.visit(node.string_node, context))
    code = None

    filepath = None

    dumped_import_path = yaml.safe_load(IMPORT_PATH)

    for yaml_node, val in dumped_import_path.items():
      if yaml_node == "dependencies":
        for name, path in val.items():
          filepath = path

          if name == filename.value:
            try:
              with open(path, "r") as f:
                code = f.read()
                beginning = "/" if path.startswith("/") else ""
                split = path.split("/")
                split = beginning + "/".join(split[:-1]), split[-1]
                os.chdir(split[0])
                filename = split[1]
                break
            except FileNotFoundError:
              continue
    
    if code is None:
      return res.failure(RTError(
        node.string_node.pos_start.copy(), node.string_node.pos_end.copy(),
        f"Can't find file '{filepath}' in '{filename}'. Please add the directory your file is into that file",
        context
      ))
    
    _, error = run(filepath, code, context, node.pos_start.copy())
    if error: return res.failure(error)

    return res.success(Number.null)

  
  def visit_DoNode(self, node, context):
    res = RTResult()
    new_context = Context("<do statement>", context, node.pos_start.copy())
    new_context.symbol_table = SymbolTable(context.symbol_table)
    res.register(self.visit(node.statements, new_context))
    
    return_value = res.func_return_value
    yield_value = res.func_yield_value
    if res.should_yield() and yield_value is None or res.should_return() and return_value is None:
      return res
    
    return_value = return_value or Number.null
    yield_value = yield_value or Number.null
    
    if isinstance(return_value, Null):
      return res.success(yield_value)
    else:
      return res.success(return_value)

  
  def visit_TryNode(self, node: TryNode, context):
    res = RTResult()
    res.register(self.visit(node.try_block, context))
    handled_error = res.error
    if res.should_return() and res.error is None: return res

    elif handled_error is not None:
      var_name = node.exc_iden.value
      context.symbol_table.set(var_name, res.error)
      res.error = None

      res.register(self.visit(node.catch_block, context))
      if res.error: 
        return res.failure(TryError(
          res.error.pos_start, res.error.pos_end, res.error.details, res.error.context, handled_error
        ))
      return res.success(Number.null)
    else:
      return res.success(Number.null)

  
  def visit_ForInNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        body = node.body_node
        should_return_null = node.should_return_null

        iterable = res.register(self.visit(node.iterable_node, context))
        it = iterable.iter()

        elements = []

        for it_res in it:
            elt = res.register(it_res)
            if res.should_return():
                return res

            context.symbol_table.set(var_name, elt)

            elements.append(res.register(self.visit(body, context)))
            if res.should_return():
                return res

        if should_return_null:
            return res.success(Number.null)
        return res.success(elements)
        
  
  def visit_IndexGetNode(self, node, context):
    res = RTResult()
    indexee = res.register(self.visit(node.indexee, context))
    if res.should_return(): return res

    index = res.register(self.visit(node.index, context))
    if res.should_return(): return res

    result, error = indexee.get_index(index)
    if error: return res.failure(error)
    return res.success(result)

  
  def visit_IndexSetNode(self, node, context):
    res = RTResult()
    indexee = res.register(self.visit(node.indexee, context))
    if res.should_return(): return res

    index = res.register(self.visit(node.index, context))
    if res.should_return(): return res

    value = res.register(self.visit(node.value, context))
    if res.should_return(): return res

    result, error = indexee.set_index(index, value)
    if error: return res.failure(error)

    return res.success(result)

  
  def visit_DictNode(self, node, context):
    res = RTResult()
    values = {}

    for key_node, value_node in node.pairs:
      key = res.register(self.visit(key_node, context))
      if res.should_return(): return res

      if not isinstance(key, String):
        return res.failure(RTError(
          key_node.pos_start, key_node.pos_end,
          f"Non-string key for dict: '{key!r}'",
          context
        ))

      value = res.register(self.visit(value_node, context))
      if res.should_return(): return res

      values[key.value] = value
    
    return res.success(Dict(values))

  
  def visit_SwitchNode(self, node, context):
    res = RTResult()
    condition = res.register(self.visit(node.condition, context))
    if res.should_return(): return res

    for case, body in node.cases:
      case = res.register(self.visit(case, context))
      if res.should_return(): return res

      eq, error = condition.get_comparison_eq(case)
      if error: return res.failure(error)

      if eq.value:
        res.register(self.visit(body, context))
        if res.should_return(): return res
        break
    else: # no break
      else_case = node.else_case
      if else_case:
        res.register(self.visit(else_case, context))
        if res.should_return(): return res
    
    return res.success(Number.null)

  
  def visit_DotGetNode(self, node, context):
    res = RTResult()
    noun = res.register(self.visit(node.noun, context))
    if res.should_return(): return res

    global current_class

    verb = node.verb.value

    if isinstance(noun, ClassInstance):
      current_class = node.noun
    else:
      current_class = None

    result, error = noun.get_dot(verb)

    if error: return res.failure(error)
    return res.success(result)

  
  def visit_DotSetNode(self, node, context):
    res = RTResult()
    noun = res.register(self.visit(node.noun, context))
    if res.should_return(): return res
    
    verb = node.verb.value

    value = res.register(self.visit(node.value, context))
    if res.should_return(): return res
    
    result, error = noun.set_dot(verb, value)
    if error: return res.failure(error)

    return res.success(result)

  
  def visit_StructNode(self, node, ctx):
        # TODO: report struct redefinition
        ctx.symbol_table.structs[node.name] = node.fields
        return RTResult().success(Number.null)

  
  def visit_ClassNode(self, node, ctx):
    # TODO: report class redefinition 
    global classes

    ctx.symbol_table.classes[node.name] = node.fields
    return RTResult().success(Number.null)

  
  def visit_StructCreationNode(self, node, ctx):
        res = RTResult()
        struct = ctx.symbol_table.structs[node.name]

        return res.success(StructInstance(node.name, {field: Number.null for field in struct}).set_pos(node.pos_start, node.pos_end).set_context(ctx))

  
  def visit_ClassCreationNode(self, node, ctx):
        res = RTResult()
        fields = {}
        global classes
        global global_variables

        clases = None

        try:
          clases = ctx.symbol_table.classes[node.name]
        except:
          clases = classes[node.name]
        
        for clss in clases:
          fields[clss] = self.visit(classes[node.name][clss], self.context).value

        return res.success(ClassInstance(node.name, fields).set_pos(node.pos_start, node.pos_end).set_context(ctx))

#######################################
# CREATE FAKE POS
#######################################

def create_fake_pos(desc: str) -> Position:
  return Position(0, 0, 0, desc, "<native code>")

#######################################
# RUN
#######################################

def make_argv():
  argv = []
  fake_pos = create_fake_pos("<argv>")
  for arg in sys.argv[1:]:
    argv.append(String(arg).set_pos(fake_pos, fake_pos))
  return List(argv).set_pos(fake_pos, fake_pos)

global_symbol_table.set("null", Number.null)
global_symbol_table.set("false", Number.false)
global_symbol_table.set("true", Number.true)
global_symbol_table.set("Argv", make_argv())
global_symbol_table.set("math_pi", Number.math_PI)
global_symbol_table.set("pointer", BuiltInFunction.pointer)
global_symbol_table.set("get_mem_val", BuiltInFunction.get_mem_val)
global_symbol_table.set("hex", BuiltInFunction.hex)
global_symbol_table.set("print", BuiltInFunction.print)
global_symbol_table.set("error", BuiltInFunction.error)
global_symbol_table.set("println", BuiltInFunction.println)
global_symbol_table.set("backgroundColor", BuiltInFunction.backgroundColor)
global_symbol_table.set("foregroundColor", BuiltInFunction.foregroundColor)
global_symbol_table.set("resetColor", BuiltInFunction.resetColor)
global_symbol_table.set("id", BuiltInFunction.id)
global_symbol_table.set("bit_to_int", BuiltInFunction.bit_to_int)
global_symbol_table.set("Int", BuiltInFunction.Int)
global_symbol_table.set("Char", BuiltInFunction.Char)
global_symbol_table.set("List", BuiltInFunction.List)
global_symbol_table.set("Bin", BuiltInFunction.Bin)
global_symbol_table.set("Float", BuiltInFunction.Float)
global_symbol_table.set("Str", BuiltInFunction.Str)
global_symbol_table.set("split", BuiltInFunction.split)
global_symbol_table.set("from_bytes_int", BuiltInFunction.from_bytes_int)
global_symbol_table.set("system", BuiltInFunction.system)
global_symbol_table.set("print_ret", BuiltInFunction.print_ret)
global_symbol_table.set("input", BuiltInFunction.input)
global_symbol_table.set("input_key", BuiltInFunction.input_key)
global_symbol_table.set("clear", BuiltInFunction.clear)
global_symbol_table.set("exit", BuiltInFunction.exit)
global_symbol_table.set("cls", BuiltInFunction.clear)
global_symbol_table.set("is_number", BuiltInFunction.is_number)
global_symbol_table.set("is_string", BuiltInFunction.is_string)
global_symbol_table.set("is_list", BuiltInFunction.is_list)
global_symbol_table.set("is_function", BuiltInFunction.is_function)
global_symbol_table.set("append", BuiltInFunction.append)
global_symbol_table.set("pop", BuiltInFunction.pop)
global_symbol_table.set("extend", BuiltInFunction.extend)
global_symbol_table.set("len", BuiltInFunction.len)
global_symbol_table.set("Run", BuiltInFunction.run)
global_symbol_table.set("wait", BuiltInFunction.wait)
global_symbol_table.set("deref", BuiltInFunction.deref)
global_symbol_table.set("address", BuiltInFunction.address)
global_symbol_table.set("malloc", BuiltInFunction.malloc)
global_symbol_table.set("memmove", BuiltInFunction.memmove)
global_symbol_table.set("free", BuiltInFunction.free)
global_symbol_table.set("virtualAlloc", BuiltInFunction.virtualAlloc)
global_symbol_table.set("virtualFree", BuiltInFunction.virtualFree)
global_symbol_table.set("chr", BuiltInFunction.chr)
global_symbol_table.set("encode", BuiltInFunction.encode)
global_symbol_table.set("decode", BuiltInFunction.decode)
global_symbol_table.set("range", BuiltInFunction.range)
global_symbol_table.set("reverse", BuiltInFunction.reverse)
global_symbol_table.set("extern", BuiltInFunction.extern)
global_symbol_table.set("typeof", BuiltInFunction.typeof)
global_symbol_table.set("repr", BuiltInFunction.repr)

global_symbol_table.classes["CUDA"] = {
    "add": GPUFunction.add,
    "sub": GPUFunction.sub,
    "mul": GPUFunction.mul,
    "div": GPUFunction.div,
    "sqrt": GPUFunction.sqrt,
    "round": GPUFunction.round,
    "sin": GPUFunction.sin,
    "cos": GPUFunction.cos,
    "tan": GPUFunction.tan,
    "exp": GPUFunction.exp,
    "gamma": GPUFunction.gamma,
    "log2": GPUFunction.log2,
    "factorial": GPUFunction.factorial,
    "floor": GPUFunction.floor,
    "radians": GPUFunction.radians
}

global_symbol_table.set("CUDA", ClassInstance("CUDA", global_symbol_table.classes["CUDA"]))

def run(fn, text, context=None, entry_pos=None):
    # Generate tokens
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()

    n_tokens = []

    for tok in tokens:
        if tok is not None:
            n_tokens.append(tok)

    if error:
        return None, error

    # Generate AST
    parser = Parser(n_tokens)
    ast = parser.parse()
    if ast.error:
        return None, ast.error

    # Run program
    interpreter = Interpreter()

    # Ensure context is not None
    if context is None:
        context = Context('<program>', None, entry_pos)
        context.symbol_table = global_symbol_table
    else:
        context.symbol_table = context.parent.symbol_table if context.parent else global_symbol_table
        
    result = interpreter.visit(ast.node, context)
    ret = result.func_return_value
    if context.parent is None and ret:
        if not isinstance(ret, Number):
            return None, RTError(
                ret.pos_start, ret.pos_end,
                "Exit code must be Number",
                context
            )
        sys.exit(ret.value)

    return result.value, result.error

def isLangExtension(x):
	if x.endswith(".br"):
		return x
	else:
		y = x.split('.')

		print("Extension Error: The extension needs be '.br', not '." + y[-1] + "'")

		sys.exit(1)

		return None
