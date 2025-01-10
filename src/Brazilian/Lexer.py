import string, os, math, time, sys, pickle, pydantic, importlib
from dataclasses import dataclass
from enum import Enum, auto
from typing import *
from .Pos import *
from .Constants import *
from .More import *
from .Errors import *
from .Tokens import *

class Lexer:
  def __init__(self, fn, text):
    self.fn = fn
    self.text = text
    self.pos = Position(-1, 0, -1, fn, text)
    self.current_char = None
    self.advance()
  
  def advance(self):
    self.pos.advance(self.current_char)
    self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

  def make_tokens(self):
    tokens = []

    while self.current_char != None:
      if self.current_char in SINGLE_CHAR_TOKS:
        tt = SINGLE_CHAR_TOKS[self.current_char]
        pos = self.pos.copy()
        self.advance()
        tokens.append(Token(tt, pos_start=pos))
      elif self.current_char.isspace():
        self.advance()
      elif self.current_char == '#':
        self.skip_comment()
      elif self.current_char in DIGITS:
        tokens.append(self.make_number())
      elif self.current_char in VALID_IDENTIFIERS:
        tokens.append(self.make_identifier())
      elif self.current_char == '"':
        tokens.append(self.make_string())
      elif self.current_char == '~':
        tokens.append(self.make_bytes())
      elif self.current_char == '-':
        tokens.append(self.make_minus_or_arrow())
      elif self.current_char == '!':
        token, error = self.make_not_equals()
        if error: return [], error
        tokens.append(token)
      elif self.current_char == '\\':
        self.advance()
      elif self.current_char == '=':
        tokens.append(self.make_equals())
      elif self.current_char == '<':
        tokens.append(self.make_less_than())
      elif self.current_char == '>':
        tokens.append(self.make_greater_than())
      else:
        pos_start = self.pos.copy()
        char = self.current_char
        self.advance()
        return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

    tokens.append(Token(TokenType.EOF, pos_start=self.pos))
    return tokens, None

  def make_number(self):
    num_str = ''
    dot_count = 0
    pos_start = self.pos.copy()

    while self.current_char != None and self.current_char in DIGITS + '.':
      if self.current_char == '.':
        if dot_count == 1: break
        dot_count += 1
      num_str += self.current_char
      self.advance()

    if dot_count == 0:
      return Token(TokenType.INT, int(num_str), pos_start, self.pos)
    else:
      return Token(TokenType.FLOAT, float(num_str), pos_start, self.pos)

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

    while self.current_char != None and self.current_char in VALID_IDENTIFIERS:
      id_str += self.current_char
      self.advance()

    tok_type = TokenType.KEYWORD if id_str in KEYWORDS else TokenType.IDENTIFIER
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

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def make_greater_than(self):
    tok_type = TokenType.GT
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      tok_type = TokenType.GTE

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