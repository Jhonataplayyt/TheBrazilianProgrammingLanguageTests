import string, os, math, time, sys, pickle, pydantic, importlib
from dataclasses import dataclass
from enum import Enum, auto
from typing import *
from .Values import *
from .Pos import *

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
        import struct
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

#######################################
# OPEN FILES (so they don't get automatically closed by GC)
#######################################

files = {}

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

  def register(self, res):
    self.error = res.error
    self.func_return_value = res.func_return_value
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

def make_argv():
  argv = []
  fake_pos = create_fake_pos("<argv>")
  for arg in sys.argv[1:]:
    argv.append(String(arg).set_pos(fake_pos, fake_pos))
  return List(argv).set_pos(fake_pos, fake_pos)

global_symbol_table = SymbolTable()
global_symbol_table.set("null", Number.null)
global_symbol_table.set("false", Number.false)
global_symbol_table.set("true", Number.true)
global_symbol_table.set("Argv", make_argv())
global_symbol_table.set("math_pi", Number.math_PI)
global_symbol_table.set("print", BuiltInFunction.print)
global_symbol_table.set("println", BuiltInFunction.println)
global_symbol_table.set("id", BuiltInFunction.id)
global_symbol_table.set("bit_to_int", BuiltInFunction.bit_to_int)
global_symbol_table.set("int", BuiltInFunction.int)
global_symbol_table.set("bin", BuiltInFunction.bin)
global_symbol_table.set("float", BuiltInFunction.float)
global_symbol_table.set("str", BuiltInFunction.str)
global_symbol_table.set("split", BuiltInFunction.split)
global_symbol_table.set("from_bytes_int", BuiltInFunction.from_bytes_int)
global_symbol_table.set("system", BuiltInFunction.system)
global_symbol_table.set("print_ret", BuiltInFunction.print_ret)
global_symbol_table.set("input", BuiltInFunction.input)
global_symbol_table.set("clear", BuiltInFunction.clear)
global_symbol_table.set("cls", BuiltInFunction.clear)
global_symbol_table.set("is_num", BuiltInFunction.is_number)
global_symbol_table.set("is_str", BuiltInFunction.is_string)
global_symbol_table.set("is_list", BuiltInFunction.is_list)
global_symbol_table.set("is_fun", BuiltInFunction.is_function)
global_symbol_table.set("append", BuiltInFunction.append)
global_symbol_table.set("pop", BuiltInFunction.pop)
global_symbol_table.set("extend", BuiltInFunction.extend)
global_symbol_table.set("len", BuiltInFunction.len)
global_symbol_table.set("Run", BuiltInFunction.run)
global_symbol_table.set("open", BuiltInFunction.open)
global_symbol_table.set("read", BuiltInFunction.read)
global_symbol_table.set("write", BuiltInFunction.write)
global_symbol_table.set("close", BuiltInFunction.close)
global_symbol_table.set("wait", BuiltInFunction.wait)