import string, os, math, time, sys, pickle, pydantic, importlib
from dataclasses import dataclass
from enum import Enum, auto
from typing import *
from .Errors import Value, RTError
from .More import to_bytes, convert_forL, to_bytes_forL, RTResult, files
from .Pos import Position, create_fake_pos 
from .Lang import Interpreter
from .Brazilian import run

class Number(Value):
  def __init__(self, value):
    super().__init__()
    self.value = value

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
    copy = Number(self.value)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def is_true(self):
    return self.value != 0

  def __str__(self):
    return str(self.value)
  
  def __repr__(self):
    return str(self.value)

Number.null = Number(0)
Number.false = Number(0)
Number.true = Number(1)
Number.math_PI = Number(math.pi)

class Bin(Value):
  def __init__(self, value):
    super().__init__()
    self.value = value

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
      return Bin(bin(self.value == other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_ne(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bin(bin(self.value != other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_lt(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bin(bin(self.value < other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_gt(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bin(bin(self.value > other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_lte(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bin(bin(self.value <= other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_gte(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bin(bin(self.value >= other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def anded_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bin(bin(self.value and other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def ored_by(self, other):
    if isinstance(other, Bin) or isinstance(other, Number) or isinstance(other, Bytes) or isinstance(other, String):
      return Bin(bin(self.value or other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def notted(self):
    return Bin(bin(1) if self.value == bin(0) else bin(0)).set_context(self.context), None

  def copy(self):
    copy = Bin(self.value)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def is_true(self):
    return self.value != 0

  def __str__(self):
    return str(bin(self.value))
  
  def __repr__(self):
    return str(bin(self.value))
  
class Bytes(Value):
  def __init__(self, value):
    super().__init__()
    self.value = value

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
    copy = Bytes(self.value)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def is_true(self):
    return self.value != 0

  def __str__(self):
    return str(to_bytes(self.value))
  
  def __repr__(self):
    return str(to_bytes(self.value))

class String(Value):
  def __init__(self, value):
    super().__init__()
    self.value = value

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

  def percent_by(self, other):
    if isinstance(other, Number):
      return String(self.value % other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def gen(self):
    for char in self.value:
      yield RTResult().success(String(char))

  def get_index(self, index):
    if not isinstance(index, Number):
      return None, self.illegal_operation(index)
    try:
      return self.value[index.value], None
    except IndexError:
      return None, RTError(
        index.pos_start, index.pos_end,
        f"Cannot retrieve character {index} from string {self!r} because it is out of bounds.",
        self.context
      )
  
  def get_comparison_eq(self, other):
    if not isinstance(other, String):
      return None, self.illegal_operation(other)
    return Number(int(self.value == other.value)), None
  
  def get_comparison_ne(self, other):
    if not isinstance(other, String):
      return None, self.illegal_operation(other)
    return Number(int(self.value != other.value)), None

  def is_true(self):
    return len(self.value) > 0

  def copy(self):
    copy = String(self.value)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def __str__(self):
    return self.value

  def __repr__(self):
    return f'"{self.value}"'

class List(Value):
  def __init__(self, elements):
    super().__init__()
    self.elements = elements
    self.value = elements

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
      try:
        return self.elements[other.value], None
      except:
        return None, RTError(
          other.pos_start, other.pos_end,
          'Element at this index could not be retrieved from list because index is out of bounds',
          self.context
        )
    else:
      return None, Value.illegal_operation(self, other)
  
  def gen(self):
    for elt in self.elements:
      yield RTResult().success(elt)

  def get_index(self, index):
    if not isinstance(index, Number):
      return None, self.illegal_operation(index)
    try:
      return self.elements[index.value], None
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
      self.elements[index.value] = value
    except IndexError:
      return None, RTError(
        index.pos_start, index.pos_end,
        f"Cannot set element {index} from list {self!r} to {value!r} because it is out of bounds.",
        self.context
      )
    
    return self, None

  def copy(self):
    copy = List(self.elements)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def __str__(self):
    return ", ".join([str(x) for x in self.elements])

  def __repr__(self):
    return f'[{", ".join([repr(x) for x in self.elements])}]'

class BaseFunction(Value):
  def __init__(self, name):
    super().__init__()
    self.name = name or "<anonymous>"

  def set_context(self, context=None):
    if hasattr(self, "context") and self.context: return self
    return super().set_context(context)

  def generate_new_context(self):
    new_context = Context(self.name, self.context, self.pos_start)
    new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)
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
      dynamic = dynamics[i]
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
  def __init__(self, name, body_node, arg_names, defaults, dynamics, should_auto_return):
    super().__init__(name)
    self.body_node = body_node
    self.arg_names = arg_names
    self.defaults = defaults
    self.dynamics = dynamics
    self.should_auto_return = should_auto_return

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
    copy = Function(self.name, self.body_node, self.arg_names, self.defaults, self.dynamics, self.should_auto_return)
    copy.set_context(self.context)
    copy.set_pos(self.pos_start, self.pos_end)
    return copy

  def __repr__(self):
    return f"<function {self.name}>"

class BuiltInFunction(BaseFunction):
  def __init__(self, name):
    super().__init__(name)

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
    copy = BuiltInFunction(self.name)
    copy.set_context(self.context)
    copy.set_pos(self.pos_start, self.pos_end)
    return copy

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
  
  @args(['value'])
  def execute_println(self, exec_ctx):
    print(str(exec_ctx.symbol_table.get('value')))
    return RTResult().success(Number.null)
  
  @args(['value'])
  def execute_id(self, exec_ctx):
    val = exec_ctx.symbol_table.get('value')

    if isinstance(val, String):
      return RTResult().success(Number(id(str(val))))
    elif isinstance(val, Number):
      return RTResult().success(Number(id(int(str(val)))))
    elif isinstance(val, Bin):
      return RTResult().success(Number(id(bin(convert_forL(val)))))
    elif isinstance(val, Bytes):
      return RTResult().success(Number(id(to_bytes_forL(val))))
    elif isinstance(val, BaseFunction):
      return RTResult().success(Number(id(val)))
    else:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "The index needs be Number, String, Bytes or Bin type, not " + str(type(val)),
        exec_ctx
      ))

  @args(['value'])
  def execute_int(self, exec_ctx):
    val = str(exec_ctx.symbol_table.get('value'))

    return RTResult().success(Number(int(val)))
  
  @args(['value'])
  def execute_bit_to_int(self, exec_ctx):
    val = str(exec_ctx.symbol_table.get('value'))

    return RTResult().success(Number(int(val, 2)))

  @args(['value'])
  def execute_float(self, exec_ctx):
    val = str(exec_ctx.symbol_table.get('value'))

    return RTResult().success(Number(float(val)))

  @args(['value'])
  def execute_str(self, exec_ctx):
    val = exec_ctx.symbol_table.get('value')

    return RTResult().success(String(str(val)))

  @args(['value'])
  def execute_bin(self, exec_ctx):
    val = convert_forL(str(exec_ctx.symbol_table.get('value')))

    return RTResult().success(Bin(bin(val)))

  @args(['value'])
  def execute_from_bytes_int(self, exec_ctx):
    val = convert_forL(exec_ctx.symbol_table.get('value'))

    return RTResult().success(Number(int.from_bytes(to_bytes(val))))
  
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
  
  @args(['value'])
  def execute_input(self, exec_ctx):
    text = input(exec_ctx.symbol_table.get('value'))
    return RTResult().success(String(text))
  
  @args([])
  def execute_input_int(self, exec_ctx):
    while True:
      text = input()
      try:
        number = int(text)
        break
      except ValueError:
        print(f"'{text}' must be an integer. Try again!")
    return RTResult().success(Number(number))
  
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
    list_ = exec_ctx.symbol_table.get("value")

    if not isinstance(list_, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Argument must be list, string or char",
        exec_ctx
      ))

    return RTResult().success(Number(len(list_.elements)))
  
  @args(["fn"])
  def execute_run(self, exec_ctx):
    fn = exec_ctx.symbol_table.get("fn")

    if not isinstance(fn, String):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Second argument must be string",
        exec_ctx
      ))

    print("WARNING: run() is deprecated. Use 'IMPORT' instead")
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

  @args(["fn", "mode"], [None, String("r")])
  def execute_open(self, exec_ctx):
    sym = exec_ctx.symbol_table
    fake_pos = create_fake_pos("<built-in function open>")
    res = RTResult()

    fn = sym.get("fn")
    if not isinstance(fn, String):
      return res.failure(RTError(
        fake_pos, fake_pos,
        f"1st argument of function 'open' ('fn') must be String",
        exec_ctx
      ))
    fn = fn.value

    mode = sym.get("mode")
    if not isinstance(mode, String):
      return res.failure(RTError(
        fake_pos, fake_pos,
        f"2nd argument of function 'open' ('mode') must be String",
        exec_ctx
      ))
    mode = mode.value

    try:
      f = open(fn, mode)
    except (TypeError, OSError) as err:
      if isinstance(err, TypeError):
        return res.failure(RTError(
          fake_pos, fake_pos,
          f"Invalid file open mode: '{mode}'",
          exec_ctx
        ))
      elif isinstance(err, FileNotFoundError):
        return res.failure(RTError(
          fake_pos, fake_pos,
          f"Cannot find file '{fn}'",
          exec_ctx
        ))
      else:
        return res.failure(RTError(
          fake_pos, fake_pos,
          f"{err.args[-1]}",
          exec_ctx
        ))

    fd = f.fileno()
    files[fd] = f

    return res.success(Number(fd).set_pos(fake_pos, fake_pos).set_context(exec_ctx))
  
  @args(["fd", "bytes"])
  def execute_read(self, exec_ctx):
    sym = exec_ctx.symbol_table
    fake_pos = create_fake_pos("<built-in function read>")
    res = RTResult()

    fd = sym.get("fd")
    if not isinstance(fd, Number):
      return res.failure(RTError(
        fake_pos, fake_pos,
        f"1st argument of function 'read' ('fd') must be Number",
        exec_ctx
      ))
    fd = fd.value

    bts = sym.get("bytes")
    if not isinstance(bts, Number):
      return res.failure(RTError(
        fake_pos, fake_pos,
        f"2nd argument of function 'read' ('bytes') must be Number",
        exec_ctx
      ))
    bts = bts.value

    try:
      result = os.read(fd, bts).decode("utf-8")
    except OSError:
      return res.failure(RTError(
        fake_pos, fake_pos,
        f"Invalid file descriptor: {fd}",
        exec_ctx
      ))
    
    return res.success(String(result).set_pos(fake_pos, fake_pos).set_context(exec_ctx))

  @args(["fd", "bytes"])
  def execute_write(self, exec_ctx):
    sym = exec_ctx.symbol_table
    fake_pos = create_fake_pos("<built-in function write>")
    res = RTResult()

    fd = sym.get("fd")
    if not isinstance(fd, Number):
      return res.failure(RTError(
        fake_pos, fake_pos,
        f"1st argument of function 'write' ('fd') must be Number",
        exec_ctx
      ))
    fd = fd.value

    bts = sym.get("bytes")
    if not isinstance(bts, String):
      return res.failure(RTError(
        fake_pos, fake_pos,
        f"2nd argument of function 'write' ('bytes') must be String",
        exec_ctx
      ))
    bts = bts.value

    try:
      num = os.write(fd, bytes(bts, "utf-8"))
    except OSError:
      return res.failure(RTError(
        fake_pos, fake_pos,
        f"Invalid file descriptor: {fd}",
        exec_ctx
      ))
    
    return res.success(Number(num).set_pos(fake_pos, fake_pos).set_context(exec_ctx))

  @args(["fd"])
  def execute_close(self, exec_ctx):
    sym = exec_ctx.symbol_table
    fake_pos = create_fake_pos("<built-in function close>")
    res = RTResult()

    fd = sym.get("fd")
    if not isinstance(fd, Number):
      return res.failure(RTError(
        fake_pos, fake_pos,
        f"1st argument of function 'close' ('fd') must be Number",
        exec_ctx
      ))
    fd = fd.value
    std_desc = ["stdin", "stdout", "stderr"]

    if fd < 3:
      return res.failure(RTError(
        fake_pos, fake_pos,
        f"Cannot close {std_desc[fd]}",
        exec_ctx
      ))

    try:
      os.close(fd)
    except OSError:
      return res.failure(RTError(
        fake_pos, fake_pos,
        f"Invalid file descriptor '{fd}'",
        exec_ctx
      ))

    del files[fd]

    return res.success(Number.null)

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

BuiltInFunction.print           = BuiltInFunction("print")
BuiltInFunction.println         = BuiltInFunction("println")
BuiltInFunction.print_ret       = BuiltInFunction("print_ret")
BuiltInFunction.system          = BuiltInFunction("system")
BuiltInFunction.id              = BuiltInFunction("id")
BuiltInFunction.int             = BuiltInFunction("int")
BuiltInFunction.bit_to_int      = BuiltInFunction("bit_to_int")
BuiltInFunction.bin             = BuiltInFunction("bin")
BuiltInFunction.float           = BuiltInFunction("float")
BuiltInFunction.str             = BuiltInFunction("str")
BuiltInFunction.split           = BuiltInFunction("split")
BuiltInFunction.from_bytes_int  = BuiltInFunction("from_bytes_int")
BuiltInFunction.input           = BuiltInFunction("input")
BuiltInFunction.clear           = BuiltInFunction("clear")
BuiltInFunction.is_number       = BuiltInFunction("is_number")
BuiltInFunction.is_string       = BuiltInFunction("is_string")
BuiltInFunction.is_list         = BuiltInFunction("is_list")
BuiltInFunction.is_function     = BuiltInFunction("is_function")
BuiltInFunction.append          = BuiltInFunction("append")
BuiltInFunction.pop             = BuiltInFunction("pop")
BuiltInFunction.extend          = BuiltInFunction("extend")
BuiltInFunction.len		          = BuiltInFunction("len")
BuiltInFunction.run			        = BuiltInFunction("run")
BuiltInFunction.open            = BuiltInFunction("open")
BuiltInFunction.read            = BuiltInFunction("read")
BuiltInFunction.write           = BuiltInFunction("write")
BuiltInFunction.close           = BuiltInFunction("close")
BuiltInFunction.wait            = BuiltInFunction("wait")

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

#######################################
# CONTEXT
#######################################

class Context:
  def __init__(self, display_name, parent=None, parent_entry_pos=None):
    self.display_name = display_name
    self.parent = parent
    self.parent_entry_pos = parent_entry_pos
    self.symbol_table = None

class SymbolTable:
  def __init__(self, parent=None):
    self.symbols = {}
    self.structs = {}
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