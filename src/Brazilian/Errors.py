import string, os, math, time, sys, pickle, pydantic, importlib
from .More import RTResult, string_with_arrows
from Values import Iterator

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

  def notted(self):
    return None, self.illegal_operation()
  
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
    t = type(self)
    attr = f"inner_{verb}"
    if not hasattr(t, attr):
      return None, RTError(
        self.pos_start, self.pos_end,
        f"Object of type '{t.__name__}' has no property of name '{verb}'",
        self.context
      )
    return getattr(t, attr), None

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