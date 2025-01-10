from .Values import *
from .Lexer import *
from .AST import *
from .Lang import *

def run(fn, text, context=None, entry_pos=None):
  # Generate tokens
  lexer = Lexer(fn, text)

  tokens, error = lexer.make_tokens()
  if error: return None, error
  
  # Generate AST
  parser = Parser(tokens)
  ast = parser.parse()
  if ast.error: return None, ast.error

  # Run program
  interpreter = Interpreter()
  context_was_none = context is None
  context = Context('<program>', context, entry_pos)
  if context_was_none:
    context.symbol_table = global_symbol_table
  else:
    context.symbol_table = context.parent.symbol_table
  result = interpreter.visit(ast.node, context)
  ret = result.func_return_value
  if context_was_none and ret:
    if not isinstance(ret, Number):
      return None, RTError(
        ret.pos_start, ret.pos_end,
        "Exit code must be Number",
        context
      )
    exit(ret.value)

  return result.value, result.error

def isLangExtension(x):
	if x.endswith(".br"):
		return x
	else:
		y = x.split('.')

		print("Extension Error: The extension needs be '.br', not '." + y[1] + "'")

		exit(1)