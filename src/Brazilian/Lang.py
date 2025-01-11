import string, os, math, time, sys, pickle, pydantic, importlib
from dataclasses import dataclass
from enum import Enum, auto
from typing import *
from .Nodes import *
from .Errors import RTError
from .AST import TryError
from Tokens import TokenType
from Constants import IMPORT_PATHS, IMPORT_PATH_NAME
from .Values import RTResult, Bin, Bytes, Number, String, Function, SymbolTable, StructInstance, Context
from .Brazilian import run

class Interpreter:
  def visit(self, node, context):
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

    value = value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
    return res.success(value)

  def visit_VarAssignNode(self, node, context):
    res = RTResult()
    var_name = node.var_name_tok.value
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
    if res.should_return(): return res

    end_value = res.register(self.visit(node.end_value_node, context))
    if res.should_return(): return res

    if node.step_value_node:
      step_value = res.register(self.visit(node.step_value_node, context))
      if res.should_return(): return res
    else:
      step_value = Number(1)

    i = start_value.value

    if step_value.value >= 0:
      condition = lambda: i < end_value.value
    else:
      condition = lambda: i > end_value.value
    
    while condition():
      context.symbol_table.set(node.var_name_tok.value, Number(i))
      i += step_value.value

      value = res.register(self.visit(node.body_node, context))
      if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False: return res
      
      if res.loop_should_continue:
        continue
      
      if res.loop_should_break:
        break

      if res.loop_should_pass:
        pass

      elements.append(value)

    return res.success(
      Number.null if node.should_return_null else
      List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_WhileNode(self, node, context):
    res = RTResult()
    elements = []

    while True:
      condition = res.register(self.visit(node.condition_node, context))
      if res.should_return(): return res

      if not condition.is_true():
        break

      value = res.register(self.visit(node.body_node, context))
      if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False: return res

      if res.loop_should_continue:
        continue
      
      if res.loop_should_break:
        break
        
      if res.loop_should_pass:
        pass

      elements.append(value)

    return res.success(
      Number.null if node.should_return_null else
      List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
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
  
  def visit_MethodDefNode(self, node, context):
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

    value_to_call = res.register(self.visit(node.node_to_call, context))
    if res.should_return(): return res
    value_to_call = value_to_call.copy().set_pos(node.pos_start, node.pos_end)

    for arg_node in node.arg_nodes:
      args.append(res.register(self.visit(arg_node, context)))
      if res.should_return(): return res

    return_value = res.register(value_to_call.execute(args))
    if res.should_return(): return res
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

    for path in IMPORT_PATHS:
      try:
        filepath = os.path.join(path, filename.value)
        with open(filepath, "r") as f:
          code = f.read()
          beginning = "/" if filepath.startswith("/") else ""
          split = filepath.split("/")
          split = beginning + "/".join(split[:-1]), split[-1]
          os.chdir(split[0])
          filename = split[1]
          break
      except FileNotFoundError:
        continue
    
    if code is None:
      return res.failure(RTError(
        node.string_node.pos_start.copy(), node.string_node.pos_end.copy(),
        f"Can't find file '{filepath}' in '{IMPORT_PATH_NAME}'. Please add the directory your file is into that file",
        context
      ))
    
    _, error = run(filename, code, context, node.pos_start.copy())
    if error: return res.failure(error)

    return res.success(Number.null)
  
  def visit_DoNode(self, node, context):
    res = RTResult()
    new_context = Context("<do statement>", context, node.pos_start.copy())
    new_context.symbol_table = SymbolTable(context.symbol_table)
    res.register(self.visit(node.statements, new_context))

    return_value = res.func_return_value
    if res.should_return() and return_value is None: return res

    return_value = return_value or Number.null

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
      if res.should_return(): return res

      context.symbol_table.set(var_name, elt)

      elements.append(res.register(self.visit(body, context)))
      if res.should_return(): return res
    
    if should_return_null: return res.success(Number.null)
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

      print(f"[DEBUG] {object.__repr__(case)}")

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

    verb = node.verb.value

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

  def visit_StructCreationNode(self, node, ctx):
        res = RTResult()
        struct = ctx.symbol_table.structs[node.name]

        return res.success(StructInstance(node.name, {field: Number.null for field in struct})
                           .set_pos(node.pos_start, node.pos_end)
                           .set_context(ctx))