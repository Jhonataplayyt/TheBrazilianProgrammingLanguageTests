from functools import wraps, lru_cache

def fast_memorize(func):
    cache: dict = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key: str = str(args) + str(kwargs)

        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    return wrapper

@fast_memorize
def for_loop(self, res, node, context, Number):
    elements = []

    start_value = res.register(self.visit(node.start_value_node, context))
    if res.should_return(): return res
    if res.should_yield(): yield res

    end_value = res.register(self.visit(node.end_value_node, context))
    if res.should_return(): return res
    if res.should_yield(): yield res

    if node.step_value_node:
        step_value = res.register(self.visit(node.step_value_node, context))
        if res.should_return(): return res
        if res.should_yield(): yield res
    else:
        step_value = Number(1)

    i = start_value.value

    if step_value.value >= 0:
        condition = lambda: i <= end_value.value
    else:
        condition = lambda: i >= end_value.value
    
    while condition():
        context.symbol_table.set(node.var_name_tok.value, Number(i))
        i += step_value.value

        value = res.register(self.visit(node.body_node, context))
        if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False: return res
        
        if res.should_yield() and res.loop_should_continue == False and res.loop_should_break == False: yield rea

        if res.loop_should_continue:
            continue

        if res.loop_should_break:
            break

        if res.loop_should_pass:
            pass

        elements.append(value)

    return elements

#@fast_memorize
def while_loop(self, res, node, context):
    elements = []
    
    print('nwentr')

    while True:
        condition = res.register(self.visit(node.condition_node, context))
        
        if res.should_return(): return res
        if res.should_yield(): yield res

        if not condition.is_true():
            break

        value = res.register(self.visit(node.body_node, context))
        if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False: return res
        
        if res.should_yield() and res.loop_should_continue == False and res.loop_should_break == False: yield res

        if res.loop_should_continue:
            continue

        if res.loop_should_break:
            break

        if res.loop_should_pass:
            pass

        elements.append(value)
    
    return elements

@fast_memorize
def for_in_loop(self, res, context, var_name, body, it):
    elements = []

    for it_res in it:
        elt = res.register(it_res)
        
        if res.should_return(): return res
        if res.should_yield(): yield res

        context.symbol_table.set(var_name, elt)

        elements.append(res.register(self.visit(body, context)))
        
        if res.should_return(): return res
        if res.should_yield(): yield res
    
    return elements

@fast_memorize
@lru_cache(maxsize=None)
def fast_memorize_for_loop(self, res, node, context, Number):
    return for_loop(self, res, node, context, Number)

#@fast_memorize
#@lru_cache(maxsize=None)
def fast_memorize_while_loop(self, res, node, context):
    return while_loop(self, res, node, context)

@fast_memorize
@lru_cache(maxsize=None)
def fast_memorize_for_in_loop(self, res, context, var_name, body, it):
    return for_in_loop(self, res, context, var_name, body, it)