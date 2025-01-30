def get_fn_name(func):
    # Obtém o código-fonte da função como uma string
    code = inspect.getsource(func)
    # Expressão regular para encontrar o nome da função
    function_regex = r"def\s+(\w+)\s*\("
    match = re.search(function_regex, code)
    
    if match:
        return match.group(1)
    else:
        print("Função não encontrada.", file=sys.stderr)