a = open('./src/Brazilian/__init__.py', "r").read()

a = a.replace('@jit(nopython=True)', '')

with open('./src/Brazilian/__init__.py', 'w') as f:
    f.write(a)