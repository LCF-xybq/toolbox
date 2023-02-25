import ast

filename = '../tests/configs/config_file.py'

with open(filename, encoding='utf-8') as f:
    # Setting encoding explicitly to resolve coding issue on windows
    content = f.read()
try:
    ast.parse(content)
except SyntaxError as e:
    raise SyntaxError('There are syntax errors in config '
                      f'file {filename}: {e}')

