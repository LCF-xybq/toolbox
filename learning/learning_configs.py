import os
import ast
import sys
import types
import importlib


filename = '../tests/configs/config_file.py'

def func1(pth):
    with open(pth, encoding='utf-8') as f:
        # Setting encoding explicitly to resolve coding issue on windows
        content = f.read()
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise SyntaxError('There are syntax errors in config '
                        f'file {pth}: {e}')


def func2():
    abs_pth = os.path.abspath(filename)
    dir_pth = os.path.dirname(abs_pth)
    sys.path.insert(0, dir_pth)
    func1(abs_pth)                        
    module_name, _ = os.path.splitext(os.path.basename(abs_pth))
    mod = importlib.import_module(module_name)
    sys.path.pop(0)
    cfg_dict = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
        and not isinstance(value, types.ModuleType)
        and not isinstance(value, types.FunctionType)
    }
    print(cfg_dict)


if __name__ == "__main__":
    func2()