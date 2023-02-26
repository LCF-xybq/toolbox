import pkgutil
import importlib

def load_ext(name, funcs):
    ext = importlib.import_module('cv.' + name)
    for fun in funcs:
        assert hasattr(ext, fun), f'{fun} miss in module {name}'
    return ext


def check_ops_exist() -> bool:
    ext_loader = pkgutil.find_loader('cv._ext')
    return ext_loader is not None
