import torch
from cv.image import imwrite, tensor2imgs


if __name__ == '__main__':
    import sys
    import importlib
    sys.path.insert(0, r'D:\Program_self\toolbox')
    mod = importlib.import_module(r'D:\Program_self\toolbox\cv')
    image = torch.randn(1, 3, 256, 256)
    save_path = r'D:\Program_self\toolbox\tools\work_dirs\test_config\x.png'
    imwrite(tensor2imgs(image), save_path)
    sys.path.pop(0)
