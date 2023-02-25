import cv2
import math
import torch
import numpy as np
import os.path as osp

from cv.utils import is_str
from pathlib import Path
from torchvision.utils import make_grid

def is_filepath(x):
    return is_str(x) or isinstance(x, Path)

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to (min, max), image values will be normalized to [0, 1].

    For different tensor shapes, this function will have different behaviors:

        1. 4D mini-batch Tensor of shape (N x 3/1 x H x W):
            Use `make_grid` to stitch images in the batch dimension, and then
            convert it to numpy array.
        2. 3D Tensor of shape (3/1 x H x W) and 2D Tensor of shape (H x W):
            Directly change to numpy array.

    Note that the image channel in input tensors should be RGB order. This
    function will convert it to cv2 convention, i.e., (H x W x C) with BGR
    order.

    Args:
        tensor (Tensor | list[Tensor]): Input tensors.
        out_type (numpy type): Output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple): min and max values for clamp.

    Returns:
        (Tensor | list[Tensor]): 3D ndarray of shape (H x W x C) or 2D ndarray
        of shape (H x W).
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        # Squeeze two times so that:
        # 1. (1, 1, h, w) -> (h, w) or
        # 3. (1, 3, h, w) -> (3, h, w) or
        # 2. (n>1, 3/1, h, w) -> (n>1, 3/1, h, w)
        _tensor = _tensor.squeeze(0).squeeze(0)
        _tensor = _tensor.float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise ValueError('Only support 4D, 3D or 2D tensor. '
                             f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    result = result[0] if len(result) == 1 else result
    return result

def imwrite(img,
            file_path,
            params=None):
    """Write image to file.

    Note:
        In v1.4.1 and later, add `file_client_args` parameters.

    Warning:
        The parameter `auto_mkdir` will be deprecated in the future and every
        file clients will make directory automatically.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically. It will be deprecated.

    Returns:
        bool: Successful or not.
    """
    assert is_filepath(file_path)
    file_path = str(file_path)
    img_ext = osp.splitext(file_path)[-1]
    # Encode image according to image suffix.
    # For example, if image path is '/path/your/img.jpg', the encode
    # format is '.jpg'.
    flag, img_buff = cv2.imencode(img_ext, img, params)
    return flag
