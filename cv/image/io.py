import re
import cv2
import warnings
import numpy as np
import os.path as osp

from typing import Union
from pathlib import Path
from cv2 import (IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_IGNORE_ORIENTATION,
                 IMREAD_UNCHANGED)
from cv.utils import is_filepath, is_str, mkdir_or_exist

try:
    from turbojpeg import TJCS_RGB, TJPF_BGR, TJPF_GRAY, TurboJPEG
except ImportError:
    TJCS_RGB = TJPF_GRAY = TJPF_BGR = TurboJPEG = None

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None

try:
    import tifffile
except ImportError:
    tifffile = None

jpeg = None

imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED,
    'color_ignore_orientation': IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    'grayscale_ignore_orientation':
    IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE
}

imread_backend = 'cv2'


def _jpegflag(flag='color', channel_order='bgr'):
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'color':
        if channel_order == 'bgr':
            return TJPF_BGR
        elif channel_order == 'rgb':
            return TJCS_RGB
    elif flag == 'grayscale':
        return TJPF_GRAY
    else:
        raise ValueError('flag must be "color" or "grayscale"')

def _format_path(path: str) -> str:
    return re.sub(r'\\+', '/', path)

def byte_stream_get(filepath: Union[str, Path]) -> bytes:
    """reads the file as a byte stream
        hard disk only!
    """
    filepath = _format_path(filepath)
    with open(filepath, 'rb') as f:
        value_buf = f.read()
    return value_buf

def byte_stream_put(obj: bytes, filepath: Union[str, Path]) -> None:
    """reads the file as a byte stream
        hard disk only!
    """
    mkdir_or_exist(filepath)
    with open(filepath, 'wb') as f:
        f.write(obj)

def imread(img_or_path,
           flag='color',
           channel_order='bgr'):
    """Read an image.

    Note:
        In v1.4.1 and later, add `file_client_args` parameters.

    Args:
        img_or_path (ndarray or str or Path): Either a numpy array or str or
            pathlib.Path. If it is a numpy array (loaded image), then
            it will be returned as is.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale`, `unchanged`,
            `color_ignore_orientation` and `grayscale_ignore_orientation`.
            By default, `cv2` and `pillow` backend would rotate the image
            according to its EXIF info unless called with `unchanged` or
            `*_ignore_orientation` flags. `turbojpeg` and `tifffile` backend
            always ignore image's EXIF info regardless of the flag.
            The `turbojpeg` backend only supports `color` and `grayscale`.
        channel_order (str): Order of channel, candidates are `bgr` and `rgb`.

    Returns:
        ndarray: Loaded image array.
    """

    if isinstance(img_or_path, Path):
        img_or_path = str(img_or_path)

    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif is_str(img_or_path):
        img_bytes = byte_stream_get(img_or_path)
        return imfrombytes(img_bytes, flag, channel_order)
    else:
        raise TypeError('"img" must be a numpy array or a str or '
                        'a pathlib.Path object')


def imfrombytes(content, flag='color', channel_order='bgr'):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Same as :func:`imread`.
        channel_order (str): The channel order of the output, candidates
            are 'bgr' and 'rgb'. Default to 'bgr'.

    Returns:
        ndarray: Loaded image array.

    """

    img_np = np.frombuffer(content, np.uint8)
    flag = imread_flags[flag] if is_str(flag) else flag
    img = cv2.imdecode(img_np, flag)
    if flag == IMREAD_COLOR and channel_order == 'rgb':
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    return img


def imwrite(img,
            file_path,
            params=None,
            auto_mkdir=None):
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
    if auto_mkdir is not None:
        warnings.warn(
            'The parameter `auto_mkdir` will be deprecated in the future and '
            'every file clients will make directory automatically.')
    img_ext = osp.splitext(file_path)[-1]
    # Encode image according to image suffix.
    # For example, if image path is '/path/your/img.jpg', the encode
    # format is '.jpg'.
    flag, img_buff = cv2.imencode(img_ext, img, params)
    byte_stream_put(img_buff.tobytes(), file_path)
    return flag
