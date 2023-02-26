import numpy as np

from ..registry import PIPELINES
from cv.image import imfrombytes, bgr2ycbcr, rgb2ycbcr, byte_stream_get


@PIPELINES.register_module()
class LoadImageFromFile:
    """Load image from file.

    Args:
        key (str): Keys in results to find corresponding path. Default: 'gt'.
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        convert_to (str | None): The color space of the output image. If None,
            no conversion is conducted. Default: None.
        save_original_img (bool): If True, maintain a copy of the image in
            `results` dict with name of `f'ori_{key}'`. Default: False.
        use_cache (bool): If True, load all images at once. Default: False.
        kwargs (dict): Args for file client, disk now.
    """

    def __init__(self,
                 key='gt',
                 flag='color',
                 channel_order='bgr',
                 convert_to=None,
                 save_original_img=False,
                 use_cache=False,
                 **kwargs):

        self.key = key
        self.flag = flag
        self.save_original_img = save_original_img
        self.channel_order = channel_order
        self.convert_to = convert_to
        self.kwargs = kwargs
        self.use_cache = use_cache
        self.cache = dict() if use_cache else None

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        filepath = str(results[f'{self.key}_path'])
        if self.use_cache:
            if filepath in self.cache:
                img = self.cache[filepath]
            else:
                img_bytes = byte_stream_get(filepath)
                img = imfrombytes(
                    img_bytes,
                    flag=self.flag,
                    channel_order=self.channel_order)  # HWC
                self.cache[filepath] = img
        else:
            img_bytes = byte_stream_get(filepath)
            img = imfrombytes(
                img_bytes,
                flag=self.flag,
                channel_order=self.channel_order)  # HWC

        if self.convert_to is not None:
            if self.channel_order == 'bgr' and self.convert_to.lower() == 'y':
                img = bgr2ycbcr(img, y_only=True)
            elif self.channel_order == 'rgb':
                img = rgb2ycbcr(img, y_only=True)
            else:
                raise ValueError('Currently support only "bgr2ycbcr" or '
                                 '"bgr2ycbcr".')
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

        results[self.key] = img
        results[f'{self.key}_path'] = filepath
        results[f'{self.key}_ori_shape'] = img.shape
        if self.save_original_img:
            results[f'ori_{self.key}'] = img.copy()

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(key={self.key}, flag={self.flag}, '
            f'save_original_img={self.save_original_img}, '
            f'channel_order={self.channel_order}, use_cache={self.use_cache})')
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFile_Color_Compensate:
    def __init__(self,
                 key='gt',
                 flag='color',
                 channel_order='bgr',
                 color_compensate=False,
                 convert_to=None,
                 save_original_img=False,
                 use_cache=False,
                 **kwargs):

        self.key = key
        self.flag = flag
        self.save_original_img = save_original_img
        self.channel_order = channel_order
        self.color_compensate = color_compensate
        self.convert_to = convert_to
        self.kwargs = kwargs
        self.use_cache = use_cache
        self.cache = dict() if use_cache else None

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        filepath = str(results[f'{self.key}_path'])
        if self.use_cache:
            if filepath in self.cache:
                img = self.cache[filepath]
            else:
                img_bytes = byte_stream_get(filepath)
                img = imfrombytes(
                    img_bytes,
                    flag=self.flag,
                    channel_order=self.channel_order)  # HWC
                self.cache[filepath] = img
        else:
            img_bytes = byte_stream_get(filepath)
            img = imfrombytes(
                img_bytes,
                flag=self.flag,
                channel_order=self.channel_order)  # HWC

        if self.convert_to is not None:
            if self.channel_order == 'bgr' and self.convert_to.lower() == 'y':
                img = bgr2ycbcr(img, y_only=True)
            elif self.channel_order == 'rgb':
                img = rgb2ycbcr(img, y_only=True)
            else:
                raise ValueError('Currently support only "bgr2ycbcr" or '
                                 '"bgr2ycbcr".')
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

        if self.color_compensate:
            r, g, b = img[:, :, 0:1], img[:, :, 1:2], img[:, :, 2:3]
            mean_r = np.mean(r)
            mean_g = np.mean(g)
            mean_b = np.mean(b)
            lst = sorted([mean_r, mean_g, mean_b])
            ms = lst[1] - lst[0]
            lm = lst[2] - lst[1]

            if lst.index(mean_r) == 0:
                img[:, :, 0:1] = img[:, :, 0:1] + ms
            elif lst.index(mean_r) == 2:
                img[:, :, 0:1] = img[:, :, 0:1] - lm

            if lst.index(mean_g) == 0:
                img[:, :, 1:2] = img[:, :, 1:2] + ms
            elif lst.index(mean_g) == 2:
                img[:, :, 1:2] = img[:, :, 1:2] - lm

            if lst.index(mean_b) == 0:
                img[:, :, 2:3] = img[:, :, 2:3] + ms
            elif lst.index(mean_b) == 2:
                img[:, :, 2:3] = img[:, :, 2:3] - lm

        results[self.key] = img
        results[f'{self.key}_path'] = filepath
        results[f'{self.key}_ori_shape'] = img.shape
        if self.save_original_img:
            results[f'ori_{self.key}'] = img.copy()

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(key={self.key}, flag={self.flag}, '
            f'save_original_img={self.save_original_img}, '
            f'channel_order={self.channel_order}, use_cache={self.use_cache})')
        return repr_str