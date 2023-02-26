import numpy as np

from ..registry import PIPELINES


@PIPELINES.register_module()
class PairedRandomCrop:
    """Paried random crop.

    It crops a pair of input and gt images with corresponding locations.
    It also supports accepting input list and gt list.
    Required keys are "input", and "gt",
    added or modified keys are "input" and "gt".

    Args:
        gt_patch_size (int): cropped gt patch size.
    """

    def __init__(self, gt_patch_size):
        self.gt_patch_size = gt_patch_size

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        input_patch_size = self.gt_patch_size

        input_is_list = isinstance(results['input'], list)
        if not input_is_list:
            results['input'] = [results['input']]
        gt_is_list = isinstance(results['gt'], list)
        if not gt_is_list:
            results['gt'] = [results['gt']]

        h_input, w_input, _ = results['input'][0].shape
        h_gt, w_gt, _ = results['gt'][0].shape

        if h_gt != h_input or w_gt != w_input:
            raise ValueError(
                f'mismatches. GT ({h_gt}, {w_gt})')
        if h_input < input_patch_size or w_input < input_patch_size:
            raise ValueError(
                f'input ({h_input}, {w_input}) is smaller than patch size '
                f'({input_patch_size}, {input_patch_size}). Please check '
                f'{results["input_path"][0]} and {results["gt_path"][0]}.')

        # randomly choose top and left coordinates for input patch
        top = np.random.randint(h_input - input_patch_size + 1)
        left = np.random.randint(w_input - input_patch_size + 1)
        # crop input patch
        results['input'] = [
            v[top:top + input_patch_size, left:left + input_patch_size, ...]
            for v in results['input']
        ]
        # crop corresponding gt patch
        top_gt, left_gt = int(top), int(left)
        results['gt'] = [
            v[top_gt:top_gt + self.gt_patch_size,
              left_gt:left_gt + self.gt_patch_size, ...] for v in results['gt']
        ]

        if not input_is_list:
            results['input'] = results['input'][0]
        if not gt_is_list:
            results['gt'] = results['gt'][0]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(gt_patch_size={self.gt_patch_size})'
        return repr_str