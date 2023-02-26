import copy
import os.path as osp
from collections import defaultdict
from pathlib import Path

from cv.utils import scandir
from .base_dataset import BaseDataset
from .registry import DATASETS


IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')

@DATASETS.register_module()
class PairedImageDataset(BaseDataset):
    '''
    data_root
        ├── lq
        │   ├── 0001.IMG_EXTENSIONS
        │   ├── 0002.IMG_EXTENSIONS
        ├── gt
        │   ├── 0001.IMG_EXTENSIONS
        │   ├── 0002.IMG_EXTENSIONS
    '''
    def __init__(self,
                 root,
                 prefix,
                 pipeline,
                 test_mode=False):
        super().__init__(pipeline, test_mode)
        self.root = str(root)
        assert isinstance(prefix, dict), f'prefix must be dict, but got {type(prefix)}.'
        self.prefix = prefix
        self.data_infos = self.get_paired_images()

    @staticmethod
    def scan_folder(path):
        """Obtain image path list (including sub-folders) from a given folder.

        Args:
            path (str | :obj:`Path`): Folder path.

        Returns:
            list[str]: image list obtained form given folder.
        """

        if isinstance(path, (str, Path)):
            path = str(path)
        else:
            raise TypeError("'path' must be a str or a Path object, "
                            f'but received {type(path)}.')

        images = list(scandir(path, suffix=IMG_EXTENSIONS))
        images = [osp.join(path, v) for v in images]
        assert images, f'{path} has no valid image file.'
        return images

    def get_paired_images(self):
        data_infos = []
        assert ('img' and 'gt' in list(self.prefix.keys())) \
            and len(self.prefix.keys()) == 2
        input_path = self.prefix['img']
        input_path = osp.join(self.root, input_path)
        gt_path = self.prefix['gt']
        gt_path = osp.join(self.root, gt_path)
        input_imgs = self.scan_folder(input_path)
        gt_imgs = self.scan_folder(gt_path)
        assert len(input_imgs) == len(gt_imgs), (
            f'gt and input datasets have different number of images: '
            f'{len(input_imgs)}, {len(gt_imgs)}.')
        for img in input_imgs:
            basename = osp.basename(img)
            gt_image = osp.join(self.root, self.prefix['gt'], basename)
            assert gt_image in gt_imgs
            data_infos.append(dict(input_path=img, gt_path=gt_image))
        return data_infos


    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def evaluate(self, results, logger=None):
        """Evaluate with different metrics.

        Args:
            results (list[tuple]): The output of forward_test() of the model.

        Return:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        results = [res['eval_result'] for res in results]  # a list of dict
        eval_result = defaultdict(list)  # a dict of list

        for res in results:
            for metric, val in res.items():
                eval_result[metric].append(val)
        for metric, val_list in eval_result.items():
            assert len(val_list) == len(self), (
                f'Length of evaluation result of {metric} is {len(val_list)}, '
                f'should be {len(self)}')

        # average the results
        eval_result = {
            metric: sum(values) / len(self)
            for metric, values in eval_result.items()
        }

        return eval_result
