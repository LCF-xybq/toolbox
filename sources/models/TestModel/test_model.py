import numbers
import os.path as osp

from ..base import BaseModel
from ..registry import MODELS
from ..builder import build_loss, build_component
from sources.core import mse, psnr, ssim
from cv.image import imwrite, tensor2imgs

@MODELS.register_module()
class Test(BaseModel):
    def __init__(self,
                 body,
                 mse_loss,
                 perc_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
                 ):
        super(Test, self).__init__()
        self.allowed_metrics = {'MSE': mse, 'PSNR': psnr, 'SSIM': ssim}
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.mse_loss = build_loss(mse_loss) if mse_loss else None
        self.perc_loss = build_loss(perc_loss) if perc_loss else None
        self.body = build_component(body)
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        self.body.init_weights(pretrained=pretrained)

    def forward(self, input, gt=None, test_mode=False, **kwargs):
        if test_mode:
            return self.forward_test(input, gt, **kwargs)

        return self.forward_train(input, gt)

    def forward_train(self, input, gt):
        losses = dict()
        out = self.body(input)

        if self.mse_loss:
            loss_mse = self.mse_loss(out, gt)
            losses['loss_mse'] = loss_mse
        if self.perc_loss:
            loss_perc, _ = self.perc_loss(out, gt)
            losses['loss_perc'] = loss_perc

        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(input=input.cpu(), gt=gt.cpu(), output=out.cpu()))

        return outputs

    def evaluate(self, output, gt):
        crop_border = self.test_cfg.crop_border
        output = tensor2imgs(output)
        gt = tensor2imgs(gt)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = self.allowed_metrics[metric](output, gt,
                                                               crop_border)
        return eval_result

    def forward_test(self, input, gt=None, meta=None,
                     save_image=False, save_path=None, iteration=None):
        output = self.body(input)
        if gt is not None and self.test_cfg is not None:
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(input=input.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            input_path = meta[0]['input_path']
            folder_name = osp.splitext(osp.basename(input_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            imwrite(tensor2imgs(output), save_path)

        return results

    def train_step(self, data_batch, optimizer):
        """Train step.input

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs.update({'log_vars': log_vars})
        return outputs

    def val_step(self, data_batch, **kwargs):
        """Validation step.

        Args:
            data_batch (dict): A batch of data.
            kwargs (dict): Other arguments for ``val_step``.

        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Input image.

        Returns:
            Tensor: Output image.
        """
        out = self.body(img)

        return out