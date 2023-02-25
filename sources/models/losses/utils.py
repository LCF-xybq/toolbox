import functools

import torch.nn.functional as F


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    if reduction_enum == 1:
        return loss.mean()

    return loss.sum()


def mask_reduce_loss(loss, weight=None, reduction='mean', sample_wise=False):
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if weight is not specified or reduction is sum, just reduce the loss
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then compute mean over masked region
    elif reduction == 'mean':
        # expand weight from N1HW to NCHW
        if weight.size(1) == 1:
            weight = weight.expand_as(loss)
        # small value to prevent division by zero
        eps = 1e-12

        # perform sample-wise mean
        if sample_wise:
            weight = weight.sum(dim=[1, 2, 3], keepdim=True)  # NCHW to N111
            loss = (loss / (weight + eps)).sum() / weight.size(0)
        # perform pixel-wise mean
        else:
            loss = loss.sum() / (weight.sum() + eps)

    return loss


def masked_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                sample_wise=False,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = mask_reduce_loss(loss, weight, reduction, sample_wise)
        return loss

    return wrapper
