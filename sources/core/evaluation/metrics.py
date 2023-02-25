import cv2
import numpy as np

def _convert_input_type_range(img):
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, '
                        f'but got {img_type}')
    return img

def _convert_output_type_range(img, dst_type):
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, '
                        f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)

def bgr2ycbcr(img, y_only=False):
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                  [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img

def reorder_image(img, input_order='HWC'):
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

def mse(alpha, trimap, pred_alpha):
    if alpha.ndim != 2 or trimap.ndim != 2 or pred_alpha.ndim != 2:
        raise ValueError(
            'input alpha, trimap and pred_alpha should has two dimensions, '
            f'alpha {alpha.shape}, please check their shape: '
            f'trimap {trimap.shape}, pred_alpha {pred_alpha.shape}')
    assert (pred_alpha[trimap == 0] == 0).all()
    assert (pred_alpha[trimap == 255] == 255).all()
    alpha = alpha.astype(np.float64) / 255
    pred_alpha = pred_alpha.astype(np.float64) / 255
    weight_sum = (trimap == 128).sum()
    if weight_sum != 0:
        mse_result = ((pred_alpha - alpha)**2).sum() / weight_sum
    else:
        mse_result = 0
    return mse_result

def psnr(img1, img2, crop_border=0, input_order='HWC', convert_to=None):
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
    if isinstance(convert_to, str) and convert_to.lower() == 'y':
        img1 = bgr2ycbcr(img1 / 255., y_only=True) * 255.
        img2 = bgr2ycbcr(img2 / 255., y_only=True) * 255.
    elif convert_to is not None:
        raise ValueError('Wrong color model. Supported values are '
                         '"Y" and None.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    mse_value = np.mean((img1 - img2)**2)
    if mse_value == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse_value))

def _ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def ssim(img1, img2, crop_border=0, input_order='HWC', convert_to=None):
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if isinstance(convert_to, str) and convert_to.lower() == 'y':
        img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
        img1 = bgr2ycbcr(img1 / 255., y_only=True) * 255.
        img2 = bgr2ycbcr(img2 / 255., y_only=True) * 255.
        img1 = np.expand_dims(img1, axis=2)
        img2 = np.expand_dims(img2, axis=2)
    elif convert_to is not None:
        raise ValueError('Wrong color model. Supported values are '
                         '"Y" and None')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()
