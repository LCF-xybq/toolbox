model = dict(
    type='MSDC',
    body=dict(
        type='DualColorED',
        num_clocks=1,
    ),
    mse_loss=dict(type='MSELoss', loss_weight=1.0),
    ssim_loss=dict(type='SSIMLoss', loss_weight=1.0),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={'34': 1.0},
        vgg_type='vgg19',
        perceptual_weight=0.1,
        style_weight=0,
        norm_img=False),
)