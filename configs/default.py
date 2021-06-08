config = {
  'lr': 0.001,

    '3d': False,

    'use_transform': True,

    'frac_test': 0.2,
    'data_path': '/data/training/',

    # E.g. 001_orig.nii.gz and 001_masks.nii.gz
    'data_format': 'nii',
    'compression_file_format': 'gz',
    'img_suffix': '_orig',
    'mask_suffix': '_masks',

    'batch_size': 2,
    'channels': 1,
    'width_and_height': 256,
    'z_dim': 220
}