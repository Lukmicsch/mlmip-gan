config = {

  'data_path': '/data/training/',

  'algorithm': 'dcgan256',

  # E.g. 001_orig.nii.gz and 001_masks.nii.gz
  'data_format': 'nii',
  'compression_file_format': 'gz',
  'img_suffix': '_orig',
  'mask_suffix': '_masks',

  'batch_size': 1,
  'channels': 1,
  'width_and_height': 256,
  'width_and_height_to_model': 256,
  'z_dim': 220,
  'noise_dim': 100,
  'images_per_batch_iter': 150,

  'save_model': True,

  'criterion': 'BCE',
  'beta_1': 0.5,
  'beta_2': 0.999,
    'n_epochs': 500,
  'display_step': 109,
  'lr': 0.0002,
  'device': 'cuda'
}
