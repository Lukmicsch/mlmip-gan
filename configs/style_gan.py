config = {

  'data_path': '/data/training/',

  'algorithm': 'style_gan',

  # E.g. 001_orig.nii.gz and 001_masks.nii.gz
  'data_format': 'nii',
  'compression_file_format': 'gz',
  'img_suffix': '_orig',
  'mask_suffix': '_masks',

  'batch_size': 1,
  'channels': 1,
  'width_and_height': 256,
  'width_and_height_to_model': 64,
  'z_dim': 220,

  'save_model': True,

  'beta_1': 0.5,
  'beta_2': 0.999,

  'criterion': 'BCE',
  'noise_dim': 128,
  'map_hidden_dim': 1024,
  'w_dim': 496,
  'in_chan': 512,
  'out_chan': 1,
  'kernel_size': 3,
  'hidden_chan': 256,
  'truncation': 0.7,
  'alpha': 3,
  'n_epochs': 300,
  'display_step': 500,
  'lr': 0.0002,
  'device': 'cuda'
}
