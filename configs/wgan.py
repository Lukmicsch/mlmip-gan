config = {

  'data_path': '/data/training/',
    
  'algorithm': 'wgan',

  # E.g. 001_orig.nii.gz and 001_masks.nii.gz
  'data_format': 'nii',
  'compression_file_format': 'gz',
  'img_suffix': '_orig',
  'mask_suffix': '_masks',
    
  'use_transform': True,

  'batch_size': 1,
  'channels': 1,
  'width_and_height': 256,
  'z_dim': 220,

  'save_model': True,

  'c_lambda': 10,
  'crit_repeats': 5,
    
    
  'beta_1': 0.5,
  'beta_2': 0.999,
  'n_epochs': 100,
  'display_step': 50,
  'lr': 0.0002,
  'device': 'cuda'
}