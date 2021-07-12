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
  'images_per_batch_iter': 64,
  'flip_labels': True,

  'save_model': True,
  'save_frequency': 20,
  'model_path': 'data/model/',

  'criterion': 'BCE',
  'beta_1': 0.5,
  'beta_2': 0.999,
  'n_epochs': 1000,
  'display_step': 327,
  'lr': 0.0001,
  'device': 'cuda'
}
