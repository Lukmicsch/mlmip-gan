config = {

  'data_path': '/data/training/',

  # E.g. 001_orig.nii.gz and 001_masks.nii.gz
  'data_format': 'nii',
  'compression_file_format': 'gz',
  'img_suffix': '_orig',
  'mask_suffix': '_masks',

  'batch_size': 4,
  'channels': 1,
  'width_and_height': 256,
  'z_dim': 220,

  'adv_criterion': 'BCE',
  'recon_criterion': 'L1',
  'lambda_recon': 200,
  'n_epochs': 20,
  'display_step': 200,
  'lr': 0.0002,
  'device': 'cuda'
}