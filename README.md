## MLMIP - GAN's

### General
This Repository holds two DCGAN implementation's. One for square images of size 28x28 and one for images 256x256. The Data pipeline is 
written s.t. 3d scans can easily be processed with corresponding masks.

Credits:
- The small dcgan was created after partaking in deeplearning.ai's GAN spezialisation
- The Bigger GAN was designed according to https://github.com/t0nberryking/DCGAN256, which is written in keras and itself is based on https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9. 

### Usage
```bash
pip install -r requirements.txt
```

###### Via Notebook
```python
# Indicate which config and therfore algorithm to use
config_name = 'dcgan256'
# Indicate which GPU to use (comment this if non available)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

###### Via Shell
```bash
python start.py --train --config dcgan256
```

### Config Options
- data_path: where to find the medical scans
- algorithm: 'dcgan' or 'dcgan256'
- 'data_format': e.g. 'nii'
- 'compression_file_format': e.g. 'gz'
- 'img_suffix': e.g. '_orig'
- 'mask_suffix': e.g. '_masks'
- 'batch_size': this referes to a whole 3d scan (s.t. 1 might be 220 slices)
- 'channels': color channels of the scans
- 'width_and_height': the actual width and height of the slices
- 'width_and_height_to_model': the width and height the model gets
- 'z_dim': how many slices per 3d scan
- 'noise_dim': the noise given to the generator,
- 'images_per_batch_iter': how many slices of z_dim many
- 'flip_labels': Set to True, can stabalize learning

- 'save_model': True
- 'save_frequency': 20
- 'model_path': 'data/model/'

- 'criterion': 'BCE' 
- 'beta_1': hyperparameter for optimizer e.g. 0.5
- 'beta_2': e.g. 0.999
- 'n_epochs': e.g. 1000
- 'display_step': e.g. 327
- 'lr': e.g. 0.0001
- 'device': 'cuda' or 'cpu'
