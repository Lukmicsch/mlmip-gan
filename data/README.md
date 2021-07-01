## Data Module

### Import
```python
from data.data_manager import DataManager
```

### Dependencies
```python
from torch.utils.data import DataLoader
from utils.python_utils import load_config
```

### Usage
###### General
```python
config = load_config('configs.my_config')
data_manager = DataManager(config)
frac_test = config['frac_test']

# Get all cases
full_cases = data_manager.get_full_cases()

# Or train, test
train_cases, test_cases = data_manager.get_train_test_split_cases()

# Continuing with all cases
full_dataset = data_manager.get_dataset(full_cases)

full_dataloader = DataLoader(
    full_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=0,
)

for images, masks in full_dataloader:
    image_slices, indexing = data_manager.prepare_image_batch(images)
    mask_slices = data_manager.prepare_mask_batch(masks, indexing)

    # Now you do you
```

