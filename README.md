# LPR_Jax

This repository is a JAX implementations of lightweight license plate recognition (LPR) models.

## Data Preparation

The labeled data is required to train the model. The data should be organized as follows:

```dir
- data
  - labels.names
  - train
    - {license_plate_number}_{image_number}.jpg
    - {license_plate_number}_{image_number}.npy
    - ...
  - val
    - ...
```

`license_plate_number` is the license plate number and make sure that the number is formatted like `12가1234`, `서울12가1234` and prepare a dict to parse the every character of the license plate number to the integer. The dict should be saved as `labels.names` file. `image_number` is the number of the image and it is used to distinguish the same license plate number. The `.npy` file is the mask label of each character of the license plate number. It does not need pixel-wise label, just need character-wise label. The mask label should be a 3D array and the shape should be `(height, width, character_number)`. The `character_number` is the number of the characters of the license plate number. The rectagle mask label should be a binary mask and the background should be 0 and the character should be 1.

## Benchmark

|  Model  |  Size  | Accuracy | Speed (ms) |
| ------- | ------ | -------- | ----------:|
| tinyLPR | 74 KB  |  -       | 0.13 ms    |
