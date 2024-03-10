import os, sys, random, time, glob, math

import numpy as np
import torch
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

sys.path.append("./utils")
from data_aug import *


def collate_fn(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    else:
        return jnp.asarray(batch)


class LPR_Data(torch.utils.data.Dataset):
    def __init__(self, key, data_dir, img_size=(64, 128), aug=False):
        self.img_size = img_size
        self.imgs = glob.glob(os.path.join(data_dir, '*.jpg'))
        self.key = key

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = load_image(img_path)
        img = augment_image(img, self.key)
        img = resize_image(img)
        self.key = jax.random.split(self.key)[0]
        return img


if __name__ == "__main__":
    jax.config.update('jax_platform_name', 'cpu')
    key = jax.random.PRNGKey(0)

    data_dir = "data/val"
    dataset = LPR_Data(key, data_dir)

    start_time = time.process_time()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
    )
    print(f'elapsed time: {(time.process_time() - start_time) * 1000} ms')

    # using plt show some samples in one figure
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(dataloader.dataset[i], cmap='gray')
        ax.axis('off')

    plt.show()

    # for batch in dataloader:
    #     print(batch.shape)

    print(f'elapsed time: {(time.process_time() - start_time) * 1000} ms')
