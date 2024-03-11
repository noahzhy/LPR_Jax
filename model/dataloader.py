import os, sys, random, time, glob, math

from torch.utils.data import Dataset, DataLoader
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import matplotlib.pyplot as plt

sys.path.append("./utils")
from data_aug import *


def collate_fn(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        return type(batch[0])(collate_fn(samples) for samples in zip(*batch))
    else:
        return jnp.asarray(batch)


class LPR_Data(Dataset):
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
        # 50 % chance to resize the image with or without keeping aspect ratio
        if jax.random.bernoulli(self.key, 0.5):
            img = resize_image(img)
            # mask = mask.resize((64, 128))
        else:
            img = resize_image_keep_aspect_ratio(img)
            # mask = mask.resize((64, 128))

        self.key = jax.random.split(self.key)[0]
        return img


# show data augmentation via matplotlib
def show_augment_image(samples=8):
    jax.config.update('jax_platform_name', 'cpu')
    key = jax.random.PRNGKey(random.randint(0, 1000))
    data_dir = "data/val"
    dataset = LPR_Data(key, data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=samples,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
    )

    start_time = time.process_time()
    for batch in dataloader:
        print(batch.shape)
    print(f'elapsed time: {(time.process_time() - start_time) * 1000} ms')

    row = int(math.sqrt(samples) - 0.5)
    col = samples // row
    fig, axes = plt.subplots(row, col, figsize=(10, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(dataloader.dataset[i], cmap='gray')
        ax.axis('off')

    plt.show()


if __name__ == "__main__":
    jax.config.update('jax_platform_name', 'cpu')
    key = jax.random.PRNGKey(0)

    show_augment_image()
