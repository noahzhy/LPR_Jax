import os, sys, random, time, glob, math

from torch.utils.data import Dataset, DataLoader
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import matplotlib.pyplot as plt

sys.path.append("./utils")
from data_aug import *
from gen_label import gen_mask


# jit with static argnums
pad_mask_fn = jax.jit(pad_mask, static_argnums=1)
resize_fn = jax.jit(resize_image, static_argnums=(1, 2))
resize_ratio_fn = jax.jit(resize_image_keep_aspect_ratio, static_argnums=(1, 2))
insert0align2right_fn = jax.jit(insert0align2right, static_argnums=1)


def collate_fn(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        return type(batch[0])(collate_fn(samples) for samples in zip(*batch))
    else:
        return jnp.asarray(batch)


class LPR_Data(Dataset):
    def __init__(self, key, data_dir, time_step=15, img_size=(64, 128), aug=True, **kwargs):
        self.key = key
        self.time_step = time_step
        self.img_size = (img_size[0], img_size[1])
        self.imgs = glob.glob(os.path.join(data_dir, '*.jpg'))
        # # keep 128 only
        self.imgs = self.imgs[:128]
        self.aug = aug

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        _mask, label = gen_mask(img_path)
        img = load_image(img_path)

        if self.aug and jax.random.bernoulli(self.key, 0.5):
            img = augment_image(img, self.key)
            img = resize_fn(img, self.img_size)
            mask = resize_fn(_mask, self.img_size, method='nearest')
        else:
            img = to_grayscale(img)
            img = resize_ratio_fn(img, self.img_size)
            mask = resize_ratio_fn(_mask, self.img_size, method='nearest')

        mask = pad_mask_fn(mask, self.time_step)
        label = insert0align2right_fn(label, self.time_step)

        self.key = jax.random.split(self.key)[0]
        return img, mask, label


# show data augmentation via matplotlib
def show_augment_image(samples=8):
    key = jax.random.PRNGKey(0)
    data_dir = "/Users/haoyu/Documents/datasets/lpr/val"
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
        img, mask, label = batch
        print(img.shape, mask.shape, label.shape)
        break

    print(f'elapsed time: {(time.process_time() - start_time) * 1000} ms')


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    show_augment_image()
