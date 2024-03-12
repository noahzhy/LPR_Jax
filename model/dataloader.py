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


def collate_fn(batch):
    return batch
    # if isinstance(batch[0], jnp.ndarray):
        # return jnp.stack(batch[0]), jnp.stack(batch[1]), batch[2]
    #     return jnp.stack(batch)
    # elif isinstance(batch[0], (tuple, list)):
    #     return type(batch[0])(collate_fn(samples) for samples in zip(*batch))
    # else:
    #     return jnp.asarray(batch)


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
        _mask, label = gen_mask(img_path)
        # 50 % chance to resize the image with or without keeping aspect ratio
        if jax.random.bernoulli(self.key, 0.5):
            img = resize_fn(img, self.img_size)
            mask = resize_fn(_mask, self.img_size, method='nearest')
        else:
            img = resize_ratio_fn(img, self.img_size)
            mask = resize_ratio_fn(_mask, self.img_size, method='nearest')

        mask = pad_mask_fn(mask, 16)

        self.key = jax.random.split(self.key)[0]
        return img, mask, label


# show data augmentation via matplotlib
def show_augment_image(samples=8):
    jax.config.update('jax_platform_name', 'cpu')
    key = jax.random.PRNGKey(random.randint(0, 1000))
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
        img, mask, label = batch[0]
        print(img.shape, mask.shape, len(label))
        break

    print(f'elapsed time: {(time.process_time() - start_time) * 1000} ms')

    # sum mask into one channel
    mask = mask.sum(axis=-1)
    img = img.squeeze()
    # show image and mask together
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].imshow(img, cmap='gray')
    axes[1].imshow(mask, cmap='gray')
    plt.show()

    # row = int(math.sqrt(samples) - 0.5)
    # col = samples // row
    # fig, axes = plt.subplots(row, col, figsize=(10, 5))
    # for i, ax in enumerate(axes.flatten()):
    #     ax.imshow(dataloader.dataset[i], cmap='gray')
    #     ax.axis('off')

    # plt.show()


if __name__ == "__main__":
    jax.config.update('jax_platform_name', 'cpu')
    key = jax.random.PRNGKey(0)

    show_augment_image()
