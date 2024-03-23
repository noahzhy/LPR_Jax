import os, sys, random, time, glob, math
import multiprocessing
n_map_threads = multiprocessing.cpu_count() * 2

import jax
import tqdm
from PIL import Image
import jax.numpy as jnp
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")

sys.path.append("./utils")
from data_aug import *


def decode_data(example):
    ds_desc = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
        'size': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, ds_desc)

    image = tf.io.decode_raw(example['image'], tf.uint8)
    label = tf.io.decode_raw(example['label'], tf.int64)
    mask = tf.io.decode_raw(example['mask'], tf.int64)
    size = tf.io.decode_raw(example['size'], tf.int64)
    # convert to float32 and normalize
    image = tf.cast(image, tf.float32) / 255.
    return image, mask, label, size


def reshape_fn(image, mask, label, size, time_step=15, target_size=(96, 192)):
    image = tf.reshape(image, [size[0], size[1], 3])
    mask = tf.reshape(mask, [size[0], size[1], time_step])
    return image, mask, label, size


def resize_image(image, mask, label, size, target_size=(96, 192)):
    image = tf.image.resize(
        image, target_size, method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    mask = tf.image.resize(
        mask, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, antialias=False
    )
    return image, mask, label, size


def pad_label(label, time_step=15):
    _label = tf.zeros(len(label) * 2 - 1)
    for i in range(len(label)):
        _label = tf.tensor_scatter_nd_update(_label, [[i * 2]], [label[i]])

    return tf.pad(_label, [[time_step - len(_label), 0]], 'CONSTANT')


# def pad_label(label, time_step=15):
#     return tf.pad(label, [[0, time_step - len(label)]], 'CONSTANT')

def pad_mask(mask, time_step=16):
    ''' given mask shape (H, W, T), pad to (H, W, time_step) '''
    return tf.pad(mask, [[0, 0], [0, 0], [time_step - mask.shape[-1], 0]], 'CONSTANT')


def pad_image_mask(image, mask, label, size, time_step=16, target_size=(96, 192)):
    image = tf.image.rgb_to_grayscale(image)
    label = pad_label(label, time_step-1)
    mask = pad_mask(mask, time_step)
    return image, mask, label


def data_augment(image, mask, label, size):
    gamma = np.random.uniform(low=1.0, high=2, size=[1,])
    gain = np.random.uniform(low=0.7, high=1.5, size=[1,])
    image = tf.image.adjust_gamma(image, gamma[0], gain[0])
    image = tf.image.random_contrast(image, 0.2, 1.5)
    image = tf.image.random_hue(image, 0.3)
    image = tf.image.random_saturation(image, 0.1, 2.0)
    image = tf.image.random_brightness(image, 0.3)
    # clip to [0, 1]
    image = tf.clip_by_value(image, 0, 1)
    return image, mask, label, size


def get_data(tfrecord, batch_size=32, data_aug=True, n_map_threads=n_map_threads):
    dataset = tf.data.TFRecordDataset(tfrecord, compression_type='ZLIB')
    ds_len = sum(1 for _ in dataset) // batch_size
    ds = dataset.map(decode_data, num_parallel_calls=n_map_threads)
    ds = ds.map(reshape_fn, num_parallel_calls=n_map_threads)
    ds = ds.map(resize_image, num_parallel_calls=n_map_threads)

    if data_aug: ds = ds.map(data_augment, num_parallel_calls=n_map_threads)

    ds = ds.map(pad_image_mask, num_parallel_calls=n_map_threads)

    ds = ds.shuffle(2048, reshuffle_each_iteration=data_aug)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds, ds_len


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    key = jax.random.PRNGKey(0)
    batch_size = 8
    # img_size = (64, 128)
    img_size = (96, 192)
    time_step = 16
    aug = True

    # load dict from names file to dict
    with open("data/labels.names", "r") as f:
        names = f.readlines()
    names = [name.strip() for name in names]
    names = {i: name for i, name in enumerate(names)}
    print(names)

    # label = tf.constant([1, 2, 3, 4, 5, 6, 7, 8])
    # res = pad_label(label, time_step=15)
    # print(res)

    tfrecord_path = "/home/ubuntu/datasets/lpr/val.tfrecord"
    # tfrecord_path = "data/val.tfrecord"
    ds, ds_len = get_data(tfrecord_path, batch_size, aug)
    dl = tfds.as_numpy(ds)

    for data in tqdm.tqdm(dl, total=ds_len):
        img, mask, label = data
        print(img.shape, mask.shape, label.shape)

    #     # save one image as test.jpg
    #     img = img[0] * 255
    #     img = np.squeeze(img, -1)
    #     img = Image.fromarray(np.uint8(img))
    #     img.save('test.jpg')

        print(label[0])

        for i in range(16):
            print(mask[0][:,:,i])
            # save as i.png
            mask_ = mask[0][:,:,i] * 255
            mask_ = Image.fromarray(np.uint8(mask_))
            mask_.save(f'{i}.png')

    #     # sum the mask to one channel
    #     mask = mask[0] * 255
    #     mask = np.sum(mask, axis=-1)
    #     # save the mask as test.png
    #     mask = Image.fromarray(np.uint8(mask))
    #     mask.save('test.png')
        break
