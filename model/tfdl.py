import os, sys, random, time, glob, math
import multiprocessing
n_map_threads = multiprocessing.cpu_count() * 2

import jax
import tqdm
from PIL import Image
import jax.numpy as jnp
import tensorflow as tf

sys.path.append("./utils")
from data_aug import *


print(jax.devices())

# # jit with static argnums
# pad_mask_fn = jax.jit(pad_mask, static_argnums=1)
# resize_fn = jax.jit(resize_image, static_argnums=(1, 2))
# resize_ratio_fn = jax.jit(resize_image_keep_aspect_ratio, static_argnums=(1, 2))
# insert0align2right_fn = jax.jit(insert0align2right, static_argnums=1)


ds_desc = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
    'bbox': tf.io.FixedLenFeature([], tf.string),
    'size': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, ds_desc)


def load_tfreocrd(tffile_paths, batch_size=32, n_map_threads=n_map_threads):
    dataset = tf.data.TFRecordDataset(tffile_paths, compression_type='ZLIB')
    ds = dataset.map(_parse_image_function, num_parallel_calls=n_map_threads)
    ds = ds.shuffle(ds.cardinality())
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds


def gen_mask(bbox, h, w, len_label, time_step=None):
    # _mask = np.zeros((h, w, len_label), dtype=np.int32)
    _black = np.zeros((h, w, time_step), dtype=np.int32)

    for i, box in enumerate(bbox):
        b0 = max(0, box[0])
        b1 = max(0, box[1])
        b2 = min(w, box[2])
        b3 = min(h, box[3])
        # _mask[b1:b3, b0:b2, i] = 1
        _black[b1:b3, b0:b2, time_step-(2*(len_label-i)-1)] = 1

    return _black


def pad_mask(mask, time_step):
    # # sum to one channel
    _mask = np.sum(mask, axis=-1)
    # save it to test.png
    img = Image.fromarray(np.array(_mask * 255, dtype=np.uint8))
    img.save("test.png")

    # save each _black channel to {i}.png
    for i in range(mask.shape[-1]):
        img = Image.fromarray(np.array(mask[:, :, i] * 255, dtype=np.uint8))
        img.save(f"{i}.png")

    return mask

    # via tf
    # given mask shape (h, w, c)
    # pad mask to shape (h, w, time_step)
    # return tf.pad(mask, [[0, 0], [0, 0], [0, time_step - mask.shape[-1]]], 'CONSTANT')


def pad_label(label, time_step=15):
    _label = tf.zeros(len(label) * 2 - 1, dtype=tf.int32)

    for i in range(len(label)):
        _label = tf.tensor_scatter_nd_update(_label, [[i * 2]], [label[i]])

    return tf.pad(_label, [[time_step - len(_label), 0]], 'CONSTANT')


def resize_keep_ratio(image, target_size=(64, 128), method='bilinear'):
    """
    Scales an image to the specified size while keeping its aspect ratio.
    Args:
        image: A 3D array representing the image.
        target_size: A tuple representing the target size.
    Returns:
        A new 3D array representing the scaled image.
    """
    anti_alias = (method != 'nearest')

    # static target size
    h, w, c = image.shape
    th, tw = target_size
    # calculate aspect ratio
    aspect = w / h
    # calculate new size
    if aspect > tw / th:
        new_w = tw
        new_h = int(tw / aspect)
    else:
        new_h = th
        new_w = int(th * aspect)
    # resize image
    img = tf.image.resize(image, (new_h, new_w), method=method, antialias=anti_alias)
    img = tf.pad(img, [[0, th - new_h], [0, tw - new_w], [0, 0]], 'CONSTANT')
    return img


def data_augment(image):
    """
    Applies a series of random augmentations to an image.
    Args:
        image: A 3D array representing the image.
    Returns:
        A new 3D array representing the augmented image.
    """
    _bright = .5
    _contrast = [0.1, 1.2]
    _gamma = [0.1, 1.0]
    _hue = .2
    _saturation = [0.1, 1.0]

    # # k size
    # _k = 5
    # _sigma = 3

    _bright = tf.image.random_brightness(image, _bright)
    _contrast = tf.image.random_contrast(image, _contrast[0], _contrast[1])
    _gamma = tf.image.adjust_gamma(image, tf.random.uniform(()) * (_gamma[1] - _gamma[0]) + _gamma[0])
    _hue = tf.image.random_hue(image, _hue)
    _saturation = tf.image.random_saturation(image, _saturation[0], _saturation[1])

    _aug_fn = [
        _bright,
        _contrast,
        _gamma,
        _hue,
        _saturation,
        # _blur,
    ]

    return random.choice(_aug_fn)


# get tfrecord data length
def get_tfrecord_len(tfrecord):
    dataset = tf.data.TFRecordDataset(tfrecord, compression_type='ZLIB')
    return len(list(dataset))


def get_data(tfrecord, batch_size, img_size, time_steps, aug):
    for data in load_tfreocrd(tfrecord, batch_size):

        image_arr = tf.TensorArray(tf.float32, size=batch_size)
        mask_arr = tf.TensorArray(tf.int32, size=batch_size)
        label_arr = tf.TensorArray(tf.int32, size=batch_size)

        image = data['image']
        label = data['label']
        bbox = data['bbox']
        size = data['size']

        for i in range(batch_size):
            _size = tf.io.decode_raw(size[i].numpy(), tf.int64)
            _image = tf.io.decode_raw(image[i].numpy(), tf.uint8).numpy().reshape(_size[0], _size[1], 3)
            _label = tf.io.decode_raw(label[i].numpy(), tf.int64)
            _bbox = tf.io.decode_raw(bbox[i].numpy(), tf.int64).numpy().reshape(-1, 4)
            # tf to [0, 1]
            img = tf.cast(_image, tf.float32) / 255.
            _mask = gen_mask(_bbox, _size[0], _size[1], len(_label), time_steps)

            if aug: img = data_augment(img)

            if tf.random.uniform(()) > 0.5 and aug:
                img = tf.image.rgb_to_grayscale(img)
                img = tf.image.resize(img, img_size, antialias=True)
                mask = tf.image.resize(_mask, img_size, method='nearest', antialias=False)
            else:
                img = tf.image.rgb_to_grayscale(img)
                img = resize_keep_ratio(img, img_size)
                mask = resize_keep_ratio(_mask, img_size, method='nearest')

            # mask = pad_mask(mask, time_steps)
            _label = pad_label(_label, time_steps)

            image_arr = image_arr.write(i, img)
            mask_arr = mask_arr.write(i, mask)
            label_arr = label_arr.write(i, _label)

            # yield image_arr.stack(), mask_arr.stack(), label_arr.stack()
            # to np
            # yield image_arr.stack().numpy(), mask_arr.stack().numpy(), label_arr.stack().numpy()
        yield jnp.array(image_arr.stack()), jnp.array(mask_arr.stack()), jnp.array(label_arr.stack())


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    batch_size = 32
    # img_size = (64, 128)
    img_size = (96, 192)
    time_step = 15
    aug = True

    tfrecord_path = "data/val.tfrecord"
    _len = get_tfrecord_len(tfrecord_path)
    print(_len)
    data = get_data(tfrecord_path, batch_size, img_size, time_step, aug)

    for i in data:
        print(i[2])
        print(i[0].shape, i[1].shape, i[2].shape)
        break
