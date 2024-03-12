import os, sys, random, time, glob

import jax
import numpy as np
import dm_pix as pix
import PIL.Image as pil
import jax.numpy as jnp
import matplotlib.pyplot as plt


def load_image(image_path):
    return jnp.array(pil.open(image_path), dtype=jnp.float32) / 255.


@jax.jit
def insert_zeros(data):
    """
    Inserts a zero between each element of a list.
    Args:
        data: A list of elements.
    Returns:
        A new list with zeros inserted between each element of the original list.
    """
    result = jnp.zeros(len(data) * 2 - 1, dtype=jnp.int32)
    result = jnp.array(result.at[::2].set(data))
    return result


# test unit
def test_insert_zeros():
    data = [1, 2, 3, 4, 5]
    result = insert_zeros(data).tolist()
    assert jnp.all(result == [1, 0, 2, 0, 3, 0, 4, 0, 5])
    # color print
    print("\033[92m[pass]\033[00m insert_zeros() test passed.")


# insert zeros and align to the right
def insert0align2right(data, n=16):
    """
    Inserts a zero between each element of a list and aligns to the right.
    Args:
        data: A list of elements.
        n: A number to align the list to the right.
    Returns:
        A new list with zeros inserted between each element of the original list and aligned to the right.
    """
    # insert zeros
    data = insert_zeros(data)
    result = jnp.zeros(n, dtype=jnp.int32)
    result = jnp.array(result.at[n - len(data):].set(data))
    return result


def test_insert0align2right():
    data = [1, 2, 3, 4, 5]
    n = random.randint(len(data) * 2, len(data) * 4)
    result = insert0align2right(data, n).tolist()
    # padding zeros
    _z = insert_zeros(data)
    assert jnp.pad(_z, (n - len(_z), 0)).tolist() == result
    # color print
    print("\033[92m[pass]\033[00m insert0align2right() test passed.")


# jax jit to grayscale
@jax.jit
def to_grayscale(image):
    return pix.rgb_to_grayscale(image)


# jax.jit image resize but not keep aspect ratio
# @jax.jit
def resize_image(image, target_size=(64, 128), method='bilinear'):
    """
    Scales an image to the specified size while keeping its aspect ratio.
    Args:
        image: A 3D array representing the image.
        target_size: A tuple representing the target size.
    Returns:
        A new 3D array representing the scaled image.
    """
    _, _, c = image.shape
    return jax.image.resize(image, (*target_size, c), method=method)


# resize and keep aspect ratio
# @jax.jit
def resize_image_keep_aspect_ratio(image, target_size=(64, 128), method='bilinear'):
    """
    Scales an image to the specified size while keeping its aspect ratio.
    Args:
        image: A 3D array representing the image.
        target_size: A tuple representing the target size.
    Returns:
        A new 3D array representing the scaled image.
    """
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
    image = jax.image.resize(image, (new_h, new_w, c), method=method)
    image = jnp.pad(image, ((0, th - new_h), (0, tw - new_w), (0, 0)))
    return image


# jax jit pix augmentation
@jax.jit
def augment_image(image, key):
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

    # k size
    _k = 5
    _sigma = 3

    _hue = pix.random_hue(key, image, _hue)
    _brightness = pix.random_brightness(key, image, _bright)
    _gamma = pix.random_gamma(key, image, _gamma[0], _gamma[1])
    _contrast = pix.random_contrast(key, image, _contrast[0], _contrast[1])
    _saturation = pix.random_saturation(key, image, _saturation[0], _saturation[1])
    _blur = pix.gaussian_blur(image, _k, _sigma)

    aug_fn = jnp.array([
        _hue,
        _brightness,
        _gamma,
        _contrast,
        _saturation,
        _blur,
    ])

    # jax random choice and apply
    image = jax.random.choice(key, aug_fn)
    image = to_grayscale(image)
    image = jnp.clip(image, 0, 1)
    return image


def test_augment_image(output_dir='logs/images'):
    # load image
    _path = random.choice(glob.glob('data/val/*.jpg'))
    # _path = 'data/val/6907.jpg'
    img_raw = load_image(_path)
    for i in range(20):
        key = jax.random.PRNGKey(random.randint(0, 1000))
        img = augment_image(img_raw, key)
        img = resize_image_keep_aspect_ratio(img)
        # save grayscale image via matplotlib
        img = jnp.clip(img * 255, 0, 255).astype(jnp.uint8)
        img = img[..., 0]
        # save grayscale image
        plt.imsave(f'{output_dir}/{i}.jpg', img, cmap='gray')

    print('\033[92m[pass]\033[00m augment_image() test passed.')


# pad mask
# @jax.jit
def pad_mask(mask, n=16):
    # pad the mask with 0 to make it (H, W, n)
    return jnp.pad(mask, ((0, 0), (0, 0), (0, n - mask.shape[-1])), mode='constant', constant_values=0)


def test_pad_mask():
    mask = jnp.ones((64, 128, 10), dtype=jnp.int32)
    n = random.randint(11, 20)
    pad_mask_fn = jax.jit(pad_mask, static_argnums=1)
    mask = pad_mask_fn(mask, n)
    assert mask.shape[-1] == n
    # color print
    print("\033[92m[pass]\033[00m pad_mask() test passed.")


if __name__ == "__main__":
    # cpu mode
    jax.config.update('jax_platform_name', 'cpu')
    key = jax.random.PRNGKey(random.randint(0, 1000))

    test_pad_mask()
    test_insert_zeros()
    test_insert0align2right()
    test_augment_image()
