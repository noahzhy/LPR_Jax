import time, timeit
from time import perf_counter
from functools import partial

import jax
import optax
import jax.numpy as jnp


# ctc loss
@partial(jax.jit, static_argnums=(2,))
def ctc_loss(logits, targets, blank_id=0):
    logits_padding = jnp.zeros(logits.shape[:2])
    labels_padding = jnp.where(targets == blank_id, 1, 0)
    return optax.ctc_loss(
        logits=logits,
        labels=targets,
        logit_paddings=logits_padding,
        label_paddings=labels_padding,
    ).mean()


@partial(jax.jit, static_argnums=(2, 3, 4))
def focal_ctc_loss(logits, targets, blank_id=0, alpha=0.25, gamma=2):
    loss = ctc_loss(logits, targets, blank_id)
    return alpha * (1 - jnp.exp(-loss)) ** gamma * loss


def focal_ctc_loss_test():
    labels = jnp.array([
        #0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F
        [6,2,1,1,7,7,5,8,0,0,0,0,0,0,0],
        [9,0,4,0,1,0,2,0,4,0,8,0,9,0,8],
        [9,0,4,0,2,0,2,0,4,0,8,0,9,0,8],
    ])
    tmp = jnp.linspace(0, 1, labels.shape[1]*10).reshape(labels.shape[1], -1)
    logits = jnp.array([tmp, tmp, tmp])
    loss = 0
    # timeit and get function result
    start_t = perf_counter()
    for i in range(1000):
        loss = focal_ctc_loss(logits, labels, alpha=0.25, gamma=3)
    end_t = perf_counter()
    avg_time = (end_t - start_t) / 1000
    print('\33[92m[pass]\33[00m focal_ctc_loss() test passed.')
    print('  - loss:', loss)
    print('  - time: {:.6f} ms'.format(avg_time*1000))


# dice bce loss via optax
@jax.jit
def dice_bce_loss(logits, targets, smooth=1e-7):
    # logits: (B, H, W, C), get from sigmoid activation, not raw logits
    # targets: (B, H, W, C)
    # smooth: (float) smooth value
    # return: (B,)

    # dice loss
    pred = logits.flatten()
    true = targets.flatten()
    intersection = jnp.sum(pred * true)
    union = jnp.sum(pred) + jnp.sum(true)
    dice = 1 - (2 * intersection + smooth) / (union + smooth)

    # bce loss
    # clip and log
    logits = jnp.clip(logits, 1e-7, 1 - 1e-7)
    logits = jnp.log(logits / (1 - logits))
    bce = optax.sigmoid_binary_cross_entropy(logits, targets).mean()

    return bce + dice


def dice_bce_test():
    # test dice bce loss
    logits = jnp.zeros((1, 10, 10, 3))
    logits = logits.at[:, 2:8, 2:8, :].set(1)
    targets = jnp.zeros((1, 10, 10, 3))
    targets = targets.at[:, 2:8, 2:8, :].set(1)

    start_t = perf_counter()
    for i in range(1000):
        loss = dice_bce_loss(logits, targets)
    end_t = perf_counter()

    avg_time = (end_t - start_t) / 1000
    print('\33[92m[pass]\33[00m dice_bce_loss() test passed.')
    print('  - loss:', loss)
    print('  - time: {:.6f} ms'.format(avg_time*1000))


if __name__ == "__main__":
    # cpu mode
    jax.config.update('jax_platform_name', 'cpu')
    print(jax.devices())

    dice_bce_test()
    focal_ctc_loss_test()
