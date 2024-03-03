import time
from time import perf_counter

import jax
import optax
import jax.numpy as jnp

# ctc loss
@jax.jit
def ctc_loss(logits, targets, blank_id=0):
    logits_padding = jnp.zeros(logits.shape[:2])
    labels_padding = jnp.where(targets == blank_id, 1, 0)
    return optax.ctc_loss(
        logits=logits,
        labels=targets,
        logit_paddings=logits_padding,
        label_paddings=labels_padding,
    )


@jax.jit
def focal_ctc_loss(logits, targets, blank_id=0, alpha=0.25, gamma=2):
    loss = ctc_loss(logits, targets, blank_id)
    fc_loss = alpha * (1 - jnp.exp(-loss)) ** gamma * loss
    return fc_loss.mean()


def focal_ctc_loss_test():
    logits = jnp.ones((4, 16, 127))
    labels = jnp.array([
        [1,2,2,4,5,0,0,0],
        [6,2,1,1,7,7,5,0],
        [1,2,2,0,0,0,0,0],
        [3,2,1,2,0,0,0,0],
    ])

    loss = 0
    start_time = time.perf_counter()
    for i in range(1000):
        loss = focal_ctc_loss(logits, labels, blank_id=0)

    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / 1000
    return loss, avg_time


# dice bce loss via optax
@jax.jit
def dice_bce_loss(logits, targets, smooth=1e-6):
    # logits: (B, C, H, W)
    # targets: (B, C, H, W)
    # smooth: (float) smooth value
    # return: (B,)
    logits = jax.nn.log_softmax(logits)
    targets = jax.nn.log_softmax(targets)
    bce = optax.sigmoid_binary_cross_entropy(logits, targets)
    pred = jax.nn.sigmoid(logits)
    targets = jax.nn.sigmoid(targets)
    intersection = jnp.sum(pred * targets, axis=(1, 2, 3))
    union = jnp.sum(pred, axis=(1, 2, 3)) + jnp.sum(targets, axis=(1, 2, 3))
    dice = 1 - (2 * intersection + smooth) / (union + smooth)
    loss = bce + dice
    # mean loss
    return jnp.mean(loss)


def dice_bce_test():
    # test dice bce loss
    logits = jnp.array([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]])
    targets = jnp.array([[[[0, 1], [1, 0]], [[0, 1], [1, 0]]]])
    start_t = perf_counter()
    for i in range(1000):
        loss = dice_bce_loss(logits, targets)
    end_t = perf_counter()
    avg_time = (end_t - start_t) / 1000
    return loss, avg_time


if __name__ == "__main__":
    jax.config.update("jax_platform_name", "cpu")
    print(jax.devices())

    print(dice_bce_test())
    print(focal_ctc_loss_test())
