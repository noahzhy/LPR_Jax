import jax
# cpu mode
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import optax
from tqdm import tqdm
from itertools import groupby
from functools import partial

import sys
sys.path.append("./model")
from loss import focal_ctc_loss


# load dict from names file to dict
with open("data/labels.names", "r") as f:
    names = f.readlines()
names = [name.strip() for name in names]
names = {i: name for i, name in enumerate(names)}
print(names)
# count number of classes
num_classes = len(names)
print(num_classes)


# @partial(jax.jit, static_argnums=(2, 3, 4))
# def focal_ctc_loss(logits, targets, blank=0, alpha=1.0, gamma=2.0):
#     """
#     Focal CTC loss
#     """
#     logits_padding = jnp.zeros(logits.shape[:2])
#     labels_padding = jnp.where(targets == -1, 1, 0)
#     # labels_padding = jnp.zeros(targets.shape)
#     loss = optax.ctc_loss(
#         logits=logits,
#         labels=targets,
#         logit_paddings=logits_padding,
#         label_paddings=labels_padding,
#     )
#     fc_loss = alpha * (1 - jnp.exp(-loss)) ** gamma * loss
#     return fc_loss.mean()
#     # return loss.mean()

targets = jnp.array([
    #0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F
    # [6,2,1,1,7,7,5,8,0,0,0,0,0,0,0,0],
    # [0,0,0,0,0,6,2,1,1,7,7,5,8,0,0,0],
    [9,0,4,0,1,0,2,0,4,0,8,0,9,0,8],
    [9,0,4,0,1,0,2,0,4,0,8,0,9,0,8],
    [-1,-1,4,0,1,0,2,0,4,0,8,0,9,0,8],
    [-1,-1,4,0,1,0,2,0,4,0,8,0,9,0,8],
])
n, t = targets.shape
tmp = jnp.linspace(0, 1, n * t * 99)
logits = jnp.array(tmp.reshape(n, t, 99))

print("logits:", logits.shape)

loss = focal_ctc_loss(logits, targets, -1)
print("initial loss:", loss)


blank_id = -1
lr = 2
# grad 100 times
pbar = tqdm(range(100))
for _ in pbar:
    grad = jax.grad(focal_ctc_loss)(logits, targets, blank_id, 0.25, 2.0)
    logits -= lr * grad
    loss = focal_ctc_loss(logits, targets, blank_id, 0.25, 2.0)
    pbar.set_description(f"loss: {loss:.4f}")

print(jnp.argmax(logits, axis=-1))


def ctc_greedy_decoder(logits, blank=0):
    logits = jnp.argmax(logits, axis=-1)
    return [int(k) for k, _ in groupby(logits) if k != blank]


# batch ctc greedy decoder
def batch_ctc_greedy_decoder(logits, blank=0):
    return [ctc_greedy_decoder(logit, blank) for logit in logits]


print(batch_ctc_greedy_decoder(logits))





