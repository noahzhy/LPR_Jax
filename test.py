# %%
import yaml
import jax
# cpu mode
jax.config.update('jax_platform_name', 'cpu')
import optax
import jax.numpy as jnp
from flax.training import checkpoints
from flax.training.train_state import TrainState

from model.model import TinyLPR
from model.tfr_dl import get_data


# load config from config.yaml
cfg = yaml.safe_load(open("config.yaml"))


# load model from checkpoint
def load_model(model, model_path, state):
    with open(model_path, "rb") as f:
        params = jax.pickle.load(f)
    state = state.replace(params=params)
    return state


# %%
key = jax.random.PRNGKey(0)
model = TinyLPR(**cfg["model"])

state = TrainState.create(
    apply_fn=model.apply,
    params=model.init(key, jnp.ones((1, 96, 192, 1))),
    tx=optax.adam(1e-3),
)

state = checkpoints.restore_checkpoint("checkpoints", state, 94)

tfrec = "/home/ubuntu/datasets/lpr/val.tfrecord"
ds = get_data(tfrec, batch_size=32, data_aug=False, n_map_threads=4)


# %%
import tensorflow_datasets as tfds

dl = tfds.as_numpy(ds[0])

# data[0]

for data in dl:
    imgs, mask, label = data
    pred_mask, _, ctc = state.apply_fn(state.params, imgs)
    print(jnp.argmax(ctc, axis=-1))
    _mask = jnp.sum(pred_mask[0], axis=-1)
    break

# show _mask via matplotlib
import matplotlib.pyplot as plt
plt.imshow(_mask)
plt.show()


# %%
# show _mask via matplotlib
import matplotlib.pyplot as plt
plt.imshow(imgs[0], cmap='gray')
plt.show()

# %%
