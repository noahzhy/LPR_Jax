import yaml
import jax
# cpu mode
jax.config.update('jax_platform_name', 'cpu')
import optax
import jax.numpy as jnp
import tensorflow_datasets as tfds

from model.model import TinyLPR
from model.tfr_dl import get_data
from utils import *
from fit import lr_schedule, fit, TrainState, load_ckpt


def predict(state: TrainState, batch):
    img, _, label = batch
    pred_ctc = state.apply_fn({
        'params': state.params,
        'batch_stats': state.batch_stats
        }, img, train=False)
    return pred_ctc, label


def eval_step(state: TrainState, batch):
    pred_ctc, label = jax.jit(predict)(state, batch)
    label = batch_remove_blank(label)
    pred = batch_ctc_greedy_decoder(pred_ctc)
    acc = jnp.mean(jnp.array([1 if jnp.array_equal(
        l, p) else 0 for l, p in zip(label, pred)]))
    return state, acc


def eval(key, model, input_shape, ckpt_dir, test_val):
    var = model.init(key, jnp.zeros(input_shape, jnp.float32), train=False)
    params = var["params"]
    batch_stats = var["batch_stats"]

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=optax.adam(1e-3),
    )

    state = load_ckpt(state, ckpt_dir)

    ds, _ = get_data(test_val, batch_size=128, data_aug=False, n_map_threads=8)
    test_ds = tfds.as_numpy(ds)

    acc = []
    for batch in test_ds:
        _, a = eval_step(state, batch)
        acc.append(a)
    acc = jnp.stack(acc).mean()
    return acc


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    cfg = yaml.safe_load(open("config.yaml"))
    model = TinyLPR(**cfg["model"])

    input_shape = (1, *cfg["img_size"], 1)
    ckpt_dir = "tmp/blank_0"
    test_val = "data/val.tfrecord"

    acc = eval(key, model, input_shape, ckpt_dir, test_val)
    print("\33[32m", f"Accuracy: {acc}", "\33[0m")
