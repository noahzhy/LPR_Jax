import os, sys, random, time

import jax
jax.config.update('jax_platform_name', 'cpu')
import yaml
import optax
import jax.numpy as jnp
import tensorflow_datasets as tfds
from flax.training import train_state

sys.path.append("./utils")
from utils import batch_ctc_greedy_decoder, batch_remove_blank
from fit import lr_schedule, fit
from model.tfr_dl import get_data
from model.model import TinyLPR
from model.loss import *


cfg = yaml.safe_load(open("config.yaml"))

train_ds, train_len = get_data(**cfg["train"])
val_ds, val_len = get_data(**cfg["val"])

train_dl = tfds.as_numpy(train_ds)
val_dl = tfds.as_numpy(val_ds)

lr_fn = lr_schedule(cfg["lr"], train_len, cfg["epochs"], cfg["warmup"])


class TrainState(train_state.TrainState):
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params, batch):
            img, mask, label = batch
            pred_mask, pred_feats_ctc, pred_ctc = state.apply_fn(params, img)
            loss_ctc = focal_ctc_loss(pred_ctc, label, **cfg["focal_ctc_loss"])
            loss_mask = dice_bce_loss(pred_mask, mask)
            loss = 1.5 * loss_ctc + 0.5 * loss_mask
            return loss, {"loss_ctc": loss_ctc, "loss_mask": loss_mask}
        
        (loss, _dict), grad = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch)
        state = state.apply_gradients(grads=grad)
        return state, loss, _dict

    @jax.jit
    def test_step(state, batch):
        img, mask, label = batch
        pred_mask, pred_feats_ctc, pred_ctc = state.apply_fn(params, img)
        label = batch_remove_blank(label)
        pred = batch_ctc_greedy_decoder(pred_ctc)
        mean = jnp.mean(jnp.array([1 if jnp.array_equal(l, p) else 0 for l, p in zip(label, pred)]))
        return mean


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    model = TinyLPR(**cfg["model"])

    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(key, jnp.ones((1, *cfg["img_size"], 1))),
        tx=optax.adam(lr_fn),
        # tx=optax.chain(
        #     optax.clip_by_global_norm(1.0),
        #     optax.adam(lr_fn),
        #     optax.ema(0.999),
        # ),
        )

    fit(state, train_dl, val_dl, epochs=cfg["epochs"], lr_fn=lr_fn)
