import os, sys, random, time

import jax
# jax.config.update('jax_platform_name', 'cpu')
import yaml
import optax
import jax.numpy as jnp

sys.path.append("./utils")
from utils import batch_ctc_greedy_decoder, batch_remove_blank
from fit import lr_schedule, TrainState
from model.tfdl import get_data, get_tfrecord_len
from model.model import TinyLPR
from model.loss import *


key = jax.random.PRNGKey(0)
cfg = yaml.safe_load(open("config.yaml"))

train_dl = get_data(**cfg["train"])
val_dl = get_data(**cfg["val"])

steps_per_epoch = get_tfrecord_len(cfg["train"]["tfrecord"]) // cfg["batch_size"]
print(f"steps_per_epoch: {steps_per_epoch}")
lr_fn = lr_schedule(cfg["lr"], steps_per_epoch, cfg["epochs"], cfg["warmup"])


def loss_fn(params, batch, model):
    img, mask, label = batch
    pred_mask, pred_feats_ctc, pred_ctc = model(params, img)
    loss_ctc = focal_ctc_loss(pred_ctc, label, **cfg["focal_ctc_loss"])
    loss_mask = dice_bce_loss(pred_mask, mask)
    loss = 1.5 * loss_ctc + 0.5 * loss_mask
    return loss, {"loss_ctc": loss_ctc, "loss_mask": loss_mask}


def eval_fn(params, batch, model):
    img, mask, label = batch
    pred_mask, pred_feats_ctc, pred_ctc = model(params, img)
    label = batch_remove_blank(label)
    pred = batch_ctc_greedy_decoder(pred_ctc)
    mean = jnp.mean(jnp.array([1 if jnp.array_equal(l, p) else 0 for l, p in zip(label, pred)]))
    return mean


if __name__ == "__main__":
    model = TinyLPR(**cfg["model"])

    state = TrainState.create(
        val_frequency=1,
        log_name="tiny_lpr",
        apply_fn=model.apply,
        params=model.init(key, jnp.ones((1, *cfg["img_size"], 1))),
        tx=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr_fn),
            optax.ema(0.999),
        ),
        # tx=optax.adam(lr_fn),
        lr_fn=lr_fn,
        eval_fn=eval_fn,
        loss_fn=loss_fn,)

    state.fit(train_dl, val_dl, epochs=cfg["epochs"])
