import os, sys, random, time

import jax
# cpu mode
jax.config.update('jax_platform_name', 'cpu')
import yaml
import torch
import optax
import jax.numpy as jnp

from model.dataloader import *
from model.model import TinyLPR
from model.loss import *

sys.path.append("./utils")
from utils import ctc_greedy_decoder
from fit import lr_schedule, TrainState


# load yaml config
cfg = yaml.safe_load(open("config.yaml"))
print(cfg)

key = jax.random.PRNGKey(0)

def loss_fn(params, batch, model):
    img, mask, label = batch
    pred_mask, pred_feats_ctc, pred_ctc = model(params, img)
    loss_ctc = focal_ctc_loss(
        pred_ctc, label,
        alpha=cfg["focal_loss"]["alpha"],
        gamma=cfg["focal_loss"]["gamma"],
    )
    loss_mask = dice_bce_loss(pred_mask, mask)
    return loss_ctc + loss_mask


def eval_fn(params, batch, model):
    img, mask, label = batch
    pred_mask, pred_feats_ctc, pred_ctc = model(params, img)
    pred = ctc_greedy_decoder(pred_ctc)
    return jnp.mean(pred == label)


train_dl = torch.utils.data.DataLoader(
    LPR_Data(key, cfg["train_dir"], time_step=cfg["time_steps"], img_size=cfg["img_size"], aug=True),
    batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], collate_fn=collate_fn, shuffle=True,
)

val_dl = torch.utils.data.DataLoader(
    LPR_Data(key, cfg["val_dir"], time_step=cfg["time_steps"], img_size=cfg["img_size"], aug=False),
    batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], collate_fn=collate_fn, shuffle=False,
)

lr_fn = lr_schedule(cfg["lr"], len(train_dl), cfg["epochs"], cfg["warmup"])


if __name__ == "__main__":
    # cpu mode
    jax.config.update("jax_platform_name", "cpu")

    print(f"train dataset size: {len(train_dl)}", f"val dataset size: {len(val_dl)}")

    model = TinyLPR(
        time_steps=cfg["time_steps"],
        n_class=cfg["n_class"],
        n_feat=cfg["n_feat"],
        train=True,
    )

    state = TrainState.create(
        log_name="tiny_lpr",
        apply_fn=model.apply,
        params=model.init(key, jnp.ones((1, *cfg["img_size"], 1))),
        tx=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr_fn)),
        lr_fn=lr_fn,
        eval_fn=eval_fn,
        loss_fn=loss_fn,)

    state.fit(train_dl, val_dl, epochs=cfg["epochs"])
