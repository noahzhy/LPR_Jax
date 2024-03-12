import os, sys, random, time

import yaml
import torch
import jax
import optax
from tqdm import tqdm
import jax.numpy as jnp
import tensorboardX as tbx
from flax.training import train_state
from flax.training import checkpoints

from model.dataloader import *
from model.model import TinyLPR
from model.loss import *


# load yaml config
cfg = yaml.safe_load(open("config.yaml"))
print(cfg)

# learning rate schedule
def lr_schedule(base_lr, steps_per_epoch, epochs=100, warmup_epochs=5):
    return optax.warmup_cosine_decay_schedule(
        init_value=base_lr / 10,
        peak_value=base_lr,
        warmup_steps=steps_per_epoch * warmup_epochs,
        decay_steps=steps_per_epoch * (epochs - warmup_epochs),
    )


class TrainState(train_state.TrainState):
    log_name: str = "tiny_lpr"

    def train_step(self, batch):
        def loss_fn(params, batch):
            img, mask, label = batch
            pred_mask, pred_feats_ctc, pred_ctc = self.apply_fn(params, img)
            loss_ctc = focal_ctc_loss(
                pred_ctc, label,
                alpha=cfg["focal_loss"]["alpha"],
                gamma=cfg["focal_loss"]["gamma"],
            )
            loss_mask = dice_bce_loss(pred_mask, mask)
            return loss_ctc + loss_mask

        loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(self.params, batch)
        self = self.apply_gradients(grads=grads)
        return self, loss

    def test_step(self, batch):
        img, mask, label = batch
        pred_mask, pred_feats_ctc, pred_ctc = self.apply_fn(self.params, img)
        pred = ctc_greedy_decoder(pred_ctc)
        return jnp.mean(pred == y)

    def fit(self, train_ds, test_ds, epochs=10, lr_fn=None):
        tbx_writer = tbx.SummaryWriter("logs/{}".format(self.log_name))
        for epoch in range(1, epochs + 1):
            pbar = tqdm(train_ds)
            for batch in pbar:
                self, loss = self.train_step(batch)
                lr = lr_fn(self.step)
                tbx_writer.add_scalar("loss", loss, self.step)
                tbx_writer.add_scalar("learning rate", lr, self.step)
                pbar.set_description(f"epoch: {epoch:3d}, loss: {loss:.4f}, lr: {lr:.4f}")

            if epoch % 1 == 0:
                accuracy = jnp.array([])
                for batch in test_ds:
                    accuracy = jnp.append(accuracy, self.test_step(batch))
                accuracy = accuracy.mean()
                tbx_writer.add_scalar("accuracy", accuracy, self.step)
                print(f"epoch: {epoch:3d}, accuracy: {accuracy:.4f}")

                # get absolute path of this file
                ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
                checkpoints.save_checkpoint(
                    ckpt_dir=ckpt_dir,
                    target=self,
                    step=epoch,
                    overwrite=True,
                    keep=cfg["keep_ckpts"],
                )

        tbx_writer.close()


if __name__ == "__main__":
    # cpu mode
    jax.config.update("jax_platform_name", "cpu")
    key = jax.random.PRNGKey(0)
    # Create a dataset
    train_dl = torch.utils.data.DataLoader(
        LPR_Data(key, cfg["train_dir"], time_step=cfg["time_steps"], img_size=cfg["img_size"], aug=True),
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
        shuffle=True,
    )
    val_dl = torch.utils.data.DataLoader(
        LPR_Data(key, cfg["val_dir"], time_step=cfg["time_steps"], img_size=cfg["img_size"], aug=False),
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
    )
    print(f"train dataset size: {len(train_dl)}", f"val dataset size: {len(val_dl)}")

    lr_fn = lr_schedule(
        cfg["lr"],
        len(train_dl),
        epochs=cfg["epochs"],
        warmup_epochs=cfg["warmup"],
    )

    x = jnp.ones((1, *cfg["img_size"], 1))
    print(f"input shape: {x.shape}")
    # Create a state
    model = TinyLPR(
        time_steps=cfg["time_steps"],
        n_class=cfg["n_class"],
        n_feat=cfg["n_feat"],
        train=True,
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(key, x),
        tx=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr_fn),
        ),
    )

    # Fit the model
    state.fit(train_dl, val_dl, epochs=cfg["epochs"], lr_fn=lr_fn)
