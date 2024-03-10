import os, sys, random, time

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
# from utils.utils import *


def lr_schedule(base_lr, steps_per_epoch, epochs=10, warnup_epochs=2):
    return optax.warmup_cosine_decay_schedule(
        init_value=base_lr / 10,
        peak_value=base_lr,
        warmup_steps=steps_per_epoch * warnup_epochs,
        decay_steps=steps_per_epoch * (epochs - warnup_epochs),
    )


class TrainState(train_state.TrainState):
    log_name: str = "tiny_lpr"

    def train_step(self, batch):
        def loss_fn(params, batch):
            x, y = batch
            mask, feats_ctc, ctc = self.apply_fn(params, x)
            return focal_ctc_loss(ctc, y)

        loss, grads = jax.value_and_grad(loss_fn)(self.params, batch)
        self = self.apply_gradients(grads=grads)
        return self, loss

    def test_step(self, batch):
        x, y = batch
        logits = self.apply_fn(self.params, x)
        pred = ctc_greedy_decoder(logits)
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
                    keep=2,)

        tbx_writer.close()


if __name__ == "__main__":
    # cpu mode
    jax.config.update("jax_platform_name", "cpu")
    key = jax.random.PRNGKey(0)
    bs = 8
    epochs = 100
    data_dir = "data/val"

    # Create a dataset
    train_ds = LPR_Data(key, data_dir, img_size=(64, 128), aug=True)
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
    )
    print(f"train dataset size: {len(train_dl)}")
    
    lr_fn = lr_schedule(2e-3, len(train_dl), epochs=epochs, warnup_epochs=5)

    x = jnp.ones((1, 64, 128, 1))
    # Create a state
    model = TinyLPR(time_steps=16, n_class=69, n_feat=64, train=True)
    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(key, x),
        tx=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr_fn),
        ),
    )

    # Fit the model
    state.fit(train_dl, test_dl, epochs=epochs, lr_fn=lr_fn)
