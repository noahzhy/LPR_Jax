import os, sys, random, time

import jax
import optax
from tqdm import tqdm
import jax.numpy as jnp
import tensorboardX as tbx
from flax.training import train_state
from flax.training import checkpoints

from model.model import TinyLPR
from model.loss import *


def lr_schedule(base_lr, steps_per_epoch, epochs=10, warnup_epochs=2):
    warnup_fn = optax.linear_schedule(
        init_value=0,
        end_value=base_lr,
        transition_steps=steps_per_epoch * warnup_epochs,
    )
    decay_fn = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=steps_per_epoch * (epochs - warnup_epochs),
    )
    schedule_fn = optax.join_schedules(
        schedules=[warnup_fn, decay_fn],
        boundaries=[steps_per_epoch * warnup_epochs],
    )
    return schedule_fn


class TrainState(train_state.TrainState):
    log_name: str = "tiny_lpr"

    def train_step(self, batch):
        def loss_fn(params, batch):
            x, y = batch
            mask, feats_ctc, ctc = self.apply_fn(params, x)
            return focal_ctc_loss(jax.nn.log_softmax(ctc), y)

        loss, grads = jax.value_and_grad(loss_fn)(self.params, batch)
        print(loss)
        self = self.apply_gradients(grads=grads)
        return self, loss

    def test_step(self, batch):
        x, y = batch
        logits = self.apply_fn(self.params, x)
        # ctc greedy decoder

        # accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        # return accuracy

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
    bs = 32
    train_ds = jnp.ones((bs, 64, 128, 1)), jnp.ones((bs, 8), dtype=jnp.int32)
    test_ds = jnp.ones((bs, 64, 128, 1)), jnp.ones((bs, 8), dtype=jnp.int32)
    train_ds = jdl.ArrayDataset(*train_ds)
    test_ds = jdl.ArrayDataset(*test_ds)

    train_jdl = jdl.DataLoader(train_ds, 'jax', batch_size=bs, shuffle=True)
    test_jdl = jdl.DataLoader(test_ds, 'jax', batch_size=bs, shuffle=False)
    for batch in train_jdl:
        print(batch[0].shape, batch[1].shape)
        break

    epochs = 100
    lr_fn = lr_schedule(2e-3, len(train_ds), epochs=epochs, warnup_epochs=5)

    x = jnp.ones((1, 64, 128, 1))
    # Create a state
    model = TinyLPR(time_steps=16, n_class=69, n_feat=64, train=True)
    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(key, x),
        tx=optax.adam(lr_fn),
    )

    # Fit the model
    state.fit(train_jdl, test_jdl, epochs=epochs, lr_fn=lr_fn)
