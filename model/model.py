import os, sys, random, time

import jax
import jax.numpy as jnp
import flax.linen as nn


def _make_divisible(v, divisor, min_value=16):
    """https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


class UpSample(nn.Module):
    up_repeat: int = 3

    @nn.compact
    def __call__(self, x):
        for _ in range(self.up_repeat):
            x = jax.image.resize(x, shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), method="bilinear")
            x = nn.Conv(features=64, kernel_size=(5, 5), strides=(1, 1), padding="same", kernel_init=nn.initializers.he_normal(), use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=True)(x)
            x = nn.PReLU()(x)
        return x


class BottleNeck(nn.Module):
    in_size: int = 16
    exp_size: int = 16
    out_size: int = 16
    s: int = 1
    k: int = 3
    activation: callable = nn.relu
    scale: float = 1.0

    def setup(self):
        self._in = _make_divisible(self.in_size * self.scale, 8)
        self.exp = _make_divisible(self.exp_size * self.scale, 8)
        self.out = _make_divisible(self.out_size * self.scale, 8)

    @nn.compact
    def __call__(self, inputs):
        # shortcut
        x = nn.Conv(features=self.exp, kernel_size=(1, 1), strides=1, padding="same", kernel_init=nn.initializers.he_normal(), use_bias=False)(inputs)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = self.activation(x)

        x = nn.Conv(features=x.shape[-1], kernel_size=(self.k, self.k), strides=self.s, padding="same", feature_group_count=x.shape[-1], use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = self.activation(x)

        x = nn.Conv(features=self.out, kernel_size=(1, 1), strides=1, padding="same", kernel_init=nn.initializers.he_normal(), use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=True)(x)

        if self.s == 1 and self._in == self.out:
            x = jnp.add(x, inputs)

        return x


class MobileNetV3Small(nn.Module):
    width_multiplier: float = 1.0
    out_channels: int = 64

    def setup(self):
        self.bnecks = [
            # k  in    exp     out      NL          s
            [3,  16,    16,     16,     nn.relu,    2],
            [3,  16,    72,     24,     nn.relu,    1],
            [3,  24,    88,     24,     nn.relu,    1],
            [5,  24,    96,     40,     nn.relu6,   2],
            [5,  40,    240,    40,     nn.relu6,   1],
            [5,  40,    120,    48,     nn.relu6,   1],
            [5,  48,    144,    48,     nn.relu6,   1],
        ]

    @nn.compact
    def __call__(self, x):
        # 64, 128, 1
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)

        for i, (k, _in, exp, out, NL, s) in enumerate(self.bnecks):
            x = BottleNeck(_in, exp, out, s, k, NL, self.width_multiplier)(x)

        # last
        x = BottleNeck(16, 72, self.out_channels, 1, 5, nn.relu, 1.0)(x)
        return x


class ChannelAttn(nn.Module):

    @nn.compact
    def __call__(self, inputs):
        c = inputs.shape[-1]
        # global average pooling 2d
        x = jnp.mean(inputs, axis=(1, 2), keepdims=True)
        x = nn.Conv(features=c, kernel_size=(1, 1), use_bias=False)(x)
        x = nn.sigmoid(x)
        mul = jnp.multiply(x, inputs)
        return mul


class SpatialAttn(nn.Module):

    @nn.compact
    def __call__(self, inputs):
        _avg = jnp.mean(inputs, axis=-1, keepdims=True)
        _max = jnp.max(inputs, axis=-1, keepdims=True)
        x = jnp.concatenate([_max, _avg], axis=-1)
        x = nn.Conv(features=1, kernel_size=(1, 1), use_bias=False)(x)
        x = nn.sigmoid(x)
        mul = jnp.multiply(x, inputs)
        return mul


class Conv1x1(nn.Module):
    features: int = 16

    @nn.compact
    def __call__(self, inputs):
        x = nn.Conv(
            features=self.features,
            kernel_size=(1, 1),
            use_bias=True,
            kernel_init=nn.initializers.kaiming_normal()
        )(inputs)
        x = nn.relu(x)
        return x


class Attention(nn.Module):
    channels: int = 64
    temporal: int = 16

    def setup(self):
        self.channel_attention = ChannelAttn()
        self.spatial_attention = SpatialAttn()
        self.char_map = Conv1x1(features=self.temporal)

    @nn.compact
    def __call__(self, inputs):
        _channel = self.channel_attention(inputs)
        _spatial = self.spatial_attention(_channel)

        # 1x1 conv
        char_map = self.char_map(inputs)

        x = jnp.reshape(char_map, (-1, 128, self.temporal))
        y = jnp.reshape(_spatial, (-1, 128, self.channels))
        out = jnp.einsum("ijk,ijl->ikl", x, y)

        return out, char_map


class TinyLPR(nn.Module):
    time_steps: int = 15
    n_class: int = 68
    n_feat: int = 64
    train: bool = True

    def setup(self):
        self.backbone = MobileNetV3Small(0.25, self.n_feat)
        self.attention = Attention(self.n_feat, self.time_steps)
        self.dense = nn.Dense(
            features=self.n_class,
            kernel_init=nn.initializers.kaiming_normal(),
        )

    @nn.compact
    def __call__(self, inputs):
        x = self.backbone(inputs)
        mat, attn = self.attention(x)
        dense = self.dense(mat)
        ctc = nn.log_softmax(dense)

        if self.train:
            attn = UpSample(up_repeat=3)(attn)
            attn = nn.Conv(features=self.time_steps,
                kernel_size=(1, 1),
                strides=1,
                padding="same",
                kernel_init=nn.initializers.kaiming_normal(),
                use_bias=True
            )(attn)
            attn = nn.sigmoid(attn)
            feats_ctc = jnp.concatenate([mat, ctc], axis=-1)

            return attn, feats_ctc, ctc

        return ctc


if __name__ == '__main__':
    # cpu mode
    jax.config.update("jax_platform_name", "cpu")
    model = TinyLPR(time_steps=15, n_class=68, n_feat=64, train=True)
    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 64, 128, 1))
    print(model.tabulate(key, x))

    params = model.init(key, x)
    y = model.apply(params, x)
    for i in y:
        print(i.shape)

    from loss import focal_ctc_loss

    def loss_fn(params, x, y):
        time_start = time.perf_counter()
        mask, feats_ctc, ctc = model.apply(params, x)
        # ctc = model.apply(params, x)
        time_end = time.perf_counter()
        print("time: {:.4f} s".format(time_end - time_start))
        loss = focal_ctc_loss(ctc, y)
        return loss

    x, y = jnp.ones((1, 64, 128, 1)), jnp.ones((1, 8), dtype=jnp.int32)
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    print("loss:", loss)
