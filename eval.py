import jax
import jax.numpy as jnp
import numpy as np

import itertools


# eval function ctc decode
def ctc_greedy_decoder(logits, blank=0):
    logits = jnp.argmax(logits, axis=-1)
    return [int(k) for k, _ in itertools.groupby(logits) if k != blank]


# test unit for ctc greedy decoder
def test_ctc_greedy_decoder():
    logits = jnp.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.7, 0.5],
        [0.5, 0.4, 0.3, 0.8, 0.1],
        [0.7, 0.3, 0.3, 0.3, 0.3],
    ])
    # print argmax and it's index
    print(jnp.argmax(logits, axis=-1))
    print(ctc_greedy_decoder(logits))
    print('\33[92m[pass]\33[00m ctc_greedy_decoder() test passed.')


if __name__ == '__main__':
    # cpu mode
    jax.config.update('jax_platform_name', 'cpu')

    test_ctc_greedy_decoder()
