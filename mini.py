
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
 
import time
from multiprocessing import Pool
import numpy as np
from jax import numpy as jnp
import jax
a = np.ones(1000000)
# print device
print(jax.devices())

import tensorflow as tf
# gpu check
print(tf.config.list_physical_devices())

