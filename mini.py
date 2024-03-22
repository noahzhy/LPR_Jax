
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
 
import time
from multiprocessing import Pool
import numpy as np
from jax import numpy as jnp
import jax
a = np.ones(1000000)
# print device
print(jax.devices())
 
def f(a):
    b = jnp.array(a)
    time.sleep(2)
    print('Array b has been deleted!')
    return True
 
with Pool(1) as p:
    res = p.map(f, [(a,)])
 
print ('Is jax array deleted successfully?\t{}'.format(res))
