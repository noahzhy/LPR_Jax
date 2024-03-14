import jax
import jax.numpy as jnp

import numpy as np

# check if jax is using GPU
print(jax.devices())

# jnp test
x = jnp.array(np.random.rand(3, 3))
print(x)

# torch gpu
import torch
print(torch.cuda.is_available())




# from torch.utils.data import Dataset, DataLoader

# # test DataLoader

# class MyDataset(Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]


# data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# dataset = MyDataset(data)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# for i, batch in enumerate(dataloader):
#     print(i, batch)
