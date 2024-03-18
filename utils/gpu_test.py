import jax
import jax.numpy as jnp
import torch

print(torch.cuda.is_available())
print(jax.devices())
print(jax.__version__)

from torch.utils.data import Dataset, DataLoader

# test DataLoader

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dataset = MyDataset(data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)
for i, batch in enumerate(dataloader):
    print(i, batch)
