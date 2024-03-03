import os, sys, random, time

import jax
import jax.numpy as jnp
import jax_dataloader as jdl


class LPR_Data(jdl.Dataset):
    def __init__(self, data_dir, img_size=(224, 224)):
        self.img_size = img_size
        self.data = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".jpg"):
                    img_path = os.path.join(root, file)
                    label = os.path.basename(root)
                    self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = jdl.image.imread(img_path)
        img = jdl.image.resize(img, self.img_size)
        img = jdl.image.normalize(img)
        return img, label
    
if __name__ == "__main__":
    data_dir = "data"
    dataset = LPR_Data(data_dir)
    dataloader = jdl.DataLoader(dataset, 'jax', batch_size=32, shuffle=True)
    for batch in dataloader:
        x, y = batch
        print(x.shape, y.shape)
        break
