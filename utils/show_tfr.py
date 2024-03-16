import os
import sys
import glob

from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# load data from .tfrecord file

tffile_paths = 'data/val.tfrecord'
dataset = tf.data.TFRecordDataset(tffile_paths, compression_type='ZLIB')

# Create a dictionary describing the features.
dataset_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
    'bbox': tf.io.FixedLenFeature([], tf.string),
    'size': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, dataset_feature_description)


batch_size = 32

import multiprocessing
n_map_threads = multiprocessing.cpu_count()
parsed_image_dataset = dataset.map(_parse_image_function, num_parallel_calls=n_map_threads)

shuffle_buffer_size = max(batch_size * 128, 2048)  # set the minimum buffer size as 2048
parsed_image_dataset = parsed_image_dataset.shuffle(shuffle_buffer_size)

parsed_image_dataset = parsed_image_dataset.batch(batch_size, drop_remainder=True)
# show the first image
for data in parsed_image_dataset.take(1):
    # decode the image from bytes

    print(len(data['image']))

    img = data['image']
    label = data['label']
    bbox = data['bbox']
    size = data['size']

    for i in range(len(img)):
        _size = np.frombuffer(size[i].numpy(), dtype=np.int64)
        _image = np.frombuffer(img[i].numpy(), dtype=np.uint8).reshape(_size[0], _size[1], 3)
        _label = np.frombuffer(label[i].numpy(), dtype=np.int64)
        _bbox = np.frombuffer(bbox[i].numpy(), dtype=np.int64).reshape(-1, 4)

        print(_image.shape)
        print(_label.shape, _bbox.shape)

        plt.imshow(_image)
        plt.show()
        break



# if __name__ == "__main__":
#     file_path = 'data/val.tfrecord'
#     dataset = read_and_decode(file_path)
#     for data in dataset.take(1):
#         img = tf.image.decode_jpeg(data['image'])
#         label = data['label']
#         bbox = data['bbox']
#         print(img.shape, label, bbox)
#         plt.imshow(img)
#         plt.show()
