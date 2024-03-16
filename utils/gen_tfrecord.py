import os, sys, glob

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append("./utils")
from gen_label import *
from utils import load_image


# resize image to width=128 and keep aspect ratio, also resize the bboxes(4 points)
# bbox: [x1, y1, x2, y2] and int64
def resize_image_keep_aspect_ratio(image, bbox, width=192):
    h, w, _ = image.shape
    ratio = width / w
    new_h = int(h * ratio)
    image = tf.image.resize(image, (new_h, width), antialias=True)
    bbox = tf.cast(bbox, tf.float32)
    bbox = tf.cast(tf.round(bbox * ratio), tf.int64)
    return image, bbox


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def gen_tfrecord(dir_path, file_name):
    writer = tf.io.TFRecordWriter(
        'data/{}.tfrecord'.format(file_name),
        options=tf.io.TFRecordOptions(compression_type='ZLIB'))

    img_ds = glob.glob(dir_path + '/*.jpg')
    # shuffle the dataset
    np.random.shuffle(img_ds)

    for img_path in img_ds:
        txt_path = img_path.replace('.jpg', '.txt')
        bbox = np.loadtxt(txt_path, dtype=np.int64)
        _, label = gen_label(img_path)

        image = Image.open(img_path).convert('RGB')
        image, bbox = resize_image_keep_aspect_ratio(np.array(image, dtype=np.float32), bbox)
        height, width, _ = image.shape

        # # draw box on the image
        # image = Image.fromarray(np.array(image, dtype=np.uint8))
        # draw = ImageDraw.Draw(image)

        # for box in bbox:
        #     draw.rectangle([box[0], box[1], box[2], box[3]], outline='red')

        # # img.show()
        # # save the image
        # image.save('test.jpg')
        # quit()

        image = np.array(image, dtype=np.uint8).tobytes()
        label = np.array(label, dtype=np.int64).tobytes()
        bbox = np.array(bbox, dtype=np.int64).tobytes()

        size = np.array([height, width], dtype=np.int64).tobytes()

        feature = {
            'image': _bytes_feature(image),
            'label': _bytes_feature(label),
            'bbox': _bytes_feature(bbox),
            'size': _bytes_feature(size),
        }

        writer.write(tf.train.Example(features=tf.train.Features(
            feature=feature)).SerializeToString())

    writer.close()


if __name__ == '__main__':
    val_path = '/Users/haoyu/Documents/datasets/lpr/val'
    train_path = '/Users/haoyu/Documents/datasets/lpr/train'

    gen_tfrecord(val_path, 'val')
    gen_tfrecord(train_path, 'train')
    print('done')
