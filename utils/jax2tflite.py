import os, sys, random, glob

import jax
import tensorflow as tf
import jax.numpy as jnp
from jax.experimental import jax2tf

sys.path.append("./model")
sys.path.append("./utils")
from utils import *
from model import *


class RepresentativeDataset:
    def __init__(self, val_dir, input_shape=(1, 64, 128, 1), sample_size=200):
        self.input_shape = input_shape
        self.representative_list = random.sample(
            glob.glob(os.path.join(val_dir, '*.jpg')),
            sample_size,)

    def __call__(self):
        for image_path in self.representative_list:
            n, h, w, c = self.input_shape
            img = center_fit(
                cv2.cvtColor(cv2_imread(image_path), cv2.COLOR_BGR2GRAY),
                w, h, inter=cv2.INTER_AREA, top_left=True)
            img = np.reshape(img, self.input_shape).astype('float32') / 255.
            print(image_path)
            yield [img]


def jax2tflite(key, model, input_shape, dataset, save_path='model.tflite',
               inference_input_type=tf.uint8,
               inference_output_type=tf.uint8):

    params = model.init(key, jnp.ones(input_shape))

    def predict(input_img):
        return model.apply(params, input_img)

    tf_predict = tf.function(
        jax2tf.convert(predict, enable_xla=False),
        input_signature=[
            tf.TensorSpec(
                shape=list(input_shape),
                dtype=tf.float32,
                name='input_image')],
        autograph=False)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [tf_predict.get_concrete_function()], tf_predict)

    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    converter.representative_dataset = dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.inference_input_type = inference_input_type
    converter.inference_output_type = inference_output_type
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    with open('{}'.format(save_path), 'wb') as f:
        f.write(converter.convert())

    print('\033[92m[done]\033[00m Model converted to tflite.')


if __name__ == "__main__":
    IMG_SIZE = (1, 64, 128, 1)
    SAMPLE_SIZE = 100
    VAL_DIR = "/Users/haoyu/Downloads/lpr/val"
    val_ds = RepresentativeDataset(VAL_DIR, IMG_SIZE, SAMPLE_SIZE)

    model = TinyLPR(train=False)
    key = jax.random.PRNGKey(0)
    jax2tflite(key, model, IMG_SIZE, val_ds, save_path='model.tflite')
