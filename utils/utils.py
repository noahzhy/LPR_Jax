import cv2
import numpy as np
from PIL import Image
import jax.numpy as jnp


def cv2_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def cv2_imwrite(file_path, img):
    cv2.imencode('.jpg', img)[1].tofile(file_path)


def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w*r), height)
    else:
        r = width / float(w)
        dim = (width, int(h*r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# center fit and support rgb img
def center_fit(img, w, h, inter=cv2.INTER_NEAREST, top_left=True):
    # get img shape
    img_h, img_w = img.shape[:2]
    # get ratio
    ratio = min(w / img_w, h / img_h)

    if len(img.shape) == 3:
        inter = cv2.INTER_AREA
    # resize img
    img = cv2.resize(img, (int(img_w * ratio), int(img_h * ratio)), interpolation=inter)
    # get new img shape
    img_h, img_w = img.shape[:2]
    # get start point
    start_w = (w - img_w) // 2
    start_h = (h - img_h) // 2

    if top_left:
        start_w = 0
        start_h = 0

    if len(img.shape) == 2:
        # create new img
        new_img = np.zeros((h, w), dtype=np.uint8)
        new_img[start_h:start_h+img_h, start_w:start_w+img_w] = img
    else:
        new_img = np.zeros((h, w, 3), dtype=np.uint8)
        new_img[start_h:start_h+img_h, start_w:start_w+img_w, :] = img

    return new_img