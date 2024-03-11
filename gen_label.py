import os, sys, random, time, glob, math, timeit
from shutil import copyfile

import numpy as np
import PIL.Image as pil
from jamo import h2j, j2hcj, j2h


def load_dict(dict_path='data/labels.names'):
    with open(dict_path, 'r', encoding='utf-8') as f:
        _dict = f.read().splitlines()
    _dict = {h2j(_dict[i]): i for i in range(len(_dict))}
    return _dict


label_dict = load_dict('data/labels.names')
print(label_dict)

alpha_city = {
    'a': '서울', 'b': '부산', 'c': '대구', 'd': '인천',
    'e': '광주', 'f': '대전', 'g': '울산', 'h': '세종',
    'i': '경기', 'j': '강원', 'k': '충북', 'l': '충남',
    'm': '전북', 'n': '전남', 'o': '경북', 'p': '경남',
    'q': '제주',
}


def parse_numpy_label(label_path):
    # npy_data = np.load(label_path)
    npy_data = np.load(label_path)
    return npy_data


# split label
# e.g. '63루3348' -> ['6', '3', '루', '3', '3', '4', '8']
# e.g. '서울12가1234' -> ['서울', '1', '2', '가', '1', '2', '3', '4']
# e.g. 'A123B123' -> ['A', '1', '2', '3', 'B', '1', '2', '3'] 
def split_label(label):
    k_tmp = []
    split_label = []
    for i in label:
        if i.isdigit():
            if len(k_tmp) > 0:
                split_label.append(''.join(k_tmp))
                k_tmp = []
            split_label.append(i)
        else:
            k_tmp.append(i)
    return split_label


# unit test for split_label
def test_split_label():
    label = '63루3348'
    assert split_label(label) == ['6', '3', '루', '3', '3', '4', '8']
    label = '서울12가1234'
    assert split_label(label) == ['서울', '1', '2', '가', '1', '2', '3', '4']
    label = 'A123B123'
    assert split_label(label) == ['A', '1', '2', '3', 'B', '1', '2', '3']
    # color print
    print("\033[92m[pass]\033[00m split_label() test passed.")


aqAQ = list('abcdefghijklmnopqABCDEFGHIJKLMNOPQ')

# gen label
def gen_label(img_path, label_dict=label_dict):
    img_name = os.path.basename(img_path).replace(' ', '')
    label = img_name.split('.')[0].split('_')[0]
    label = split_label(label)
    if label[0] in aqAQ:
        label[0] = alpha_city[label[0].lower()]

    txt_label = ''.join(label)

    for i, char in enumerate(label):
        label[i] = label_dict[h2j(char)]

    return txt_label, label


# copy image file and gen label file
def copy_gen_label(img_path, dir, label_dict=label_dict):
    timestamp_ns = time.time_ns()

    txt_label, label = gen_label(img_path, label_dict)
    fname = '{}_{}.jpg'.format(txt_label, timestamp_ns)
    copyfile(img_path, os.path.join(dir, fname))
    # save as txt file
    txt_path = fname.replace('.jpg', '.txt')
    with open(os.path.join(dir, txt_path), 'w') as f:
        np.savetxt(f, label, fmt='%d')


# load a given txt to numpy
def txt2arr(txt_path):
    return np.loadtxt(txt_path, dtype=np.int32)


if __name__ == "__main__":
    # test split_label
    test_split_label()

    im_path = '/Users/haoyu/Downloads/lpr/train/*.jpg'
    im_path = glob.glob(im_path)
    random.shuffle(im_path)

    choose = random.choice(im_path)
    choose = '/Users/haoyu/Documents/datasets/lpr/train/379부2694_1710173672237940000.jpg'

    t = timeit.timeit(lambda: gen_label(choose), number=1000) / 1000
    print('{} ms'.format(t * 1000))
    print(gen_label(choose))

    # load a give txt to numpy
    txt_path = "/Users/haoyu/Documents/datasets/lpr/train/379부2694_1710173672237940000.txt"
    t = timeit.timeit(lambda: txt2arr(txt_path), number=1000) / 1000
    print('{} ms'.format(t * 1000))
    print(txt2arr(txt_path))
