import numpy as np
import cv2
import os
import random
import tensorflow as tf
from shutil import copyfile


def tensor2im(image_tensor, flip=-1, format="NCHW"):
    with tf.name_scope("Tensor2image"):

        if format == "NCHW":
            image_tensor = tf.transpose(image_tensor, (0, 2, 3, 1))

        image_tensor = image_tensor[:, :, :, :3] * 255.0
        image_tensor = tf.clip_by_value(image_tensor, 0, 255)
        image_tensor = image_tensor[:, :, :, ::flip]

        return image_tensor


def numpy2im(image_numpy, flip=-1, format="CHW"):

    if len(image_numpy.shape) == 4:
        image_numpy = image_numpy[0]

    if format == "CHW":
        image_numpy = np.transpose(image_numpy, (1, 2, 0))

    image_numpy = image_numpy[:, :, :3] * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    image_numpy = image_numpy[:, :, ::flip]

    return image_numpy


def makeLinearWeight(step,
                     start, end,
                     start_value, end_value,
                     clip_min=0.0, clip_max=1.0):

    with tf.name_scope("linearWeight"):
        linear = ((end_value - start_value) / (end - start) * (tf.cast(step, tf.float32) - start) + start_value)
        return tf.clip_by_value(linear, clip_min, clip_max)


def copyFile(source, target):

    try:
        copyfile(source, target)
    except IOError as e:
        print("Unable to copy file. %s" % e)
        exit(1)
    except:
        print("Unexpected error")
        exit(1)

    return


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_masks(root):
    maskFiles = []
    ms = os.listdir(root)
    for m in ms:
        if m.endswith(".png"):
            maskFiles.append(m)
    maskFiles.sort(key=lambda x: int(x[:-len(".png")]))
    for i, m in enumerate(maskFiles):
        maskFiles[i] = os.path.join(root, m)
    return maskFiles


def get_contours(mask):
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        length = []
        for i in range(len(contours)):
            length.append(len(contours[i]))
        length = np.array(length)
        argInx = np.argmax(length)
        contours = contours[argInx:argInx + 1]

    return contours


def get_all_the_dirs_with_filter(path, filter="C"):
    list = os.listdir(path)
    dirs = []
    for dir in sorted(list):
        if dir.startswith(filter):
            dirs.append(os.path.join(path, dir))

    return dirs


def remove_file(file):
    if os.path.exists(file):
        os.remove(file)


def remove_files(files):
    if isinstance(files, list) and not isinstance(files, str):
        for file in files:
            remove_file(file)
    else:
        remove_file(files)
