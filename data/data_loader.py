import os
import cv2
import random
import numpy as np
import copy
from queue import Queue
from threading import Thread
from util import util


def get_gt_labels(path, names, load_size=256, thresh=50, mdilate=True):

    out = []
    for name in names:
        b = cv2.imread(os.path.join(path, name))
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(b, thresh, 1, type=0)  # -> range in [0, 1]

        h, w = b.shape[0], b.shape[1]
        if h != load_size or w != load_size:
            b = cv2.resize(b, (load_size, load_size), interpolation=cv2.INTER_NEAREST)

        # Mouth Mask should be dilated to remove the teeth region completely
        if name.startswith("M") and mdilate:
            b = cv2.dilate(b, np.ones((7, 7), dtype=b.dtype))

        out.append(np.expand_dims(b, -1))

    return np.concatenate(out, axis=-1).astype(np.float32)     # H * W * n


def get_gt_image(path, name, load_size=256):

    img = cv2.imread(os.path.join(path, name)).astype(np.float32)

    h, w = img.shape[0], img.shape[1]
    if h != load_size or w != load_size:
        img = cv2.resize(img, (load_size, load_size))

    img = img[:, :, ::-1] / 255.0

    return img  # H * W * 3 -> RGB & range in [0, 1]


class base_loader:

    def __init__(self, dir, opt, random_shuffle=True):

        self.opt = opt
        self.dirs = util.get_all_the_dirs_with_filter(dir, opt.case_marker)
        self.load_size = opt.load_size
        self.batch_size = opt.batch_size
        self.random_shuffle = random_shuffle

        self.corpus = []
        self.nums = 0

        self.generate_corpus()

    def generate_corpus(self):

        self.corpus = copy.copy(self.dirs)
        random.shuffle(self.corpus) if self.random_shuffle else None
        self.nums = len(self.corpus)
        self.current_pos = 0

    def start_generation(self):

        if self.current_pos + self.batch_size > self.nums:
            self.current_pos = self.nums - self.batch_size
        return self.corpus[self.current_pos:self.current_pos + self.batch_size]

    def end_generation(self):

        self.current_pos += self.batch_size
        if self.current_pos >= self.nums:
            self.current_pos = 0
            random.shuffle(self.corpus) if self.random_shuffle else None

    def get_iters(self):
        return self.nums // self.batch_size

    def __len__(self):
        return self.nums


class data_loader(base_loader):

    def __init__(self, dir, opt, random_shuffle=True):
        super().__init__(dir, opt, random_shuffle)

        self.image_name = opt.image_name
        self.teeth_label_name = opt.teeth_label_name
        self.mouth_label_name = opt.mouth_label_name
        self.queue = Queue()
        self.thread = None
        self.load_thread()

    def load_thread(self):
        if self.thread:
            self.thread.join()

        self.thread = Thread(target=self._get_one_batch_data)
        self.thread.start()

    def _get_one_batch_data(self):

        samples = self.start_generation()

        input_images = []  # N * 256 * 256 * 3
        teeth_labels = []  # N * 256 * 256 * 3
        mouth_labels = []  # N * 256 * 256 * 1

        for d in samples:
            input_images.append(np.expand_dims(get_gt_image(d, self.image_name, self.load_size), 0))
            teeth_labels.append(np.expand_dims(get_gt_labels(d, self.teeth_label_name, self.load_size), 0))
            mouth_labels.append(np.expand_dims(get_gt_labels(d, self.mouth_label_name, self.load_size), 0))

        input_images = np.concatenate(input_images, axis=0)
        teeth_labels = np.concatenate(teeth_labels, axis=0)
        mouth_labels = np.concatenate(mouth_labels, axis=0)

        self.end_generation()  # be care of this place

        self.queue.put((input_images, teeth_labels, mouth_labels))#, projection_indices

    def get_one_batch_data(self):
        if self.queue.empty():
            pass
        self.load_thread()
        return self.queue.get()


class data_loader_test(base_loader):

    def __init__(self, dir, opt):
        super().__init__(dir, opt, False)

        self.image_name = opt.image_name
        self.teeth_label_name = opt.teeth_label_name
        self.mouth_label_name = opt.mouth_label_name

    def get_one_case(self):

        for case in self.corpus:
            print(case)
            input_images = np.expand_dims(get_gt_image(case, self.image_name, self.load_size), 0)
            mouth_labels = np.expand_dims(get_gt_labels(case, self.mouth_label_name, self.load_size), 0)

            step_dir = os.path.join(case, self.opt.test_step_dir)
            steps = os.listdir(step_dir)
            teeth_labels = {}
            for step in steps:
                teeth_labels[step] = np.expand_dims(get_gt_labels(os.path.join(step_dir, step), self.teeth_label_name, self.load_size), 0)

            yield input_images, teeth_labels, mouth_labels, case





