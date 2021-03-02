import tensorflow as tf
import math

slim = tf.contrib.slim


class Augmentor(object):

    def __init__(self,
                 img_size=256,
                 img_rotate_prob=0.5, img_rotate_range=0.15,
                 img_hsv_prob=0.5,
                 h_shift=0.05,
                 s_shift=0.05, s_scale=0.05,
                 v_shift=0.05, v_scale=0.05,
                 horizon_flip=True,
                 label_clip=True
                 ):

        self.img_size = img_size
        self.img_rotate_prob = img_rotate_prob
        self.img_rotate_range = img_rotate_range
        self.img_hsv_prob = img_hsv_prob
        self.h_shift = h_shift
        self.s_shift = s_shift
        self.s_scale = s_scale
        self.v_shift = v_shift
        self.v_scale = v_scale
        self.horizon_flip = horizon_flip
        self.label_clip = label_clip

    def __call__(self, imgs, labels):

        i_shape = imgs.get_shape()
        l_shape = labels.get_shape()

        if self.img_rotate_prob > 0:
            imgs, labels = self.random_rotate(imgs, labels)

        if self.img_hsv_prob > 0:
            imgs = self.random_hsv(imgs)

        if self.horizon_flip:
            imgs, labels = self.random_flip(imgs, labels)

        imgs.set_shape(i_shape)
        imgs = tf.stop_gradient(tf.to_float(imgs))

        labels.set_shape(l_shape)
        labels = tf.stop_gradient(tf.to_float(labels))

        return imgs, labels

    def random_hsv(self, imgs):

        tf.assert_equal(len(imgs.get_shape()), 4)

        def tweak_in_hsv(img):
            shape = tf.shape(img)

            img = tf.image.rgb_to_hsv(img)

            hue_shift = tf.random_uniform(shape=[1], minval=-self.h_shift, maxval=self.h_shift)
            sat_shift = tf.random_uniform(shape=[1], minval=-self.s_shift, maxval=self.s_shift)
            val_shift = tf.random_uniform(shape=[1], minval=-self.v_shift, maxval=self.v_shift)

            hue_scale = tf.ones(shape=[1])
            sat_scale = tf.random_uniform([1], 1. / (1. + self.s_scale), 1. + self.s_scale)
            val_scale = tf.random_uniform([1], 1. / (1. + self.v_scale), 1. + self.v_scale)

            scale = tf.concat([hue_scale, sat_scale, val_scale], axis=-1)
            shift = tf.concat([hue_shift, sat_shift, val_shift], axis=-1)

            img = img * scale + shift
            img = tf.clip_by_value(img, 0.01, 0.99)  # when scale and shift, we need to check
            img = tf.image.hsv_to_rgb(img)

            return img

        if_tweak = tf.less(tf.random_uniform([], 0, 1.0), self.img_hsv_prob)

        return tf.cond(if_tweak, lambda: tf.map_fn(tweak_in_hsv, imgs), lambda: imgs)

    def random_rotate(self, imgs, labels):

        tf.assert_equal(len(imgs.get_shape()), 4)

        def rotate(pair):
            img = pair[0]
            label = pair[1]

            shape = tf.shape(img)
            height = shape[0]
            width = shape[1]

            img = tf.pad(img, [[height // 4, height // 4], [width // 4, width // 4], [0, 0]], "SYMMETRIC")
            label = tf.pad(label, [[height // 4, height // 4], [width // 4, width // 4], [0, 0]], "SYMMETRIC")

            h = tf.cast(height + height // 4 * 2, tf.float32)
            w = tf.cast(width + width // 4 * 2, tf.float32)

            angle_rad = self.img_rotate_range * math.pi / 2.0
            angle = tf.random_uniform([], -angle_rad, angle_rad)
            transform = tf.contrib.image.angles_to_projective_transforms(angle, h, w)

            img = tf.contrib.image.transform(img, transform, interpolation='BILINEAR')
            img = tf.image.crop_to_bounding_box(img, height // 4, width // 4, height, width)

            label = tf.contrib.image.transform(label, transform, interpolation='BILINEAR')
            label = tf.image.crop_to_bounding_box(label, height // 4, width // 4, height, width)

            if self.label_clip:
                label = tf.to_float(label > 0.6)

            return (img, label)

        if_tweak = tf.less(tf.random_uniform([], 0, 1.0), self.img_rotate_prob)

        return tf.cond(if_tweak,
                       lambda: tf.map_fn(rotate, (imgs, labels), dtype=(tf.float32, tf.float32)),
                       lambda: (imgs, labels))

    def random_flip(self, imgs, labels):

        if_tweak = tf.less(tf.random_uniform([], 0, 1.0), 0.5)
        return tf.cond(if_tweak,
                       lambda: (imgs[:, :, ::-1, :], labels[:, :, ::-1, :]),
                       lambda: (imgs, labels))


