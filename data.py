from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf


def image_batch(image_paths, batch_size, load_size=286,
                crop_size=256, channels=3, shuffle=True,
                num_threads=4, min_after_dequeue=100,
                allow_smaller_final_batch=False):
    """For jpg and png files."""
    # queue and reader
    img_queue = tf.train.string_input_producer(image_paths, shuffle=shuffle)
    reader = tf.WholeFileReader()

    # preprocessing
    _, img = reader.read(img_queue)
    img = tf.image.decode_image(img, channels=channels)

    # img = tf.image.resize_images(img, [load_size, load_size])
    img = tf.random_crop(img, [load_size, load_size, channels])
    img = tf.cast(img, tf.float32) / 127.5 - 1

    # batch
    if shuffle:
        capacity = min_after_dequeue + (num_threads + 1) * batch_size
        img_batch = tf.train.shuffle_batch([img],
                                           batch_size=batch_size,
                                           capacity=capacity,
                                           min_after_dequeue=min_after_dequeue,
                                           num_threads=num_threads,
                                           allow_smaller_final_batch=allow_smaller_final_batch)
    else:
        img_batch = tf.train.batch([img], batch_size=batch_size,
                                   allow_smaller_final_batch=allow_smaller_final_batch)

    return img_batch, len(image_paths)


class ImageData:
    def __init__(self, session, image_paths, batch_size, **kwargs):
        self.sess = session
        self.img_batch, self.img_num = image_batch(image_paths, batch_size, **kwargs)

    def __len__(self):
        return self.img_num

    def batch_ops(self):
        return self.img_batch

    def batch(self):
        return self.sess.run(self.img_batch)
