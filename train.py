import tensorflow as tf
from models.solver import TSynNetSolver
from options.train_options import TrainOptions


def train():
    opt = TrainOptions().parse(save=True)
    with tf.Graph().as_default(), tf.Session() as sess:
        TSynNet = TSynNetSolver(sess, opt)
        TSynNet.train()


if __name__ == "__main__":
    train()