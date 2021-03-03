import argparse
import os
from util import util


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        # network arch
        self.parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in the first conv layer')
        self.parser.add_argument('--max_ngf', type=int, default=128, help='# of max gen filters')
        self.parser.add_argument('--n_latent', type=int, default=2, help='the size of the latent space, e.g., 0 for 1X1, 1 for 2X2, 2 for 4X4')

        self.parser.add_argument('--use_gan', action='store_true', help='if specified, train the model with gan')
        self.parser.add_argument('--use_ext', action='store_true', help='if specified, add background info at the begining of the generator')
        self.parser.add_argument('--use_skip', action='store_true', help='if specified, the generator will be refashioned into a skip generator')
        self.parser.add_argument('--use_style_cont', action='store_true', help='if specified, the style modulation will be added into the generator')

        # data loader
        self.parser.add_argument('--image_name', type=str, default='Img.jpg', help='original image file name')
        self.parser.add_argument('--mouth_label_name', type=str, default='MouthMask.png', help='mouth mask file name')
        self.parser.add_argument('--case_marker', type=str, default='C', help='the marker for each case dir')
        self.parser.add_argument('--load_size', type=int, default=256, help='size of loaded image')

        # experiment
        self.parser.add_argument('--name', type=str, default='TSynNet', help='name of the experiment. It decides where to store the results and models')
        self.parser.add_argument('--expr_dir', type=str, default='checkpoints', help='name of the dir to save all the experiments')
        self.initialized = True

    def parse_str_names(self, names):
        str_names = names.split(',')
        names = []
        for name in str_names:
            names.append(name)
        return names

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        # get the teeth-edge-map names and mouth-mask name for data loader
        self.opt.teeth_label_name = self.parse_str_names(self.opt.teeth_label_name)
        self.opt.mouth_label_name = self.parse_str_names(self.opt.mouth_label_name)
        self.opt.label_nc = len(self.opt.teeth_label_name) + len(self.opt.mouth_label_name)

        # override the experiment name to get the working path
        if self.opt.use_gan:
            self.opt.name += "_gan"
        if self.opt.use_ext:
            self.opt.name += "_ext"
        if self.opt.use_skip:
            self.opt.name += "_skip"
        if self.opt.use_style_cont:
            self.opt.name += "_scont"
        self.opt.name += "_{}".format(self.opt.n_latent)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.expr_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
