from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--niter', type=int, default=200000, help='# of iter before validation')
        self.parser.add_argument('--niter_fid_val', type=int, default=50000, help='# of iter during validation')
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        self.parser.add_argument('--vgg_checkpoint_dir', type=str, default='extern/vgg/vgg_19', help='the checkpoint file for vgg_19')
        self.parser.add_argument('--train_data_dir', type=str, default='../TrainData/Teeth', help='your train data dir')
        self.parser.add_argument('--val_data_dir', type=str, default='../TestData/ImagePairs', help='your val data dir')
        self.parser.add_argument('--result_dir', type=str, default='results', help='dir to save generated images')
        self.parser.add_argument('--teeth_label_name', type=str, default='TeethEdgeUp.png,TeethEdgeDown.png', help='teeth geometry file names')

        # for discriminators
        self.parser.add_argument('--ndf', type=int, default=32, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--max_ndf', type=int, default=128, help='# of max discrim filters')

        # loss function parameters
        self.parser.add_argument('--kl_weight', type=float, default=1.0, help='weight for kl divergence loss')
        self.parser.add_argument('--content_weight', type=float, default=0.001, help='weight for the vgg perceptual loss')

        self.isTrain = True


