from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--test_data_dir', type=str, default='examples/cases_for_testing')
        self.parser.add_argument('--test_save_dir', type=str, default='results', help='save dir in each case dir')
        self.parser.add_argument('--test_step_dir', type=str, default='steps', help='step dir in each case dir')
        self.parser.add_argument('--teeth_label_name', type=str, default='TeethEdgeUpNew.png,TeethEdgeDownNew.png', help='teeth geometry file names')
        self.parser.add_argument('--use_best_FID', action='store_true', help='if specified, test the model with the best FID')
        self.isTrain = False
