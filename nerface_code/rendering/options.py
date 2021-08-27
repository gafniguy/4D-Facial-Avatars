import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Options():
    def __init__(self):
        self.initialized = False
    '''
        SUBFOLDER_NAME = "full_body"
    
        GENERATE_SPLATTING = 0
        RENDER_COLOR_IMG = 1
    
        TRAIN = True
    
        IM_SIZE = 256
    
        SAVE_CAM_COORDS = True
    
        SAVE_WORLD_COORDS = not SAVE_CAM_COORDS
        SHOW_MESH = False or RENDER_COLOR_IMG  # need to show in order to generate openGL context
    if TRAIN:
        NUM_VIEWS = 2000
        SAMPLING_TYPE = "LATTICE"
        SIMPLIFIED_MESH = GENERATE_SPLATTING
        subfolder_save = os.path.join(SUBFOLDER_NAME, 'train')

    else:
        NUM_VIEWS = 200
        SAMPLING_TYPE = "SPIRAL"
        SIMPLIFIED_MESH = GENERATE_SPLATTING
        subfolder_save = os.path.join(SUBFOLDER_NAME, 'test')
    '''

    def initialize(self, parser):

        parser.add_argument('--name', default='FLAME_sample' ,help='name of model')
        parser.add_argument('--folder_name', type=str, default='', help='name of folder to look for model')
        parser.add_argument('--target_name', default='debug_color' ,help='name of folder to save')

        parser.add_argument('--n_views_train', type=int, default=200, help='max num of points to sample from for train set')
        parser.add_argument('--n_views', type=int, default=250, help='max num of points to sample')
        parser.add_argument('--n_views_test', type=int, default=200, help='max num of points to sample from for test set')
        parser.add_argument('--train', action='store_true', default=False, help='generate training set')
        parser.add_argument('--test',  action='store_true', default=False, help='generate test set')
        parser.add_argument('--render', action='store_true', default=False, help='generate rendering')
        #parser.add_argument('--save_cam_space_coords', type=bool, default=False, help='generate rendering')
        #parser.add_argument('--save_world_space_coords', type=bool, default=True, help='generate rendering')
        parser.add_argument('--im_size', type=int, default=256, help='canvas size')
        #parser.add_argument("--nice", type=str2bool, nargs='?',const=True, default=False, help="Activate nice mode.")
        parser.add_argument('--anti_alias', type = int, default=1, help='Anti Aliasing for RGB render. Renders larger image then downsamples. SLOWS poerformance!')
        parser.add_argument('--background_plane', action='store_true', default=False, help='Render a background plane behind head')

        self.initialized = True
        #self.parser = parser
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            self.parser = parser

        # get the basic options
        opt, _ = parser.parse_known_args()

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):

        opt = self.gather_options()

        self.print_options(opt)

        self.opt = opt
        return self.opt
