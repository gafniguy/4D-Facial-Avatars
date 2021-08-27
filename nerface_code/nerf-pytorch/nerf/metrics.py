"""
Utils to compute metrics and track them across training.
"""

import argparse
import skimage
from skimage.measure import compare_ssim, compare_psnr
import os
from tqdm import tqdm
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.metrics import mean_squared_error
#from skimage.measure import _structural_similarity as ssim
import lpips
import torch
import scipy.ndimage


class ScalarMetric(object):
    def __init__(self):
        self.value = 0.0
        self.num_observations = 0.0
        self.aggregated_value = 0.0
        self.reset()

    def reset(self):
        self.value = []
        self.num_observations = 0.0
        self.aggregated_value = 0.0

    def __repr__(self):
        return str(self.peek())

    def update(self, x):
        self.aggregated_value += x
        self.num_observations += 1

    def peek(self):
        return self.aggregated_value / (
            self.num_observations if self.num_observations > 0 else 1
        )


def save_L2_image(im1, im2, outname):
    diff = np.linalg.norm(im1-im2, axis=2)
    fig = plt.figure()
    fig.set_size_inches((6.4,6.4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('jet')
    ax.imshow(diff, aspect='equal')
    plt.savefig(outname, dpi=80)
    plt.close(fig)

def ssim_single_image_pair(im1, im2):
    return compare_ssim(im1,im2, multichannel=True)

def psnr_single_image_pair(im1, im2):
    return compare_psnr(im1,im2)

def lpips_single_image_pair(im1,im2):
    im1_tensor = torch.FloatTensor(im1.astype('float32')).permute(2,0,1).unsqueeze(0)
    im2_tensor = torch.FloatTensor(im2.astype('float32')).permute(2,0,1).unsqueeze(0)

    with torch.no_grad():
        score = lpips_fn(im1_tensor,im2_tensor)
        score.cpu()
    """
    # TODO check their im loading
    #img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file)))  RGB image from [-1,1]
    #img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file)))

    """
    return score.item()



def two_folders(path_gt, path_generated):

    SSIM = ScalarMetric()
    PSNR = ScalarMetric()
    L1 = ScalarMetric()
    LPIPS = ScalarMetric()

    fout = os.path.join(path_generated, "metrics.txt")
    fo = open(fout, "w")

    filenames_gt = os.listdir(os.path.join(path_gt))
    filenames_path_generated = os.listdir(path_generated)
    filenames_path_generated = [file for file in filenames_path_generated
                                if os.path.isfile(os.path.join(path_generated,file))
                                and file.split('.')[1]=='png']
    filenames_gt.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    filenames_path_generated.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    # filenames_path_generated.sort(key = lambda item: (int(item.partition('_')[0])
    #                     if item[0].isdigit() else float('inf'), item))
    assert (len(filenames_path_generated) <= len(filenames_gt))
    N = len(filenames_path_generated)
    print("Detected %d images in GT folder          %s" % (len(filenames_gt), path_gt))
    print("Detected %d images in Generated folder   %s" % (len(filenames_path_generated), path_gt))

    for i in tqdm(range(0,N)):
        #frame_id = int(filenames_path_generated[i].split('_fake.png')[0])
        #if frame_id % skip != 0:
        #    continue
        im_real = np.array(PIL.Image.open(os.path.join(path_gt, filenames_gt[i])))/255
        im_generated = np.array(PIL.Image.open(os.path.join(path_generated, filenames_path_generated[i])))/255
        # Upscale 256 images from FOMM
        #im_generated = scipy.ndimage.zoom(im_generated, (2,2,1), order=1)

        assert im_real.shape == im_generated.shape

        # make L2 diff image
        save_L2_image(im_real,im_generated, os.path.join(path_generated,'L2',"%04d.png" %i))
        curr_ssim = ssim_single_image_pair(im_real, im_generated)
        curr_psnr = psnr_single_image_pair(im_real, im_generated)
        curr_L1 = np.mean(np.abs(im_real-im_generated))
        curr_lpips = lpips_single_image_pair(im_real, im_generated)

        L1.update(curr_L1)
        PSNR.update(curr_psnr)
        SSIM.update(curr_ssim)
        LPIPS.update(curr_lpips)

        fo.write(filenames_path_generated[i] + '   L1:  \t%5f \n' % (curr_L1) )
        fo.write(filenames_path_generated[i] + '   PSNR:\t%5f \n' % (curr_psnr) )
        fo.write(filenames_path_generated[i] + '   SSIM:\t%5f \n' % (curr_ssim) )
        fo.write(filenames_path_generated[i] + '   LPIPS:\t%5f\n\n' % (curr_lpips)  )

    fo.write('='*80)
    fo.write('\n Summary \n folder 1: %s \n folder 2: %s \n' % (path_gt, path_generated))
    fo.write('-'*80)
    fo.write("\n mean L1:\t%5f" % (L1.peek()))
    fo.write("\n mean PSNR:\t%5f" % (PSNR.peek()))
    fo.write("\n mean SSIM:\t%5f" % (SSIM.peek()))
    fo.write("\n mean LPIPS\t%5f\n" % (LPIPS.peek()))

    fo.close()

    print('='*80)
    print('\n Summary \n folder 1: %s \n folder 2: %s \n' % (path_gt, path_generated))
    print('-'*80)
    print("\n mean L1:\t%5f" % (L1.peek()))
    print("\n mean PSNR:\t%5f" % (PSNR.peek()))
    print("\n mean SSIM:\t%5f" % (SSIM.peek()))
    print("\n mean LPIPS\t%5f\n" % (LPIPS.peek()))




parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--gt_path', type = str, default='', help='directory to images of GT')
# parser.add_argument('--images_path', type = str,
#                     default='/mnt/raid/guygafni/neural_textures/results/scene24_3lvl_vgg_ngf64_ours_learn_blending_sparse/orig/test_40',
#                     help='directory with directory <images>' )
parser.add_argument('--images_path', type = str,
                   default='',
                   help='directory with generated images' )

parser.add_argument('--mode', type=str, default='folders', help='fofolders | images')
parser.add_argument('--skip', type=int, default=100, help='take only every n-th image')



if __name__ == "__main__":

    opt = parser.parse_args()

    lpips_fn = lpips.LPIPS(net='alex')
    os.makedirs(os.path.join(opt.images_path,'L2'),exist_ok=True)

    if opt.mode == 'folders':
        two_folders(opt.gt_path, opt.images_path)

    if opt.mode == 'images':
        ssim_single_image_pair(opt.gt_path, os.path.join(opt.images_path,'images'))
