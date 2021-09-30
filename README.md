## NerFACE: Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar Reconstruction [CVPR 2021 Oral Presentation]

*Guy Gafni<sup>1</sup>, Justus Thies<sup>1</sup>, Michael Zollhöfer<sup>2</sup>, Matthias Nießner<sup>1</sup>*

<sup>1</sup> Technichal University of Munich, <sup>2</sup>Facebook Reality Labs

![teaser](https://justusthies.github.io/posts/nerface/teaser.jpg)

ArXiv:  <a href="https://arxiv.org/pdf/2012.03065">PDF</a>,  <a href="https://arxiv.org/abs/2012.03065">abs</a>

Project Page & Video: <a href="https://gafniguy.github.io/4D-Facial-Avatars/">https://gafniguy.github.io/4D-Facial-Avatars/</a>


**If you find our work useful, please include the following citation:**


```
@InProceedings{Gafni_2021_CVPR,
    author    = {Gafni, Guy and Thies, Justus and Zollh{\"o}fer, Michael and Nie{\ss}ner, Matthias},
    title     = {Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar Reconstruction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {8649-8658}
}
```

**Code, Dataset and License**

Dataset sample <a  href="https://github.com/gafniguy/4D-Facial-Avatars/issues/2">available</a>.

The nerf code is heavily based on <a  href="https://github.com/krrish94/nerf-pytorch">this repo by Krishna Murthy</a>. Thank you!

** Code Structure **

Installation etc:
Originally the project used torch 1.7.1, but this should also run with torch 1.9.0 (cuda 11).
If you get any errors related to `torchsearchsorted`, ignore this module and don't bother installing it, and comment out its imports. Its functionality is impmlemented in pytorch.
These two are interchangeable:
```
    #inds = torchsearchsorted.searchsorted(cdf, u, side="right")  # needs compilationo of torchsearchsorted
    inds = torch.searchsorted(cdf.detach(), u, right=True)  # native to pytorch 
```

The main training and testing scripts are `train_transformed_rays.py` and `eval_transformed_rays.py`, respectively.
The training script expects a path to a config file, e.g.:
`python train_transformed_rays.py --config ./config/dave/dave_dvp_lcode_fixed_bg_512_paper_model.yml`
The eval script will also take a path to a model checkpoint and a folder to save the rendered images:
`python eval_transformed_rays.py --config ./config/dave/dave_dvp_lcode_fixed_bg_512_paper_model_teaser.yml --checkpoint /path/to/checkpoint/checkpoint400000.ckpt --savedir ./renders/dave_rendered_frames`

The config file must refer to a dataset to use in `dataset.basedir`. An example dataset has been provided above (see <a  href="https://github.com/gafniguy/4D-Facial-Avatars/issues/2">sample</a>). 

If you have your own video sequence including per frame tracking, you can see how I create the json's for training in the `real_to_nerf.py` file (main function). This does not include the code for tracking, which unfortunately I cannot publish. 


If you want access to our video sequences to run your methods on, don't hesitate to contact us [guy.gafni at tum.de]

The material in this repository is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

Code for the webpage is borrowed from the <a href="https://github.com/daveredrum/ScanRefer">ScanRefer project</a>.
