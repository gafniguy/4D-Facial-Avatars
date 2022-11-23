## NeRFace: Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar Reconstruction [CVPR 2021 Oral Presentation]

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

**Dataset and License**

Dataset is <a  href="https://syncandshare.lrz.de/getlink/fiBTHis1fS8Zxqd55XCAjjG8/nerface_dataset.zip">available</a>.
Please do not use it for commercial use and respect the license attached within the zip file. (The material in this repository is licensed under an Attribution-NonCommercial-ShareAlike 4.0 International license). 

If you make use of this dataset or code, please cite our paper. 
MIT License applies for the code.

**Code Structure**
The nerf code is heavily based on <a  href="https://github.com/krrish94/nerf-pytorch">this repo by Krishna Murthy</a>. Thank you! 

Installation etc:
Originally the project used torch 1.7.1, but this should also run with torch 1.9.0 (cuda 11).
If you get any errors related to `torchsearchsorted`, ignore this module and don't bother installing it, and comment out its imports. Its functionality is impmlemented in pytorch.
These two are interchangeable:
```
    #inds = torchsearchsorted.searchsorted(cdf, u, side="right")  # needs compilationo of torchsearchsorted
    inds = torch.searchsorted(cdf.detach(), u, right=True)  # native to pytorch 
```

The main training and testing scripts are `train_transformed_rays.py` and `eval_transformed_rays.py`, respectively. They are in the main working folder which is in `nerface_code/nerf-pytorch/` 

The training script expects a path to a config file, e.g.:

`python train_transformed_rays.py --config ./path_to_data/person_1/person_1_config.yml `

The eval script will also take a path to a model checkpoint and a folder to save the rendered images:

`python eval_transformed_rays.py --config ./path_to_data/person_1/person_1_config.yml --checkpoint /path/to/checkpoint/checkpoint400000.ckpt --savedir ./renders/person_1_rendered_frames`

The config file must refer to a dataset to use in `dataset.basedir`. Download the dataset from the .zip shared above, and place it in the nerf-pytorch directory. 

If you have your own video sequence including per frame tracking, you can see how I create the json's for training in the `real_to_nerf.py` file (main function). This does not include the code for tracking, which unfortunately I cannot publish. 


Don't hesitate to contact [guy.gafni at tum.de] for additional questions, or open an issue here.


Code for the webpage is borrowed from the <a href="https://github.com/daveredrum/ScanRefer">ScanRefer project</a>.
