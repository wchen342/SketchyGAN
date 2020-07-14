SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis
=====================================

Code for ["SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis"](https://arxiv.org/abs/1801.02753).


## Prerequisites

- Python 3, NumPy, SciPy, OpenCV 3
- Tensorflow(>=1.7.0). Tensorflow 2.0 is not supported.
- A recent NVIDIA GPU


## Preparations

- The path to data files needs to be specified in `input_pipeline.py`. See below for detailed information on data files.
- You need to download ["Inception-V4 model"](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz), unzip it and put the checkpoint under `inception_v4_model`.


## Dataset
~~Pre-built tfrecord files are available for out of the box training.~~
- ~~Files for the Sketchy Database can be found [here](https://gtvault-my.sharepoint.com/:f:/g/personal/wchen342_gatech_edu/EtKmg1alDNdIl09WcvtJp_cBFs_7td3wKnb5FUcWZswEmw?e=eBGO6G).~~
- ~~Files for Augmented Sketchy(i.e. flickr images+edge maps), resized to 256x256 regardless of original aspect ratios, can be found [here](https://gtvault-my.sharepoint.com/:f:/g/personal/wchen342_gatech_edu/EmF7KlhqZ8ZPnpzbTIMDKBoBcjMrezh3X2eS1P_KtWiGCQ?e=BJhFPF).~~

**Note**: The webite hosting the dataset is no longer available. Please use the script under `data_processing` folder to crawl your own images.

If you want to build tfrecord files from images, run `flickr_to_tfrecord.py` or `sketchy_to_tfrecord.py` for the respective dataset.

If you wish to get the image files:
- The Sketchy Database can be found [here](http://sketchy.eye.gatech.edu/).
- Use `extract_images.py` under `data_processing` to extract images from tfrecord files. You need to specify input and output paths. The extracted images will be sorted by class names.
- The dataset I used is no longer availabe due to its large size. You can crawl your own images and run through `edge_detection/batch_hed.py` -> `edge_detection/PostprocessHED.m` -> ``flickr_to_tfrecord.py` to create your own dataset.
- ~~Please contact me if you need the original (not resized) Flickr images, since they are too large to upload to any online space.~~


## Configurations

The model can be trained out of the box, by running `main_single.py`. But there are several places you can change configurations:

- Commandline options in `main_single.py`
- Some global options in `config.py`
- Activation/Normalization functions in `models_mru.py`


## Model

- The model will be saved periodically. If you wish to resume, just use commandline switch `resume_from`.
- If you wish to test the model, change `mode` from `train` to `test` and fill in `resume_from`.


## Citation

If you use our work for your research, please cite our paper
```
@InProceedings{Chen_2018_CVPR,
author = {Chen, Wengling and Hays, James},
title = {SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```


## Credits
- Inception-V4 and VGG16 code by Tensorflow Authors.
- Tensorflow implementation of Spectral Normalization by [minhnhat93](https://github.com/minhnhat93/tf-SNDCGAN)
- [Improved WGAN](https://github.com/igul222/improved_wgan_training)
