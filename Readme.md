SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis
=====================================

Code for ["SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis"](https://arxiv.org/abs/1801.02753).


## Prerequisites

- Python, NumPy, SciPy
- Tensorflow(>=1.4.0)
- A recent NVIDIA GPU


## Preparations

- The path to data files needs to be specified in `input_pipeline.py`. The dataset will be released shortly.
- You need to download ["Inception-V4 model"](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz), unzip it and put the checkpoint under `inception_v4_model`.


## Configurations

The model can be trained out of the box, by running `main_single.py`. But there are several places you can change configurations:

- Commandline options in `main_single.py`
- Some global options in `config.py`
- Activation/Normalization functions in `models_mru.py`


## Others

- The model will be saved periodically. If you wish to resume, just use commandline switch `resume_from`.
- If you wish to test the model, change `mode` from `train` to `test` and fill in `resume_from`.