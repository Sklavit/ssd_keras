[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
# A port of [SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd) to [Keras](https://keras.io) framework.
For more details, please refer to [arXiv paper](http://arxiv.org/abs/1512.02325).
For forward pass for 300x300 model, please, follow `SSD.ipynb` for examples. For training procedure for 300x300 model, please, follow `SSD_training.ipynb` for examples. Moreover, in `testing_utils` folder there is a useful script to test `SSD` on video or on camera input.

Weights are ported from the original models and are available [here](https://mega.nz/#F!7RowVLCL!q3cEVRK9jyOSB9el3SssIA). You need `weights_SSD300.hdf5`, `weights_300x300_old.hdf5` is for the old version of architecture with 3x3 convolution for `pool6`.

This code was tested with `Keras` v1.2.2, `Tensorflow` v1.0.0, `OpenCV` v3.1.0-dev\
Also support newest `Keras` v2.0.1 (using ssd_v2.py)

# Installation guide for Windows 10
Project can be ran with `conda`.
Install `miniconda` or `anaconda`.
Create environment:
```commandline
conda create -n py36cpu python=3.6
``` 

Install packages:
```commandline
conda install numpy scipy matplotlib
conda install keras opencv imageio -c conda-forge
```

Intall tensorflow for cpu:
```commandline
conda install tensorflow
```
or GPU:
```commandline
conda install tensorflow-gpu
```

To use Cython-based training scripts install Cython:
```commandline
conda install cython
```

To use Jupyter Notebook install
```commandline
conda install jupyter
```

# Training
## VOC2007 dataset
You need to download The PASCAL Visual Object Classes Challenge 2007 training dataset:
[http://host.robots.ox.ac.uk/pascal/VOC/voc2007#devkit]
[Download the training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) 
(450MB tar file)
You should unzip it into project root as `VOCdevkit` directory.
Then you should run `PASCAL_VOC/get_data_from_XML.py` to create `VOC2007.pkl` with annotations.
Then you could run training Keras 2. implementation of SSD by `run_ssd_trainer.py`.
NOTE: you will need `cython` installed for this.


 
