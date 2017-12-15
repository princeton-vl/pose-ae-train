# Associative Embedding: Training Code

Multi-person pose estimation with PyTorch based on:

**Associative Embedding: End-to-end Learning for Joint Detection and Grouping.**
[Alejandro Newell](http://www-personal.umich.edu/~alnewell/), Zhiao Huang, and [Jia Deng](http://web.eecs.umich.edu/~jiadeng/). *Neural Information Processing Systems (NIPS)*, 2017. 

(A pretrained model in TensorFlow is also available here: https://github.com/umich-vl/pose-ae-demo)

## Getting Started

This repository provides everything necessary to train and evaluate a multi-person pose estimation model on COCO keypoints. If you plan on training your own model from scratch, we highly recommend using multiple GPUs. We also provide a [pretrained model](https://umich.box.com/s/nptgu9r46xudtjn7o8viksryiztdnqwt).

Requirements:

- Python 3 (code has been tested on Python 3.6)
- PyTorch
- CUDA and cuDNN
- Python packages (not exhaustive): opencv-python, cffi, munkres, tqdm, json

Before using the repository there are a couple of setup steps:

First, you must compile the C implementation of the associative embedding loss. Go to ```extensions/AE/``` and call ```python build.py install```. If you run into errors with missing include files for CUDA, this can be addressed by first calling ```export CPATH=/path/to/cuda/include```.

Next, set up the COCO dataset. You can download it from [here](http://cocodataset.org/#download), and update the paths in ```data/coco_pose/ref.py``` to the correct directories for both images and annotations. After that, make sure to install the COCO PythonAPI from [here](https://github.com/cocodataset/cocoapi).

You should be all set after that! For reference, the code is organized as follows:
- ```data/```: data loading and data augmentation code
- ```models/```: network architecture definitions
- ```task/```: task-specific functions and training configuration
- ```utils/```: image processing code and miscellaneous helper functions
- ```extensions/```: custom C code that needs to be compiled
- ```train.py```: code for model training
- ```test.py```: code for model evaluation

## Training and Testing

To train a network, call:

```python train.py -e test_run_001``` (```-e,--exp``` allows you to specify an experiment name)

To continue an experiment where it left off, you can call:

```python train.py -c test_run_001```

All training hyperparameters are defined in ```task/pose.py```, and you can modify ```__config__``` to test different options. It is likely you will have to change the batchsize to accommodate the number of GPUs you have available.

Once a model has been trained, you can evaluate it with:

```python test.py -c test_run_001 -m [single|multi]```

The argument ```-m,--mode``` indicates whether to do single- or multi-scale evaluation. Single scale evaluation is faster, but multiscale evaluation is responsible for large gains in performance. You can edit ```test.py``` to evaluate at more scales for further improvements.

#### Training/Validation split

This repository includes a predefined training/validation split that we use in our experiments, ```data/coco_pose/valid_id``` lists all images used for validation.

#### Pretrained model

To evaluate on the pretrained model, you can download it from [here](https://umich.box.com/s/nptgu9r46xudtjn7o8viksryiztdnqwt) and unpack the file into ```exp/```. Then call:

```python test.py -c pretrained -m single```

That should return a mAP of about 0.59 for single scale evaluation, and .66 for multiscale (performance can be improved further by evaluating at more than the default 3 scales). Results will not necessarily be the same on the COCO test sets.

To use this model for your own images, you can set up code to pass your own data to the ```multiperson``` function in ```test.py```.
