# Facial Keypoint Detection
[![Udacity Computer Vision Nanodegree](http://tugan0329.bitbucket.io/imgs/github/cvnd.svg)](https://www.udacity.com/course/computer-vision-nanodegree--nd891)

This project was completed as part of Computer Vision Nanodegree offered by Udacity. The Project has been reviewed by Udacity and graded to meet sufficient requirements.


## Pre-requisites

The project was developed using python 3.6 and the latest version of PyTorch. Though GPU can be used, was not required.

- opencv-python==3.2.0.6
- matplotlib==2.1.1
- pandas==0.22.0
- numpy==1.12.1
- pillow==5.0.0
- scipy==1.0.0
- torch>=0.4.0
- torchvision>=0.2.0
- jupyterlab

Installation with pip:

```bash
pip install -r requirements.txt
```

## Dataset

This facial keypoints dataset consists of 5770 color images. 

- 3462 of these images are training images,to create a model to predict keypoints.
- 2308 are test images, which will be used to test the accuracy of the model.

*Note: All images are extracted from the [YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/) which includes videos of people in YouTube videos.*
