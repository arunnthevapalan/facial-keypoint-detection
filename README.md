# Facial Keypoint Detection
[![Udacity Computer Vision Nanodegree](http://tugan0329.bitbucket.io/imgs/github/cvnd.svg)](https://www.udacity.com/course/computer-vision-nanodegree--nd891)

This project was completed as part of Computer Vision Nanodegree offered by Udacity. The Project has been reviewed by Udacity and graded to meet sufficient requirements.


## Pre-requisites

The project was developed using python 3.6 and the latest version of PyTorch. Though GPU can be used, was not required.

- opencv-python==3.2.0.6
- matplotlib==2.1.1
- pandas==0.22.0
- numpy==1.12.1
- pillow>=6.2.0
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

## Approach

Step 1: Load and Visualize the Facial Keypoint Data
- Visualizng the dataset gives us an understanding of the data. This includes all the preprocessing and standardizing the train and test image data.
  
Step 2: Define the CNN Network architecture 
- Next step would be to experiment with different model architectures for with different layers such as Convolutional, Fully-Connected, Max-Pooling and Dropouts. This typically is defined in models.py file and can be imported in the pipeline.

Step 3: Train the CNN to predict facial keypoints
- Training a network includes trying out various loss functions, different optimizers, batch sizes and tuning the model with the best hyper-parameters.

Step 4: Dectect Faces in the images using Haar Cascades
- The trained neural network can now detect facial keypoints, from any image that includes faces. So before inferencing we need to use a Face Detector such as Haar Cascades to detect the faces.

Step 5: Predict Facial Keypoints and Complete the pipeline
- Put all the building blocks together and build a complete pipeline from loading data to inferencing the facial keypoints.

Step 6: Try applications of the project, like Fun Face Filters!

