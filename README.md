# Computer-Vision
EECS 504- Winter 20


# Image Filtering
Pet Edge Detection
  1. Application of horizontal and vertical edge gradients and computation of total edge strength.
  2. Comparison of images for: Edges without blurring, Edges with Gaussian Filter, Edges with Box Filter.
  3. Computation of oriented edges in direction {theta} using horizontal and vertical gradients.
  

# Signal Processing
Image Blending
  1. Comparison of the output of a Gaussian filter through i) direct convolution in the spatial domain and ii) multiplication in the frequency domain.
  2. Constructing a Laplacian pyramids with 4 levels to reconstruct the original image.
  3. Blending two images: Using input of two images and a binary mask and produces the Laplacian pyramids with num levels levels for blending the two images.


# Motion Magnification and Texture Synthesis
Motion magnification in videos. 
Texture Synthesis: Method used for generating new textures from an initial sample texture.


# Backpropogation
Multi-layer perceptron
  1. Train a two-layer neural network to classify images using CIFAR-10. Network will have two layers, and a softmax layer to perform classication. Train the network
     to minimize a cross-entropy loss function (also known as softmax loss). The network uses a ReLU nonlinearity after the first fully connected layer.
  2. Setting up model hyperparameters (hidden dim, learning rate,lr decay, batch size) to get an accuracy above 45% on test data.
  

# Scene Recognition
  1. Train a CNN to solve the scene recognition problem, i.e., the problem of determining which scene category a picture depicts. 
  2. Train two neural networks, MiniVGG and MiniVGG-BN. MiniVGG is a smaller, simplified version of the VGG architecture, while MiniVGG-BN is identical to MiniVGG except that        there are batch normalization layers after each convolution layer. Dataset used- MiniPlaces
  
  
# Object Detection
Implement a single-stage object detector, based on YOLO v1 and v2. Unlike the (better-performing) R-CNN models, single-stage stage detectors predict bounding boxes and classes without explicitly cropping region proposals out of the image or feature map. This makes them significantly faster to run. Dataset used- PASCAL VOC


# Representation Learning
Implement two representation learning methods: an autoencoder and a recent constrastive learning method. Test the features that were learned by these models on a "downstream" recognition task, using the STL-10 dataset.


# Panoramic Stitching
Given two input images, we will "stitch" them together to create a simple panorama. To construct the image panorama, we will use concepts learned in class such as keypoint detection, local invariant descriptors, RANSAC, and perspective warping.
The panoramic stitching algorithm consists of four main steps:
  1. Detect keypoints and extract local invariant descriptors (using ORB) from two input images.
  2. Match the descriptors between the two images.
  3. Apply RANSAC to estimate a homography matrix between the extracted features.
  4. Apply a perspective transformation using the homography matrix to merge image into a panorama.
 
 
# Optical Flow
Implement the Lucas-Kanade (LK) optical flow algorithm for estimating dense motion between a pair of images. Since some of the motions might be too large for the Taylor approximation of the LK step, hence apply the algorithm in a coarse-to-fine manner.
