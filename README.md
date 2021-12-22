# Passenger-Action-Recognition
# About us
[![Robot Vision Lab](https://img.shields.io/badge/Robot%20Vision-Lab-brightgreen.svg?style=flat-square)](https://vision.ee.ccu.edu.tw/index.php)
[![Department of Electrical Engineering](https://img.shields.io/badge/Department%20of-Electrical_Engineering-blue.svg?style=flat-square)](http://www.ee.ccu.edu.tw/main.php)
[![National Chung Cheng University](https://img.shields.io/badge/National%20-Chung_Cheng_University-blue.svg?style=flat-square)](https://www.ccu.edu.tw/eng/index.php)

# Table of Contents
- [Abstract](#Abstract)
- [Usage](#usage)
	- [Environment](#environment)
	- [Dataset](#dataset)
	- [How_to_use](#how_to_use)
- [Related Efforts](#related-efforts)
- [Maintainers](#maintainers)
- [License](#license)
# Abstract
Through action recognition, we can know what behavior the person in the image. 

This paper [(Passenger Detection and Pose Recognition using Deep Neural Networks)](https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/login?o=dnclcdr&s=id=%22108CCU00442053%22.&searchmode=basic) proposes a action recognize method based on deep learning combined with human detector to implement a action recognition system for passengers in public transportation vehicles, and proposes an architecture for passenger counting. 

1. We set up two cameras in the environment and use 2D and 3D CNN to recognize static poses and dynamic actions. 

>Posture recognition(2D CNN) : recognize postures that do not need to consider time information. 

>Action recognition(3D CNN) : recognize the continuous motion of passengers. 

2. In order to understand the number of passengers in the environment, we use two cameras to count passengers that based on the detection results of detector and association method. 

3. To achieve the goal, we built a neural network to solve the double counting problem caused by the same person appearing on two cameras.


# Usage
### Environment
coming soon
### Dataset
coming soon
### How_to_use
coming soon
# Related Efforts
1. [People Detection and Pose Classification Inside a Moving Train Using Computer Vision](https://core.ac.uk/download/pdf/288501396.pdf)
	- human detection：Histogram of Oriented Gradients (HOG).
	- Action recognitiontrain： Support Vector Machines (SVM). 

2. [Human activity monitoring for falling detection. A realistic framework](https://ieeexplore.ieee.org/document/7743617)
	- human detection：Histogram of Oriented Gradients (HOG).
	- Action recognitiontrain： Action Bank. 
3. [Dual Viewpoint Passenger State Classification Using 3D CNNs](https://ieeexplore.ieee.org/document/8500564)

4. [DeepPose: Human Pose Estimation via Deep Neural Networks](https://ieeexplore.ieee.org/document/6909610)

5. [Human Pose Estimation Using Convolutional Neural Networks](https://ieeexplore.ieee.org/document/8701267)
# Maintainers
[@richardkuo1999](https://github.com/Richardkuo1999)

# License
