# Passenger-Action-Recognition
# About us
[![Robot Vision Lab](https://img.shields.io/badge/Robot%20Vision-Lab-brightgreen.svg?style=flat-square)](https://vision.ee.ccu.edu.tw/index.php)
[![Department of Electrical Engineering](https://img.shields.io/badge/Department%20of-Electrical_Engineering-blue.svg?style=flat-square)](http://www.ee.ccu.edu.tw/main.php)
[![National Chung Cheng University](https://img.shields.io/badge/National%20-Chung_Cheng_University-blue.svg?style=flat-square)](https://www.ccu.edu.tw/eng/index.php)

# Table of Contents
- [Abstract](#abstract)
- [Usage](#usage)
	- [Environment](#environment)
	- [How_to_use](#how_to_use)
	- [Dataset](#dataset)
- [Related_Work](#related-work)
- [Maintainers](#maintainers)
- [Contact](#contact)

# Abstract
	
Through action recognition, we can know what behavior the person in the image. 

This paper [(Passenger Detection and Pose Recognition using Deep Neural Networks)](https://ieeexplore.ieee.org/document/9575797) proposes a action recognize method based on deep learning combined with human detector to implement a action recognition system for passengers in public transportation vehicles, and proposes an architecture for passenger counting. 

1. We set up two cameras in the environment and use 2D and 3D CNN to recognize static poses and dynamic actions. 

	- Posture Recognition (2D CNN) : Recognize postures that do not need to consider time information. 

	- Action Recognition (3D CNN) : Recognize the continuous motion of passengers. 

2. In order to understand the number of passengers in the environment, we use two cameras to count passengers that based on the detection results of detector and association method. 

3. To achieve the goal, we built a neural network to solve the double counting problem caused by the same person appearing on two cameras.







# Usage

### Environment
```
python=3.7
pytorch=1.6.0
numpy=1.19
opencv=3.4.2
CUDA 10.0
cuDNN 7.4.1 
```
Use this to create environments
refer to [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#id2)
```
conda env create -f environment.yml
```
	
### How_to_use
- [bb_match](https://github.com/richardkuo1999/Passenger-Action-Recognition/wiki/bb_match)  : Using camera1 bounding box information to predict camera2 bounding box place.
- [pose_classification](https://github.com/richardkuo1999/Passenger-Action-Recognition/wiki/pose_classification) : Using a single-frame approach to classify the action, we have two categories:  seated and standed.
- [action_classification](https://github.com/richardkuo1999/Passenger-Action-Recognition/wiki/action_classification) : Using a multi-frame approach to classify the action for temporal movements, we have four categories: sitting, standing up, seated and standed.
-  [action_detection](https://github.com/richardkuo1999/Passenger-Action-Recognition/wiki/action_detection) : Combination multi-frame, single-frame and bounding box to classify the action and calculate the number of people.

### Dataset

pose_classification dataset : Using [minbus](https://vision.ee.ccu.edu.tw/bus/WinBus.rar),[Bus look down](https://vision.ee.ccu.edu.tw/bus/Chiayi_Bus_look_down.rar),[Bus side view](https://vision.ee.ccu.edu.tw/bus/Chiayi_Bus_side_view.rar).

bb_match dataset : Using [minbus](https://vision.ee.ccu.edu.tw/bus/WinBus.rar).

action_classification dataset : Using [action_frames](https://vision.ee.ccu.edu.tw/bus/action_frames.rar).

action_detection dataset : Using [pose_classification Dataset ](https://github.com/richardkuo1999/Passenger-Action-Recognition/wiki/pose_classification) and [action_classification Dataset](https://github.com/richardkuo1999/Passenger-Action-Recognition/wiki/action_classification).

# Related Work
1. [People Detection and Pose Classification Inside a Moving Train Using Computer Vision](https://core.ac.uk/download/pdf/288501396.pdf)
3. [Human activity monitoring for falling detection. A realistic framework](https://ieeexplore.ieee.org/document/7743617)
4. [Dual Viewpoint Passenger State Classification Using 3D CNNs](https://ieeexplore.ieee.org/document/8500564)
5. [DeepPose: Human Pose Estimation via Deep Neural Networks](https://ieeexplore.ieee.org/document/6909610)
6. [Human Pose Estimation Using Convolutional Neural Networks](https://ieeexplore.ieee.org/document/8701267)


# Maintainers
[@richardkuo1999](https://github.com/Richardkuo1999)



# Contact
For Passenger-Action-Recognition bugs and feature requests please visit [GitHub Issues](https://github.com/richardkuo1999/Passenger-Action-Recognition/issues). For business inquiries or professional support requests please visit https://vision.ee.ccu.edu.tw/index.php.
