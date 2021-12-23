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
- [Related_Work](#related-work)
- [Maintainers](#maintainers)
- [Contact](#contact)
# Abstract
Through action recognition, we can know what behavior the person in the image. 

This paper [(Passenger Detection and Pose Recognition using Deep Neural Networks)](https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/login?o=dnclcdr&s=id=%22108CCU00442053%22.&searchmode=basic) proposes a action recognize method based on deep learning combined with human detector to implement a action recognition system for passengers in public transportation vehicles, and proposes an architecture for passenger counting. 

1. We set up two cameras in the environment and use 2D and 3D CNN to recognize static poses and dynamic actions. 

	- Posture Recognition (2D CNN) : Recognize postures that do not need to consider time information. 

	- Action Recognition (3D CNN) : Recognize the continuous motion of passengers. 

2. In order to understand the number of passengers in the environment, we use two cameras to count passengers that based on the detection results of detector and association method. 

3. To achieve the goal, we built a neural network to solve the double counting problem caused by the same person appearing on two cameras.


# Usage
### Environment
There is still more to come.
### Dataset

<details open>
<summary>bb_match</summary>
	
- For bounding box predict
	


|Image Number |x center (YOLO Format) |y center (YOLO Format) |weight (YOLO Format) |height (YOLO Format) |x center (YOLO Format) |y center (YOLO Format) |weight (YOLO Format) |height (YOLO Format)
|---                    |---  |---    |---    |---    |---    |---    |---    |---
|0  |0.4983  |0.1089  |0.1934  |0.2164  |0.5142  |0.2112  |0.2892  |0.4178
|0  |0.8253  |0.2092  |0.3502  |0.4171  |0.3169  |0.1115  |0.1982  |0.2217
|1  |0.5065  |0.1082  |0.1919  |0.2151  |0.5268  |0.2086  |0.3103  |0.4158
|1  |0.8038  |0.1964  |0.3931  |0.3914  |0.3230  |0.1368  |0.1675  |0.2724
|2  |0.8131  |0.1921  |0.3746  |0.3829  |0.2507  |0.1368  |0.2751  |0.2724


</details>

There is still more to come.

### How_to_use

<details open>
<summary>bb_match</summary>
</details>

There is still more to come.

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
