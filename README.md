
# Team CarNage-2.0
This is a submission towards the capstone for Udacity's Self Driving Car Nanodegree by Team CarNage-2.0

## Team members 
The group was comprised of three members spread across the world:

* **Team Lead:** Sahil Malhotra (India) (sahilmalhotra17@gmail.com)
* **Team Member 1:** Marc Puig (Barcelona, Spain) (marc.puig@gmail.com)
* **Team Member 2:** Kevin Palm (United States) (kevinpalm@hotmail.com)

## Project Pipeline
We Followed the instructions of the classroom project and completed the following modules:


#### 1. Waypoint updater
The waypoint updater subscribes to the topics of base waypoints, current_pose, current velocity, traffic_waypoint and computes the final waypoints based on the current status of each of the information elements (topics) described above. 

**Pipeline:**
* Update local variables based on messages received on topics
* Create a final_waypoints message from the base waypoints; These waypoints encode target position and velocity set by the waypoint_loader
* Based on /traffic_waypoint status. Update viewpoints to stop smoothly before the Stop lines of each waypoint
* Publish output

#### 2. DBW Node
The dbw_node is responsible for converting twist cmd to steer brake and throttle commands to be sent to simulator bridge or Carla. 

**Pipeline:**
* Initialize PID controllers to achieve target throttle, brake and steering angles
* Update variables based on messages received on topics
* Calculate error between current and target velocities and orientations
* Map the output to throttle brake and steering values
* If /vehicle/dbw_enabled is true: run a PID iteration to compute new throttle, brake and steering values
* Publish output

**Note:** The PIDs are tuned keeping in mind the lag issues we had on our development machine so they converge just before the target value of speed. This allowed us to still stay under the speed limit even in the presence of small lag spikes due to performance issues

#### 3. Traffic Light Detector Node
This node is responsible for receiving images from the camera and detecting and classifying images for traffic light status. We also repurposed the code to generate Training data from the simulator to train our deep_learning based TL Detector and Classifier described in the subsequent sections

**Pipeline (Generate Training Data):**
* Subscribe to topics and update local variables
* Using received image message and the /vehicle/traffic_lights message values, calculate the pixel co-ordinates of traffic light bounding box
* Record bounding box annotation and image to permanent storage for training the classifier
* Publish traffic light ground truth state based on /vehicle/traffic lights

**Pipeline (Prediction):**
* Load frozen model
* Subscribe to topics and update local variables
* Convert image msg to numpy array usable by model
* Predict using loaded model
* Publish traffic light state based on predicted output if score of detection > 50%

## Data Generation from Simulator
Ultimately our team employed a few thousand training images created using the Udacity system integration simulator in our final traffic light classifier. These training files were generated in the format of an unzoomed 800x600 pixel image plus an annotation JSON which detailed the current state of the captured traffic light as well as coordinates for a rectangular bounding box surrounding the light.

### Challenges
The team faced two primary challenges while creating training data from the simulator:
* Latency - lag between the graphical display and ROS framework resulted in incorrectly labeled frames or poorly drawn bounding boxes
* Zooming - the team never perfectly succeeded in using simulator x,y coordinates of the vehicle and the traffic light of interest to identify which pixels in the current frame were of interest. Our final solution was good enough for training data generation, then we employed a deep neural network to locate the traffic lights when actually driving the car.

### Overcoming Latency Issues
We tried a few approaches to overcoming the poorly labeled images and poorly drawn bounding boxes created by lag, the [most novel of which probably being to try and use outlier detection techniques out of the scikit-learn module](https://github.com/raskolnikov-reborn/CarND-P13-System-Integration/blob/master/sim_data_cleanup/Simulator%20data%20cleanup.ipynb). Ultimately we ended up:
* Combing through our ROS framework for inefficiencies to help reduce the lag in the first place (it also helped a lot that our team leader had a powerful PC to use)
* Manual data checking

### Overcoming Zooming Issues
Our team had trouble getting the neccessary coordinate and perspective transforms to work perfectly for zooming the traffic light camera into the pixels of the traffic light of interest, [which was apparently a common problem amongst teams and is still being understood.](https://discussions.udacity.com/t/focal-length-wrong/358568) In generating training data, our solution to this was:
* To 'tune' our zooming function with some baseline assumptions about size and proportion of the frame the through manual trial and error to be 'most correct'
* Creating a toggle message for when the car would capture training data. This helped prevent images from being generated while the car was too far from the next traffic light and zooming would fail
* Manual data checking

### Deep Learning Models to detect traffic lights:

We used the Tensorflow [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) to train and test models. This [recently released toolset](https://github.com/tensorflow/models) provides [pretrained models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) which are useful for out-of-the-box inference if you are interested in categories already in COCO (e.g., humans, cars, etc), and they are also useful for initializing your own models when training on novel datasets (our case). 

### Preparing inputs:

For this project we used a Faster R-CNN model, pre trained on the [COCO dataset](http://cocodataset.org) to train two models: one for the simulator and another one for Carla. Both models were retrained using the [Bosch's small traffic lights data set](https://hci.iwr.uni-heidelberg.de/node/6132) as a base, and adding extra images adhoc for each environment.

On both cases, new images (from simulator and rosbag files) were manually labeled and converted into the [TFRecord file format](https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details) to be used in Tensorflow Object Detection API. With the TFRecord files, we prepared a Object Detection pipeline and ran the training jobs. Finally, to visualize the results of the new trained models, we wrote a Jupyter notebook which loads the models and show the images with a rectangle on top of the detected traffic lights. All the source code can be found in this [repo](github.com/mpuig/traffic-lights_classifier)

### The final models:

1. Simulator Model: Faster R-CNN pretrained on the coco dataset, retrained on Bosch's small traffic lights data set and simulator generated traffic light data set. For testing generalization we only recorded images from Traffic light 1, 2, 5, 6 and we were able to successfully detect and classify all traffic lights in the simulator during prediction.

2. Carla Model: Faster R-CNN pretrained on the coco dataset, retrained on Bosch's small traffic lights data set and rosbag generated traffic light data set. For testing generalization we only used recorded images from the rosbag file `just_traffic_lights.bag`. For training and we were able to successfully detect and classify all traffic lights in the rosbag file `loop_with_traffic_light.bag`. There were a few false detections where random objects in the environment were classified as traffic lights but they were very instantaneous and did not affect the performance. To test that the entire pipeline is working we launched `site.launch`, published a constant current velocity of 1 m/s and set the `/vehicle/dbw_enabled` to `true` by publishing a message using the command `rostopic pub`. We validated that the pipeline was working by echoing the topic `/vehicle/brake_cmd`. When the system detects a traffic ligh as red, the brake is applied.

### Environment

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases/tag/v1.2).

### Evaluation
1. Clone the project repository
```bash
git clone https://github.com/raskolnikov-reborn/CarND-P13-System-Integration.git
```

2. Install python dependencies
```bash
cd CarND-P13-System-Integration
pip install -r requirements.txt
```
3. Download our pretrained models from (https://drive.google.com/file/d/0ByU68LSPjWUPUGJJWGwwN2dteE0/view?usp=sharing)

4. Unzip the models.zip file 

5. Copy the contents of models/ folder into ros/src/tl_detector/light_classification/ subdirectory of the project directory

6. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
7. Run the simulator

8. Select the lowest resolution and graphics quality

9. Press Start and you should see a connection accepted print in the terminal where roslaunch is running. If not, press escape and press start again

10. Uncheck the Checkbox titled 'manual' on the simulator output.

11. The Car should start driving while responding to traffic light status

### Real world testing
1. Launch the site.launch file
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/site.launch
```
2. Connect Carla
3. Ensure all required topics are being published
4. Enable dbw by publishing true to the /vehicle/dbw_enabled topic
5. Confirm that traffic light detection works live images



## Gratitude
1. Anthony Sarkiss for his detailed explanation on how to adapt bosch data set to the traffic light detection problem for this project: https://medium.com/@anthony_sarkis/self-driving-cars-implementing-real-time-traffic-light-detection-and-classification-in-2017-7d9ae8df1c58
2. Jeremy Shannon: For his excellent post and tweaks on the focal length problem in the simulator which helped us generate usable training data from the simulator. https://discussions.udacity.com/t/focal-length-wrong/358568
3. Davy for building a usable docker container before their was official docker support which helped us push development on laptop machines as well. https://discussions.udacity.com/t/docker-container-to-compile-and-run-project-linux-and-mac/362893

