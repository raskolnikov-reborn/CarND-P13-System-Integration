
# Team CarNage-2.0
This is a submission towards the capstone for Udacity's Self Driving Car Nanodegree by Team CarNage-2.0

## Team members 
The group was comprised of three members spread across the world:

**Team Lead:** Sahil Malhotra (India) (sahilmalhotra17@gmail.com)
**Team Member 1:** Marc Puig (Barcelona) (marc.puig@gmail.com)
**Team Member 2:** Kevin Palm (United States) (kevinpalm@hotmail.com)

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

### Deep Learning Model:
We used a faster_rcnn model and google tensorflows object detection API to train two models. One for simulator and another one for Carla.

1. Sim Model: Fast RCNN pretrained on the coco dataset --> retrained on Bosch's small traffic lights data set --> Simulator generated traffic light data set. For testing generalization we only recorded images from Traffic light 1, 2, 5, 6 for training and were able to successfully detect and classify all traffic lights in the simulator during prediction

2. CarlaModel: Fast RCNN pretrained on the coco dataset --> retrained on Bosch's small traffic lights data set --> rosbag generated traffic light data set. For testing generalization we only recorded images from just_traffic_lights.bag for training and were able to successfully detect and classify all traffic lights in the loop_with_traffic_light.bag. There were a few false detections where random objects in the environment were classified as traffic lights but they were very instantaneous and did not affect the performance. To test that the entire pipeline is working we launched site.launch and published a constant current velocity of 1 m/s and set the /vehicle/dbw_enabled to true by publishing a message using rostopic pub from the command line. we validated that the pipeline was working by echoing the /vehicle/brake_cmd topic on the command line. We found that when traffic light is detected as red, brake is applied

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
3. Download our pretrained models from (https://drive.google.com/file/d/0ByU68LSPjWUPUGJJWGwwN2dteE0/view?usp=drivesdk)

4. Unzip the models.zip file 

5. copy the contents into ros/src/tl_detector/light_classification/ subdirectory of the project directory

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


