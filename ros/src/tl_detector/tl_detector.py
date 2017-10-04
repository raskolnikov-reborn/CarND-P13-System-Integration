#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Int32, Bool
from geometry_msgs.msg import PoseStamped, Pose, Point
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import os

STATE_COUNT_THRESHOLD = 3


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.tl_waypoints = []
        self.has_image = False

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        '''
        /vehicle/traffic_lights helps you acquire an accurate ground truth data source for the traffic light
        classifier, providing the location and current color state of all traffic lights in the
        simulator. This state can be used to generate classified images or subbed into your solution to
        help you work on another single component of the node. This topic won't be available when
        testing your solution in real life so don't rely on it in the final submission.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)
        sub_record_gt = rospy.Subscriber('/record_training_data', Bool, self.gt_cb, queue_size=1)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint',
                                                      Int32, queue_size=1)
        self.debug_img_pub = rospy.Publisher('/image_debug_zoomed',
                                             Image, queue_size=1)

        self.stop_line_positions = self.config['stop_line_positions']

        self.bridge = CvBridge()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.prev_light_loc = None

        self.gen_train_data = False

        # data generator file count index
        self.file_index = 0

        # mappings from each state to color
        self.states_map = {0: 'red', 1: 'yellow', 2: 'green'}

        fx = self.config['camera_info']['focal_length_x']

        if fx < 10:
            self.light_classifier = TLClassifier(True)
            self.target_encoding = 'rgb8'
        else:
            self.light_classifier = TLClassifier(False)
            self.target_encoding = 'rgb8'

        rospy.spin()

    def gt_cb(self, msg):
        self.gen_train_data = msg.data

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if ((state == TrafficLight.RED) or (state == TrafficLight.YELLOW)) else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        if self.waypoints is not None:
            wp_list = self.waypoints.waypoints
        else:
            return None
        # Create variables for nearest distance and neighbour
        neighbour_index = None
        neighbour_distance = 100000.0

        # Find Neighbour
        for i in range(len(wp_list)):
            wpi = wp_list[i].pose.pose.position
            distance = math.sqrt(
                (wpi.x - pose.position.x) ** 2 + (wpi.y - pose.position.y) ** 2 + (wpi.z - pose.position.z) ** 2)
            if distance < neighbour_distance:
                neighbour_index = i
                neighbour_distance = distance

        return neighbour_index

    def project_to_image_plane(self, point_in_world, trans, rot):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # Center of the image
        # To be changed in simulator to compensate
        cx = image_width / 2
        cy = image_height / 2

        # if transform not received just send out mid point of image
        if (trans is None) or (rot is None):
            return (image_width / 2, image_height / 2)

        # Create a tf matrix
        tf_matrix = self.listener.fromTranslationRotation(trans, rot)

        # convert point_in_world to a numpy array
        pw_np = np.array([[point_in_world.x], [point_in_world.y], [point_in_world.z], [1.0]])

        # Transform to point in camera using the tf_matrix
        pc_np = np.dot(tf_matrix, pw_np)


        # get x,y and z values from the transformed point
        x_c = pc_np[2]
        y_c = pc_np[1]
        z_c = pc_np[0]

        # Convert to image co-ordinates using image params
        if fx < 10.0:
            fx = 2544
            fy = 2744
            cx = image_width / 2 - 30
            cy = image_height + 70

        u = int(-(fx / z_c) * y_c)
        v = int(-(fy / z_c) * x_c)

        # Translation to top left origin
        u = min(image_width, int(u + cx))
        v = min(image_height, int(v + cy))

        return (u, v)

    def dist_to_closest_stop_line(self):

        neighbour_distance = 1000000.0
        neighbour_index = 0
        curr_pose = self.pose.pose.position
        orientation = self.pose.pose.orientation
        for i in range(len(self.stop_line_positions)):
            light_pose = self.stop_line_positions[i]
            distance = math.sqrt(
                (light_pose[0] - curr_pose.x) ** 2 + (light_pose[1] - curr_pose.y) ** 2)
            if distance < neighbour_distance:
                neighbour_distance = distance
                neighbour_index = i

        # Check if neighbour is ahead or behind
        quat_np = (orientation.x, orientation.y, orientation.z, orientation.w)
        rpy = tf.transformations.euler_from_quaternion(quat_np)
        yaw = rpy[2]

        # Project a unit vector along orientation
        proj_distance = 1.0
        v1 = (proj_distance*math.cos(yaw), proj_distance*math.sin(yaw))
        v2 = (self.stop_line_positions[neighbour_index][0] - curr_pose.x, self.stop_line_positions[neighbour_index][1] - curr_pose.y)

        dot = np.dot(v1,v2)

        if dot < 0:
            neighbour_distance = -1

        return neighbour_distance

    def get_light_state(self, light_wp):
        """Determines the current color of the traffic light

        Args:
            light_wp (TrafficLight): index of light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = self.lights[light_wp].pose.pose.position
        if not self.has_image:
            self.prev_light_loc = None
            return TrafficLight.UNKNOWN

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, self.target_encoding)

        # Zooming etc is only needed when we need to generate training data
        if self.gen_train_data:

            distance_to_light = self.dist_to_closest_stop_line()

            if not (0 < distance_to_light < 75):
                return TrafficLight.UNKNOWN

            # get transform between pose of camera and world frame
            trans = None
            rot = None
            try:
                now = rospy.Time.now()
                self.listener.waitForTransform("/base_link",
                                               "/world", now, rospy.Duration(1.0))
                (trans, rot) = self.listener.lookupTransform("/base_link",
                                                             "/world", now)

            except (tf.Exception, tf.LookupException, tf.ConnectivityException):
                rospy.logerr("Failed to find camera to map transform")

                if (trans is None) or (rot is None):
                    return TrafficLight.UNKNOWN

            # Use TL position from the message to figure out center in the image plane

            # Offset based on observation (The z value in the message seems to be for the top of the light
            light.z -= 0.6

            x_center, y_center = self.project_to_image_plane(light, trans, rot)

            # TODO use light location to zoom in on traffic light in image

            # Create a second light point which is 0.5 * (tl_width, tl_height ) away from the center
            # Add padding to have a slightly bigger bounding box
            # it should conceptually be fine for the deep learning pipeline as
            # The Neural Network should be able to extract away the
            light2 = light
            light2.y += 0.5  # width + padding
            light2.z += 1.0  # height + padding

            # Project the corner to the image plane as well
            x_corner, y_corner = self.project_to_image_plane(light2, trans, rot)

            # Get image parameters in the local variables so that bounding boxes can be capped by image dimensions
            image_width = self.config['camera_info']['image_width']
            image_height = self.config['camera_info']['image_height']

            # Figure out box half and full heights
            box_half_width = abs(x_corner - x_center)
            box_half_height = abs(y_corner - y_center)

            # Find top left corner
            tl_x = max(0, x_center - box_half_width)
            tl_y = max(0, y_center - box_half_height)

            # Find Bottom Right corner
            br_x = min(x_center + box_half_width, image_width - 1)
            br_y = min(y_center + box_half_height, image_height - 1)

            # save the training image and annotation
            self.generate_training_data(cv_image, image_width, image_height,
                self.lights[light_wp].state, tl_x, tl_y, int(box_half_width*2),
                int(box_half_height*2))

            # Image for debug_msgs
            new_image = cv_image
            # Draw the estimated bounding box
            cv2.rectangle(new_image, (tl_x, tl_y), (br_x, br_y), (0, 255, 0), 4)

            # create the message from the debug image
            img_msg = self.bridge.cv2_to_imgmsg(new_image, self.target_encoding)

            # publish the output
            self.debug_img_pub.publish(img_msg)

            # return Ground truth to be published on the message
            # return self.lights[light_wp].state

            return TrafficLight.UNKNOWN
        else:
            # Get classification TODO: use classifier
            light_state = self.light_classifier.get_classification(cv_image)
            img_msg = self.bridge.cv2_to_imgmsg(self.light_classifier.image_np_output, self.target_encoding)
            # publish the output
            self.debug_img_pub.publish(img_msg)
            return light_state


    def get_light_state_from_list(self, light_index):
        return self.lights[light_index].state

    def generate_training_data(self, image, image_width, image_height, state,
        x, y, w, h):

        # Setup
        data_folder = "../../../sim_training_data"
        annotation_folder = "../../../sim_training_data/annotations"

        # Check if this is the first loop
        if self.file_index == 0:

            # Check to see if there's a folder for containing the data
            if os.path.isdir(data_folder):
                training_files = os.listdir(data_folder)

                # Ensure we have an annotations folder
                if 'annotations' not in training_files:
                    os.makedirs(annotation_folder)
                else:
                    training_files.remove('annotations')

                # If there is, check and see if any training images are saved
                if len(training_files) > 0:
                    training_files.sort()
                    training_files.sort(key=len)
                    last_training_file = training_files[-1]
                    rospy.logwarn("The last file was {}".format(
                        last_training_file))
                    try:
                        last_index = int(last_training_file.split(".")[0])
                        self.file_index = last_index + 1
                    except:
                        raise ValueError(
                            "Couldn't understand the naming convention of {},\
                             was it created by the data generator?".format(
                                last_training_file))
            else:
                os.makedirs(data_folder)
                os.makedirs(annotation_folder)

        # Save a training image and annotation and update the file count index
        filename = "{0}/{1}.jpg".format(data_folder,
            str(self.file_index).zfill(4))
        annotation = """
{{
  "filename" : "{0}",
  "folder" : "sim_training_data",
  "image_w_h" : [
    {1},
    {2}
  ],
  "objects" : [
    {{
      "label" : "traffic_light-{3}",
      "x_y_w_h" : [
        {4},
        {5},
        {6},
        {7}
      ]
    }}
  ]
}}
""".format("{}.jpg".format(str(self.file_index).zfill(4)), image_width,
    image_height, self.states_map[state], x, y, w, h)
        cv2.imwrite(filename, image)
        with open(annotation_folder + "/" +
            str(self.file_index).zfill(4) + ".json", "w") as saveloc:
            saveloc.write(annotation)
        self.file_index += 1
        rospy.logwarn("Took an annotated traffic light training image")

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        light_positions = self.config['stop_line_positions']
        if (self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        # TODO find the closest visible traffic light (if one exists)

        neighbour_index = None
        neighbour_distance = 100000.0
        if self.waypoints is not None:
            pose = self.waypoints.waypoints[car_position].pose.pose
        else:
            return -1, TrafficLight.UNKNOWN

        for i in range(len(self.lights)):
            lpi = self.lights[i].pose.pose.position
            # rospy.logwarn(lpi)
            distance = math.sqrt(
                (lpi.x - pose.position.x) ** 2 + (lpi.y - pose.position.y) ** 2 + (lpi.z - pose.position.z ** 2))
            if distance < neighbour_distance:
                neighbour_index = i
                neighbour_distance = distance

        light_wp = None
        if neighbour_index is not None:
            light = self.lights[neighbour_index].pose.pose.position
            light_wp = neighbour_index

        if light:
            state = self.get_light_state(light_wp)
            # state = self.get_light_state_from_list(light_wp)
            # self.generate_training_data(light, state, zoom=False)
            return light_wp, state
        # self.waypoints = None
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
