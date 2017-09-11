#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Int32
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

STATE_COUNT_THRESHOLD = 3


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.tl_waypoints = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights helps you acquire an accurate ground truth data source for the traffic light
        classifier, providing the location and current color state of all traffic lights in the
        simulator. This state can be used to generate classified images or subbed into your solution to
        help you work on another single component of the node. This topic won't be available when
        testing your solution in real life so don't rely on it in the final submission.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)
        sub6 = rospy.Subscriber('/camera/image_raw', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint',
                                                      Int32, queue_size=1)
        self.debug_img_pub = rospy.Publisher('/image_debug_zoomed',
                                             Image, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        rospy.logwarn("received waypoints")

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
            light_wp = light_wp if state == TrafficLight.RED else -1
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

        rospy.logwarn("index is %d", neighbour_index)
        return neighbour_index

    def project_to_image_plane(self, point_in_world):
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

        # if transform not received just send out mid point of image
        if (trans is None) or (rot is None):
            return (image_width / 2, image_height / 2)

        # TODO Use tranform and rotation to calculate 2D position of light in image
        # Create a tf matrix
        tf_matrix = self.listener.fromTranslationRotation(trans, rot)

        # TODO: use matrix form equations for  converting between camer and world co-ordinates
        # convert point_in_world to a numpy array
        pw_np = np.array([[point_in_world.x], [point_in_world.y], [point_in_world.z], [1.0]])

        # rospy.logwarn("point in world is %s ", pw_np)


        # Transform to point in camera using the tf_matrix
        pc_np = np.dot(tf_matrix, pw_np)

        # get x,y and z values from the transformed point
        x_c = pc_np[2][0]
        y_c = pc_np[1][0]
        z_c = pc_np[0][0]

        rospy.logwarn("Transformed point is (%f,%f,%f)", x_c, y_c, z_c)

        # Convert to image co-ordinates using image params
        mu = 1000
        mv = 1000
        u = int(-(fx / z_c) * x_c * mu)
        v = int(-(fy / z_c) * y_c * mv)

        # Translation to top left origin
        u += image_width/2
        v = image_height/2 - v

        rospy.logwarn("U,V : (%d,%d)", u, v)

        return (u, v)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if (not self.has_image):
            self.prev_light_loc = None
            return False

        self.camera_image.encoding = "rgb8"
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # x, y = self.project_to_image_plane(light.pose.pose.position)
        x_center, y_center = self.project_to_image_plane(light)

        # TODO use light location to zoom in on traffic light in image

        light2 = light

        light2.y += 4.0

        light2.z += 8.0

        x_corner, y_corner = self.project_to_image_plane(light2)

        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        box_half_width = abs(x_corner - x_center)
        box_half_height = abs(y_corner - y_center)

        # Find top left corner
        tl_x = max(0, x_center)
        tl_y = max(0, y_center - box_half_height)

        # Find Bottom Right corner
        br_x = min(x_center + box_half_width, image_width - 1)
        br_y = min(y_center + box_half_height, image_height - 1)

        rospy.logwarn(" Light Bounding Box : (%d,%d):(%d,%d)", tl_x, tl_y, br_x, br_y)

        # new_image = cv_image[yc1:yc1+90, xc1-50:xc1]
        new_image = cv_image[tl_y:br_y, tl_x:br_x]

        ht, wd = new_image.shape[:2]
        rospy.logwarn("new size is (%d,%d)", wd, ht)
        cv_image = cv2.resize(new_image, (wd, ht), interpolation=cv2.INTER_CUBIC)

        img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.debug_img_pub.publish(img_msg)

        # Get classification
        return self.light_classifier.get_classification(cv_image)

    def get_light_state_from_list(self, light_index):
        return self.lights[light_index].state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        light_positions = self.config['light_positions']
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

        rospy.logwarn("light_index = %d", neighbour_index)
        light_wp = None
        if neighbour_index is not None:
            light = self.lights[neighbour_index].pose.pose.position
            light_wp = neighbour_index

        if light:
            state = self.get_light_state(light)
            # state = self.get_light_state_from_list(light_wp)
            return light_wp, state
        # self.waypoints = None
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
