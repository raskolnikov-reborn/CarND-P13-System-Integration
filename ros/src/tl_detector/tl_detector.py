#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
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

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights helps you acquire an accurate ground truth data source for the traffic light
        classifier, providing the location and current color state of all traffic lights in the
        simulator. This state can be used to generate classified images or subbed into your solution to
        help you work on another single component of the node. This topic won't be available when
        testing your solution in real life so don't rely on it in the final submission.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/camera/image_raw', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

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
            distance = math.sqrt((wpi.x - pose.position.x)**2 + (wpi.y - pose.position.y)**2 + (wpi.z - pose.position.z)**2)
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
            
        #TODO Use tranform and rotation to calculate 2D position of light in image
        # Create a tf matrix
        tf_matrix = self.listener.fromTranslationRotation(trans,rot)

        if (trans is None) or (rot is None):
            return (0,0)
        else:
            rospy.logwarn("trans and rot are updated")


		# TODO: use matrix form equations for  converting between camer and world co-ordinates
        # convert point_in_world to a numpy array
        pw_np = np.array([[point_in_world[0]], [point_in_world[1]],[5.0],[1.0]])

        # Transform to point in camera using the tf_matrix
        pc_np = np.dot(tf_matrix,pw_np)

        # get x,y and z values from the transformed point
        x_c = pc_np[0][0]
        y_c = pc_np[1][0]
        z_c = pc_np[2][0]

        rospy.logwarn("Transformed point is (%d,%d,%d)",x_c,y_c,z_c)

        rospy.logwarn("Focal Length is (%f,%f)",fx,fy)

        # Convert to image co-ordinates using image params
        u = int( -(fx/x_c) * y_c)
        if (u > image_width):
            u = image_width
        v = int( -(fy/x_c)*z_c)
        if (v > image_height):
            v = image_height

        return (u, v)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        self.camera_image.encoding = "rgb8"
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #x, y = self.project_to_image_plane(light.pose.pose.position)
        x, y = self.project_to_image_plane(light)

        #TODO use light location to zoom in on traffic light in image
        height_band_pixels = 20
        width_band_pixels = 20

        new_image = cv_image[y - height_band_pixels:y + height_band_pixels, x - width_band_pixels:x + width_band_pixels] 
        ht,wd = new_image.shape[:2]
        rospy.logwarn("new size is (%d,%d)",wd,ht)
        cv_image = cv2.resize(new_image,(wd,ht),interpolation = cv2.INTER_CUBIC)

        
        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        light_positions = self.config['light_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        #TODO find the closest visible traffic light (if one exists)

        neighbour_index = None
        neighbour_distance = 100000.0
        if self.waypoints is not None:
            pose = self.waypoints.waypoints[car_position].pose.pose
        else:
            return -1, TrafficLight.UNKNOWN

        for i in range(len(light_positions)):
            lpi = light_positions[i]
            # rospy.logwarn(lpi)
            distance = math.sqrt((lpi[0] - pose.position.x)**2 + (lpi[1] - pose.position.y)**2 )
            if distance < neighbour_distance:
                neighbour_index = i
                neighbour_distance = distance
        rospy.logwarn("light_index = %d",neighbour_index) 
        if neighbour_index is not None:
            light = light_positions[neighbour_index]
            light_wp = neighbour_index

        if light:
            state = self.get_light_state(light)
            return light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
