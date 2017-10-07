#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from styx_msgs.msg import TrafficLightArray, TrafficLight
import math
import yaml

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.
Once you have created dbw_node, you will update this node to use the status of traffic lights too.
Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number


class WaypointUpdater ( object ):
    def __init__(self):
        rospy.init_node ( 'waypoint_updater' )

        rospy.Subscriber ( '/current_pose', PoseStamped, self.pose_cb, queue_size=1 )

        rospy.Subscriber('/current_velocity', TwistStamped, self.vel_cb, queue_size=1)
        rospy.Subscriber ( '/base_waypoints', Lane, self.waypoints_cb, queue_size=1 )

        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber ( '/traffic_waypoint', Int32, self.traffic_cb, queue_size=1 )
        rospy.Subscriber ( 'obstacle_waypoint', Int32, self.obstacle_cb, queue_size=1 )

        # Add a subscriber for /vehicle/traffic_lights
        rospy.Subscriber ( 'vehicle/traffic_lights', TrafficLightArray, self.tla_cb, queue_size=1 )

        self.final_waypoints_pub = rospy.Publisher ( 'final_waypoints', Lane, queue_size=1 )


        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.stop_line_positions = self.config['stop_line_positions']

        # TODO: Add other member variables you need below
        self.lights = []

        self.traffic_light_behaviour = False

        # Run the iterations at 10 Hz
        rate = rospy.Rate ( 10 )
        while not rospy.is_shutdown ():
            self.iterate ()
            rate.sleep ()

    def deep_copy_wp(self, waypoint):

        new_wp = Waypoint()
        new_wp.pose.pose.position.x = waypoint.pose.pose.position.x
        new_wp.pose.pose.position.y = waypoint.pose.pose.position.y
        new_wp.pose.pose.position.z = waypoint.pose.pose.position.z
        new_wp.pose.pose.orientation.x = waypoint.pose.pose.orientation.x
        new_wp.pose.pose.orientation.y = waypoint.pose.pose.orientation.y
        new_wp.pose.pose.orientation.z = waypoint.pose.pose.orientation.z
        new_wp.pose.pose.orientation.w = waypoint.pose.pose.orientation.w
        new_wp.twist.twist.linear.x = waypoint.twist.twist.linear.x
        new_wp.twist.twist.linear.y = waypoint.twist.twist.linear.y
        new_wp.twist.twist.linear.z = waypoint.twist.twist.linear.z
        new_wp.twist.twist.angular.x = waypoint.twist.twist.angular.x
        new_wp.twist.twist.angular.y = waypoint.twist.twist.angular.y
        new_wp.twist.twist.angular.z = waypoint.twist.twist.angular.z

        return new_wp

    def iterate(self):
        # If the base waypoints and the current pose have been received
        if hasattr ( self, 'base_waypoints' ) and hasattr ( self, 'current_pose' ) and hasattr(self, 'current_velocity'):
            # Create a Standard Lane Message
            lane = Lane ()
            # Set its frame and Timestamp
            lane.header.frame_id = '/world'
            lane.header.stamp = rospy.Time.now ()

            # Create local variables from messages
            curr_pose = self.current_pose.pose.position

            wp_list = self.base_waypoints.waypoints

            # Create variables for nearest distance and neighbour
            neighbour_index = None
            # Set High value as default
            neighbour_distance = 100000

            # Find Neighbour
            for i in range ( len ( wp_list ) ):
                wpi = wp_list[i].pose.pose.position
                distance = math.sqrt (
                    (wpi.x - curr_pose.x) ** 2 + (wpi.y - curr_pose.y) ** 2 + (wpi.z - curr_pose.z) ** 2 )
                if distance < neighbour_distance:
                    neighbour_distance = distance
                    neighbour_index = i

            # Create a lookahead wps sized list for final waypoints
            for i in range ( neighbour_index, neighbour_index + LOOKAHEAD_WPS ):
                # Handle Wraparound
                index = i % len(wp_list)
                wpi = self.deep_copy_wp(wp_list[index])
                lane.waypoints.append(wpi)

            light = None
            # Check traffic light status
            if hasattr(self,'traffic_waypoint'):
                if self.traffic_waypoint != -1:
                    self.traffic_light_behaviour = True
                    light = self.lights[self.traffic_waypoint]
                else:
                    self.traffic_light_behaviour = False

            # if red light is coming up, find the waypoint closest to it from the base waypoints
            if light:
                closest_wp = None
                closest_distance = 1000000.0
                for i in range ( len ( lane.waypoints ) ):
                    waypoint = lane.waypoints[i].pose.pose.position
                    light_pose = light.pose.pose.position
                    distance = math.sqrt (
                        (waypoint.x - light_pose.x) ** 2 + (waypoint.y - light_pose.y) ** 2 + (waypoint.z - light_pose.z) ** 2 )

                    if distance < closest_distance:
                        closest_wp = i
                        closest_distance = distance

                braking_tolerance = 10
                # rospy.logwarn(" closest waypoint is %d , %f meters away", closest_wp, closest_distance)

                # Get Distance between waypoints closest to Car and Light
                distance = self.distance(lane.waypoints, 0, closest_wp)

                # find distance between stop line and traffic light
                if light:
                    light_pose = light.pose.pose.position
                    stop_line = self.stop_line_positions[self.traffic_waypoint]
                    light_to_line_dist = math.sqrt(
                        (light_pose.x - stop_line[0]) ** 2 + (light_pose.y - stop_line[1]) ** 2)

                    # If not behind and not crossed already
                    crossed_light = False
                    if distance < light_to_line_dist:
                        crossed_light = True


                    # define a max permissible deceleration for comfort
                    max_permissible_deceleration = 1.0

                    # define a min deceleration to avoid local minima
                    min_safety_deceleration = 1.0

                    # Find Min Deceleration Distance using v2 = u2 + 2as
                    min_braking_distance = (self.current_velocity**2)/(2*max_permissible_deceleration)

                    # add the distance between the traffic light and the stop line (since we want to stop before the
                    # latter)
                    min_braking_distance += light_to_line_dist

                    braking_waypoints = 0
                    # Find num of waypoints on the current path that are needed to travel that distance
                    for i in range (len(lane.waypoints)):
                        if self.distance (lane.waypoints, 0, i) > min_braking_distance:
                            braking_waypoints = i
                            break

                    if min_braking_distance < 150 and crossed_light is False:
                        # safety buffer
                        braking_clearance = 5
                        braking_start_wp = max( 0, closest_wp - braking_waypoints - braking_clearance )

                        deceleration = max(min_safety_deceleration, self.current_velocity/(braking_waypoints * 1))

                        for i in range ( braking_start_wp, LOOKAHEAD_WPS):
                            dec_step = i - braking_start_wp + 1
                            lane.waypoints[i].twist.twist.linear.x  = self.current_velocity - (deceleration * dec_step)
                            lane.waypoints[i].twist.twist.linear.x = max(0.00, lane.waypoints[i].twist.twist.linear.x - 1.0)

            self.final_waypoints_pub.publish ( lane )

        pass

    def pose_cb(self, msg):
        self.current_pose = msg
        pass

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        pass

    def traffic_cb(self, msg):
        self.traffic_waypoint = msg.data
        pass

    def obstacle_cb(self, msg):
        self.obstacle_waypoint = msg.data
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt ( (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2 )
        for i in range ( wp1, wp2 + 1 ):
            dist += dl ( waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position )
            wp1 = i
        return dist

    def tla_cb(self, msg):
        self.lights = msg.lights

    def vel_cb(self, msg):
        self.current_velocity = msg.twist.linear.x


if __name__ == '__main__':
    try:
        WaypointUpdater ()
    except rospy.ROSInterruptException:
        rospy.logerr ( 'Could not start waypoint updater node.' )