#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import math

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

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        
        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('obstacle_waypoint', Int32, self.obstacle_cb)


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

	    # Run the iterations at 10 Hz
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.iterate()
            rate.sleep()
    
    def iterate(self):
        # If the base waypoints and the current pose have been received
        if hasattr(self, 'base_waypoints') and hasattr(self, 'current_pose'):
            # Create a Standard Lane Message
            lane = Lane()
            # Set its frame and Timestamp
            lane.header.frame_id = '/world'
            lane.header.stamp = rospy.Time.now()

            # Create local variables from messages
            curr_pose = self.current_pose.pose.position
            wp_list = self.base_waypoints.waypoints

            # Create variables for nearest distance and neighbour
            neighbour_index = None
            # Set High value as default
            neighbour_distance = 100000;

            # Find Neighbour
            for i in range(len(wp_list)):
                wpi = wp_list[i].pose.pose.position
                distance = math.sqrt((wpi.x - curr_pose.x)**2 + (wpi.y - curr_pose.y)**2 + (wpi.z - curr_pose.z)**2)
                if distance < neighbour_distance:
                    neighbour_distance = distance
                    neighbour_index = i

            # Create a lookahead wps sized list for final waypoints    
            for i in range(neighbour_index, neighbour_index + LOOKAHEAD_WPS):
                # Handle Wraparound
                index = i%len(wp_list)
                wpi = wp_list[index]
                # Update Velocity (Temporary: get velocity after considering TLDC output)
                # Units: Meters/Second
                # wpi.twist.twist.linear.x = 20.0;
                lane.waypoints.append(wpi)

            self.final_waypoints_pub.publish(lane)

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
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
