import rospy
from pid import PID
import math
from yaw_controller import YawController
#GAS_DENSITY = 2.858
#ONE_MPH = 0.44704
MAX_V_MPS = 44.7 # Maximum speed in meters_per_second
BRAKE_TORQUE_SCALE = 100000
YAW_SCALE = 8.2

class Controller(object):
    def __init__(self, **kwargs):
        # TODO: Implement
        max_lat_acc = kwargs['max_lat_acc']
        max_steer_angle = kwargs['max_steer_angle']
        steer_ratio = kwargs['steer_ratio']
        wheel_base = kwargs['wheel_base']

        self.accel_limit = kwargs['accel_limit']
        self.brake_deadband = kwargs['brake_deadband']
        self.decel_limit = kwargs['decel_limit']



        # Create Variable for the last update time
        self.last_update_time = None

        self.pid_c = PID(11.2, 0.05, 0.3, -self.accel_limit, self.accel_limit)
        self.steer_pid = PID(0.8, 0.05, 0.2, -max_steer_angle/2, max_steer_angle/2)

        # Create a steering controller
        self.steer_c = YawController(wheel_base=wheel_base, steer_ratio=steer_ratio, min_speed = 0.0, max_lat_accel = max_lat_acc, max_steer_angle = max_steer_angle)

        # scale of max acceleration to deceleration
        self.brake_scale = self.decel_limit/(-self.accel_limit)
        pass

    def control(self, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        # Get the variables from the arguments
        twist_linear = kwargs['twist_command'].twist.linear
        twist_angular = kwargs['twist_command'].twist.angular

        if math.fabs(twist_linear.x) < 0.1:
            twist_linear.x = 0.
        if math.fabs(twist_angular.z) < 0.001:
            twist_angular.z = 0.

        cv_linear = kwargs['current_velocity'].twist.linear
        cv_angular = kwargs['current_velocity'].twist.angular

        steer_feedback = kwargs['steer_feedback']

        # Set the Error
        vel_err = twist_linear.x - cv_linear.x

        # Convert to fraction of v_max so pid output is directly 
        # usable as input to throttle_cmd
        vel_err = vel_err/MAX_V_MPS

        # Set the present and target values
        target_v = twist_linear.x
        present_v = cv_linear.x
        target_w = twist_angular.z

        if self.last_update_time is not None:
            # Get current time
            time = rospy.get_time()

            # Compute timestep between updates and save 
            # for next iteration
            dt = time - self.last_update_time
            self.last_update_time = time

            # if vehicle is trying to stop we want to reset the integral component
            # of the PID controllers so as to not oscillate around zero
            if present_v < 1.0:
                self.pid_c.reset()
                self.steer_pid.reset()



            # PID class returns output for throttle and 
            # brake axes as a joint forward_backward axis
            forward_axis = self.pid_c.step(vel_err, dt)
            reverse_axis = -forward_axis*(self.decel_limit/(-self.accel_limit))

            # if forward axis is positive only then give any throttle
            throttle = min(max(0.0, forward_axis), 0.33)
            # Only apply brakes if the reverse axis value is large enough to supersede the deadband
            #TODO: Figure out how this tuning will work for the real vehicle
            brake = max(0.0, reverse_axis - self.brake_deadband)

            # Convert brake to Torque value since that is what the publisher expects
            brake *= 100

            # get the steering value from the yaw controller
            steering = self.steer_c.get_steering(target_v, target_w, present_v)

            # update using steering pid loop
            steering = self.steer_pid.step(steering - steer_feedback, dt)
            # steering *= steer_ratios

            return throttle, brake, steering
        else:
            # Update the last_update time and send Zeroes tuple
            self.last_update_time = rospy.get_time()
            return 0., 0., 0.
