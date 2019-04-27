"""Takeoff-Hover-Landing Combined task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask
from quad_controller_rl.agents.policy_gradients_test_combined import DDPG
#from quad_controller_rl.agents.policy_gradients_combined import DDPG

class Combined(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))
        #print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 15.0  # secs
        self.time_per_step = 5.0
        self.target_z = 10.0  # target height (z position) to reach for successful takeoff
        
        self.mode = -1

    def reset(self):
        self.mode = -1
        # Nothing to reset; just return initial condition
        return Pose(
                position=Point(0.0, 0.0, np.random.normal(0.5, 0.1)),  # drop off from a slight random height
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        state = np.array([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        # Compute reward / penalty and check if this episode is complete
        done = False
        reward = 0
        
        if self.agent is not None and isinstance(self.agent, DDPG):
            if self.mode==-1:
                self.agent.set_takeoff_mode()
                self.mode = 0
            if self.mode==0 and pose.position.z>=self.target_z:
                self.agent.set_hover_mode()
                self.hover_poss = pose.position.z
                self.mode = 1
            #if self.mode==1 and timestamp>=self.time_per_step*2:
            if self.mode==1 and timestamp>=self.time_per_step:
                self.agent.set_land_mode()
                self.mode = 2
                
        if self.mode==0: # reward up movement when taking off
            reward += pose.position.z/10.0
        if self.mode==1: # reward staying calm when hovering
            reward += abs(pose.position.z-self.hover_poss)/10.0
        if self.mode==2: # reward moving down when landing
            reward -= pose.position.z/10.0
                
        if timestamp > self.max_duration:  # agent has run out of time
            done = True                
        
        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
